from __future__ import annotations

"""
Third condition: shared backbone (joint training) -> light per-dataset
fine-tune -> evaluate.

Each of the 8 joint seeds' best_model.pt is loaded once, then fine-tuned
independently per dataset (starting fresh from the same joint checkpoint
each time -- fine-tuning one dataset must not leak into the next). Two
modes:

  full        unfreeze the whole model (backbone + that dataset's head),
              train on that dataset alone for a few epochs at a reduced
              learning rate. Answers "how much of the joint-training
              accuracy cost is recoverable with a cheap per-dataset
              touch-up."
  head-only   freeze everything except that dataset's head. Isolates
              whether the shared features are already good and only
              per-task calibration was missing, as opposed to full's
              answer which blurs into "how close can a lightly-adapted
              shared model get to a specialist."
  head_bn_eval  same as head-only, but BatchNorm layers in the frozen
              backbone are kept in eval() throughout, so their running
              statistics are not updated by the target dataset's batches
              either. head-only alone freezes weights (requires_grad=False)
              but leaves the model in train() mode, which still lets
              BatchNorm's running mean/var drift per dataset.

Kept as clearly separate result files rather than picking one mode
silently -- run_all.py's default is "full", "both" is one flag away.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW

from src.data.ucr_io import class_weights_from_labels, load_ucr_dataset, make_loader, split_train_validation
from src.models.cnn1d_multiscale_residual import CNN1DMultiescalaResidualMultiHead
from src.models.lstm_baseline import LSTMMultitarea
from src.training.train_multitask import evaluate_dataset, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a joint checkpoint per dataset.")
    p.add_argument("--model", choices=["cnn", "lstm"], required=True)
    p.add_argument("--dataset-names", type=str, nargs="+", required=True)
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    p.add_argument("--joint-runs-dir", type=str, default=None, help="Defaults to artifacts/runs/<model>")
    p.add_argument("--joint-results-dir", type=str, default="artifacts/results",
                    help="Where the joint run's per-dataset test accuracy lives, for the before/after delta.")
    p.add_argument("--output-dir", type=str, default=None, help="Defaults to artifacts/runs/<model>_finetuned")
    p.add_argument("--results-dir", type=str, default="artifacts/results")
    p.add_argument("--mode", choices=["full", "head", "both", "head_bn_eval"], default="full")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 63, 84, 105, 126, 147])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=2e-4,
                    help="Default is 1/10th of the joint training lr (2e-3) -- standard fine-tuning practice.")
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _build_model(model_name: str, in_channels: int, num_classes_by_dataset: dict[str, int]):
    if model_name == "cnn":
        return CNN1DMultiescalaResidualMultiHead(in_channels=in_channels, num_classes_by_dataset=num_classes_by_dataset)
    return LSTMMultitarea(in_channels=in_channels, num_classes_by_dataset=num_classes_by_dataset)


def _dataset_num_classes_and_channels(dataset_root: str, dataset_name: str) -> tuple[int, int]:
    x_train, y_train, x_test, y_test = load_ucr_dataset(dataset_root, dataset_name)
    num_classes = int(np.unique(np.concatenate([y_train, y_test])).shape[0])
    return num_classes, int(x_train.shape[1])


def _set_trainable(model: nn.Module, dataset_name: str, mode: str) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = mode == "full"
    for param in model.heads[dataset_name].parameters():
        param.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def _freeze_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def finetune_one(
    model_name: str, seed: int, dataset_name: str, args: argparse.Namespace, device: str,
    num_classes_by_dataset: dict[str, int], joint_state: dict, base_test_acc: float | None, mode: str,
) -> dict:
    x_train, y_train, x_test, y_test = load_ucr_dataset(args.dataset_root, dataset_name)
    x_tr, x_val, y_tr, y_val = split_train_validation(x_train, y_train, args.val_size, seed)
    in_channels = int(x_tr.shape[1])
    num_classes = num_classes_by_dataset[dataset_name]

    loader_kwargs = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    train_loader = make_loader(x_tr, y_tr, args.batch_size, shuffle=True, seed=seed, **loader_kwargs)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False, seed=seed, **loader_kwargs)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=seed, **loader_kwargs)

    model = _build_model(model_name, in_channels, num_classes_by_dataset).to(device)
    model.load_state_dict(copy.deepcopy(joint_state))

    trainable = _set_trainable(model, dataset_name, "full" if mode == "full" else "head")
    bn_eval = mode == "head_bn_eval"
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(y_tr, num_classes).to(device))
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if bn_eval:
            _freeze_batchnorm(model)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, dataset_name)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_dataset(model, val_loader, criterion, device, dataset_name)
        epochs_trained = epoch
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Fine-tuning produced no valid state for {dataset_name} (seed {seed}, mode {mode}).")
    model.load_state_dict(best_state)

    test_metrics = evaluate_dataset(model, test_loader, criterion, device, dataset_name)
    delta = None if base_test_acc is None else test_metrics["acc"] - base_test_acc

    return {
        "model": model_name, "mode": mode, "seed": seed, "dataset": dataset_name,
        "epochs_trained": epochs_trained, "best_epoch": best_epoch,
        "test_acc": test_metrics["acc"], "test_f1_macro": test_metrics["f1_macro"],
        "base_test_acc": base_test_acc, "delta_vs_joint": delta,
    }


def load_joint_baseline(joint_results_dir: Path, model_name: str) -> dict[str, float]:
    name = "per_dataset_results.csv" if model_name == "cnn" else "per_dataset_lstm.csv"
    path = joint_results_dir / name
    if not path.exists():
        print(f"[warn] {path} not found -- delta_vs_joint will be null for every result.")
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["dataset"], df["acc_mean"]))


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    joint_runs_dir = Path(args.joint_runs_dir) if args.joint_runs_dir else Path("artifacts/runs") / args.model
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/runs") / f"{args.model}_finetuned"
    results_dir = Path(args.results_dir)
    modes = ["full", "head"] if args.mode == "both" else [args.mode]

    joint_baseline = load_joint_baseline(Path(args.joint_results_dir), args.model)

    num_classes_by_dataset: dict[str, int] = {}
    in_channels: int | None = None
    for name in args.dataset_names:
        nc, ch = _dataset_num_classes_and_channels(args.dataset_root, name)
        num_classes_by_dataset[name] = nc
        in_channels = ch if in_channels is None else in_channels

    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    summary_path = results_dir / "finetune_runs_summary.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() else None

    # Rows for any (model, mode) this invocation isn't touching -- preserved
    # as-is and rewritten alongside all_results on every flush, so a run for
    # one model/mode never erases another's results in this shared file.
    untouched = (
        existing[~((existing["model"] == args.model) & (existing["mode"].isin(modes)))]
        if existing is not None and len(existing) > 0 else None
    )

    def _flush() -> None:
        # Written after every newly-computed result, not just once at the
        # end -- a Colab disconnect mid-run used to lose the entire
        # invocation's progress since nothing was written until run()
        # returned.
        this_run_df = pd.DataFrame(all_results)
        runs_df = pd.concat([untouched, this_run_df], ignore_index=True) if untouched is not None else this_run_df
        runs_df.to_csv(summary_path, index=False)

    for mode in modes:
        for seed in args.seeds:
            ckpt_path = joint_runs_dir / f"seed_{seed}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"[warn] {ckpt_path} not found -- has train-{args.model} finished this seed? Skipping.")
                continue
            joint_state = torch.load(ckpt_path, map_location=device, weights_only=True)

            for dataset_name in args.dataset_names:
                if existing is not None and not args.force:
                    already = existing[(existing["model"] == args.model) & (existing["mode"] == mode) &
                                        (existing["seed"] == seed) & (existing["dataset"] == dataset_name)]
                    if len(already) > 0:
                        all_results.append(already.iloc[0].to_dict())
                        continue

                result = finetune_one(
                    args.model, seed, dataset_name, args, device, num_classes_by_dataset,
                    joint_state, joint_baseline.get(dataset_name), mode,
                )
                all_results.append(result)
                print(f"[{args.model} finetune-{mode}] seed={seed} dataset={dataset_name} "
                      f"test_acc={result['test_acc']:.4f} (joint was {result['base_test_acc']})")
                _flush()

    if not all_results:
        raise RuntimeError(
            f"No results for model={args.model} -- every seed was skipped (no joint "
            f"checkpoint found under {joint_runs_dir}). Check --joint-runs-dir / "
            f"--artifacts-dir and that train-{args.model} has actually finished."
        )

    _flush()
    this_model_df = pd.DataFrame(all_results)  # all_results is already only args.model's rows

    for mode in modes:
        mode_df = this_model_df[this_model_df["mode"] == mode]
        per_dataset = (
            mode_df.groupby("dataset")
            .agg(acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
                 acc_min=("test_acc", "min"), acc_max=("test_acc", "max"),
                 f1_mean=("test_f1_macro", "mean"), f1_std=("test_f1_macro", "std"),
                 delta_vs_joint_mean=("delta_vs_joint", "mean"), n_seeds=("seed", "nunique"))
            .reset_index()
        )
        out_path = results_dir / f"finetune_{mode}_{args.model}.csv"
        per_dataset.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    print(f"Wrote {summary_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
