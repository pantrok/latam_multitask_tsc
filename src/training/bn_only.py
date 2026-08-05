from __future__ import annotations

"""
The fourth cell of the frozen-head / recalibrated-stats factorial that
finetune.py's "head" and "head_bn_eval" modes and random_backbone.py's C5
leave empty: BatchNorm statistics recalibrated on the target dataset, with
the head (and every other parameter) left exactly as the joint checkpoint
produced it -- never retrained.

No gradient is computed at any point. A dataset's training batches are
passed through the model in train() mode purely so BatchNorm's running
mean/var update via the forward pass itself; there is no optimizer and no
backward call. This isolates whether recalibrating normalization statistics
alone -- without also adapting the head -- recovers any accuracy, holding
constant the same number of passes over the data that finetune.py's "head"
mode spends epochs on.
"""

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.data.ucr_io import class_weights_from_labels, load_ucr_dataset, make_loader, split_train_validation
from src.training.finetune import _build_model, _dataset_num_classes_and_channels, load_joint_baseline
from src.training.train_multitask import evaluate_dataset, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recalibrate BatchNorm stats only, head frozen (factorial cell 4).")
    p.add_argument("--model", choices=["cnn", "lstm"], required=True)
    p.add_argument("--dataset-names", type=str, nargs="+", required=True)
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    p.add_argument("--joint-runs-dir", type=str, default=None, help="Defaults to artifacts/runs/<model>")
    p.add_argument("--joint-results-dir", type=str, default="artifacts/results")
    p.add_argument("--results-dir", type=str, default="artifacts/results")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 63, 84, 105, 126, 147])
    p.add_argument("--epochs", type=int, default=10, help="Forward-pass budget, same as C3-head's epoch budget.")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def bn_only_one(
    model_name: str, seed: int, dataset_name: str, args: argparse.Namespace, device: str,
    num_classes_by_dataset: dict[str, int], joint_state: dict, base_test_acc: float | None,
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
    for param in model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(y_tr, num_classes).to(device))

    best_val_loss = float("inf")
    best_state = None
    best_pass = -1
    patience_counter = 0
    passes_done = 0

    for pass_i in range(1, args.epochs + 1):
        model.train()
        with torch.no_grad():
            for xb, _yb in train_loader:
                model(xb.to(device), dataset_name)

        val_metrics = evaluate_dataset(model, val_loader, criterion, device, dataset_name)
        passes_done = pass_i
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_pass = pass_i
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"BN-only recalibration produced no valid state for {dataset_name} (seed {seed}).")
    model.load_state_dict(best_state)

    test_metrics = evaluate_dataset(model, test_loader, criterion, device, dataset_name)
    delta = None if base_test_acc is None else test_metrics["acc"] - base_test_acc

    return {
        "model": model_name, "seed": seed, "dataset": dataset_name,
        "passes_done": passes_done, "best_pass": best_pass,
        "test_acc": test_metrics["acc"], "test_f1_macro": test_metrics["f1_macro"],
        "base_test_acc": base_test_acc, "delta_vs_joint": delta,
    }


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    joint_runs_dir = Path(args.joint_runs_dir) if args.joint_runs_dir else Path("artifacts/runs") / args.model
    results_dir = Path(args.results_dir)

    joint_baseline = load_joint_baseline(Path(args.joint_results_dir), args.model)

    num_classes_by_dataset: dict[str, int] = {}
    in_channels: int | None = None
    for name in args.dataset_names:
        nc, ch = _dataset_num_classes_and_channels(args.dataset_root, name)
        num_classes_by_dataset[name] = nc
        in_channels = ch if in_channels is None else in_channels

    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    summary_path = results_dir / "bn_only_runs_summary.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() else None

    untouched = (
        existing[existing["model"] != args.model] if existing is not None and len(existing) > 0 else None
    )

    def _flush() -> None:
        this_run_df = pd.DataFrame(all_results)
        runs_df = pd.concat([untouched, this_run_df], ignore_index=True) if untouched is not None else this_run_df
        runs_df.to_csv(summary_path, index=False)

    for seed in args.seeds:
        ckpt_path = joint_runs_dir / f"seed_{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"[warn] {ckpt_path} not found -- has train-{args.model} finished this seed? Skipping.")
            continue
        joint_state = torch.load(ckpt_path, map_location=device, weights_only=True)

        for dataset_name in args.dataset_names:
            if existing is not None and not args.force:
                already = existing[(existing["model"] == args.model) & (existing["seed"] == seed) &
                                    (existing["dataset"] == dataset_name)]
                if len(already) > 0:
                    all_results.append(already.iloc[0].to_dict())
                    continue

            result = bn_only_one(
                args.model, seed, dataset_name, args, device,
                num_classes_by_dataset, joint_state, joint_baseline.get(dataset_name),
            )
            all_results.append(result)
            print(f"[{args.model} bn-only] seed={seed} dataset={dataset_name} "
                  f"test_acc={result['test_acc']:.4f} (joint was {result['base_test_acc']})")
            _flush()

    if not all_results:
        raise RuntimeError(
            f"No results for model={args.model} -- every seed was skipped (no joint "
            f"checkpoint found under {joint_runs_dir}). Check --joint-runs-dir / "
            f"--artifacts-dir and that train-{args.model} has actually finished."
        )

    _flush()
    this_model_df = pd.DataFrame(all_results)

    per_dataset = (
        this_model_df.groupby("dataset")
        .agg(acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
             acc_min=("test_acc", "min"), acc_max=("test_acc", "max"),
             f1_mean=("test_f1_macro", "mean"), f1_std=("test_f1_macro", "std"),
             delta_vs_joint_mean=("delta_vs_joint", "mean"), n_seeds=("seed", "nunique"))
        .reset_index()
    )
    out_path = results_dir / f"bn_only_{args.model}.csv"
    per_dataset.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
