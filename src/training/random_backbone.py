from __future__ import annotations

"""
Fifth condition: backbone randomly initialized AND frozen -- never trained on
anything, joint corpus or otherwise -- with only the 30 per-dataset heads
adapted on top of it. Same protocol as C3-head (finetune.py, mode="head"):
10 epochs, lr 2e-4, patience 5, same 8 seeds, same splits, same checkpoint
selection.

C3-head alone cannot separate two explanations for its recovery: the shared
backbone learned useful features during joint training, or simply having
been through *any* training process helps (weight-space initialization,
batch-normalization statistics) regardless of what that training saw. This
condition holds the second factor at zero -- the backbone never trains at
all -- so a head trained on top of it isolates what a frozen head-only
adaptation gets from random convolutional features alone. Random
convolutional kernels plus a trained linear classifier is exactly the
mechanism behind ROCKET-family methods, so this is also a natural point of
comparison against that literature.

One random backbone per seed, reused frozen across all 30 datasets within
that seed -- not reinitialized per dataset -- so the experimental unit
matches C3-head exactly: seed -> one backbone -> 30 head adaptations.
"""

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from src.data.ucr_io import class_weights_from_labels, load_ucr_dataset, make_loader, split_train_validation
from src.training.finetune import _build_model, _dataset_num_classes_and_channels, _set_trainable, load_joint_baseline
from src.training.train_multitask import evaluate_dataset, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen random backbone + trained heads (C5 control).")
    p.add_argument("--model", choices=["cnn", "lstm"], required=True)
    p.add_argument("--dataset-names", type=str, nargs="+", required=True)
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    p.add_argument("--joint-results-dir", type=str, default="artifacts/results",
                    help="Where the joint run's per-dataset test accuracy lives, for the before/after delta.")
    p.add_argument("--results-dir", type=str, default="artifacts/results")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 63, 84, 105, 126, 147])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=2e-4,
                    help="Same as C3-head's fine-tune lr, for an exact protocol match.")
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def random_backbone_one(
    model_name: str, seed: int, dataset_name: str, args: argparse.Namespace, device: str,
    num_classes_by_dataset: dict[str, int], base_state: dict, base_test_acc: float | None,
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
    model.load_state_dict(copy.deepcopy(base_state))

    trainable = _set_trainable(model, dataset_name, "head")
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(y_tr, num_classes).to(device))
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
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
        raise RuntimeError(f"Head training produced no valid state for {dataset_name} (seed {seed}).")
    model.load_state_dict(best_state)

    test_metrics = evaluate_dataset(model, test_loader, criterion, device, dataset_name)
    delta = None if base_test_acc is None else test_metrics["acc"] - base_test_acc

    return {
        "model": model_name, "seed": seed, "dataset": dataset_name,
        "epochs_trained": epochs_trained, "best_epoch": best_epoch,
        "test_acc": test_metrics["acc"], "test_f1_macro": test_metrics["f1_macro"],
        "base_test_acc": base_test_acc, "delta_vs_joint": delta,
    }


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
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
    summary_path = results_dir / "random_backbone_runs_summary.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() else None

    # Rows for any other model this invocation isn't touching -- preserved
    # as-is and rewritten alongside all_results on every flush, so a run for
    # one model never erases the other's results in this shared file.
    untouched = (
        existing[existing["model"] != args.model] if existing is not None and len(existing) > 0 else None
    )

    def _flush() -> None:
        this_run_df = pd.DataFrame(all_results)
        runs_df = pd.concat([untouched, this_run_df], ignore_index=True) if untouched is not None else this_run_df
        runs_df.to_csv(summary_path, index=False)

    for seed in args.seeds:
        torch.manual_seed(seed)
        base_model = _build_model(args.model, in_channels, num_classes_by_dataset).to(device)
        base_state = copy.deepcopy(base_model.state_dict())
        del base_model

        for dataset_name in args.dataset_names:
            if existing is not None and not args.force:
                already = existing[(existing["model"] == args.model) & (existing["seed"] == seed) &
                                    (existing["dataset"] == dataset_name)]
                if len(already) > 0:
                    all_results.append(already.iloc[0].to_dict())
                    continue

            result = random_backbone_one(
                args.model, seed, dataset_name, args, device,
                num_classes_by_dataset, base_state, joint_baseline.get(dataset_name),
            )
            all_results.append(result)
            print(f"[{args.model} random-backbone] seed={seed} dataset={dataset_name} "
                  f"test_acc={result['test_acc']:.4f} (joint was {result['base_test_acc']})")
            _flush()

    if not all_results:
        raise RuntimeError(
            f"No results for model={args.model} -- check --dataset-names and --seeds "
            f"were passed correctly; nothing was actually run."
        )

    _flush()
    this_model_df = pd.DataFrame(all_results)  # all_results is already only args.model's rows

    per_dataset = (
        this_model_df.groupby("dataset")
        .agg(acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
             acc_min=("test_acc", "min"), acc_max=("test_acc", "max"),
             f1_mean=("test_f1_macro", "mean"), f1_std=("test_f1_macro", "std"),
             delta_vs_joint_mean=("delta_vs_joint", "mean"), n_seeds=("seed", "nunique"))
        .reset_index()
    )
    out_path = results_dir / f"random_backbone_{args.model}.csv"
    per_dataset.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
