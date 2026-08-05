from __future__ import annotations

"""
Fourth condition: per-dataset models trained from RANDOM initialization, same
architecture/epoch budget/seeds as C3 (finetune.py), no shared backbone, no
joint pretraining at all. This isolates whether the fine-tuning recovery is
really about the joint-trained features, or just about spending more epochs
of gradient descent on a clean single-task objective (He et al., ICCV 2019 --
training from scratch can catch up to pretrained models given enough
iterations).

Deliberately reuses as much of finetune.py's per-dataset training loop as
possible (same data split, same early-stopping criterion, same evaluation),
changing only what the control requires: no checkpoint to load, and nothing
frozen -- every parameter is trainable from step one.

Two learning rates are supported, not one: C3 fine-tunes at 2e-4 (a tenth of
the joint lr, appropriate when starting from already-trained weights), and
using that same low lr for C4 would leave a from-scratch model undertrained
for an unrelated reason, biasing the comparison unfairly. The joint config's
lr (1.617e-3) is the primary number; 2e-4 is also recorded for comparison
rather than dropped.
"""

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from src.data.ucr_io import class_weights_from_labels, load_ucr_dataset, make_loader, split_train_validation
from src.training.finetune import _build_model, _dataset_num_classes_and_channels, load_joint_baseline
from src.training.train_multitask import evaluate_dataset, resolve_device

PRIMARY_LR = 1.617e-3  # the joint config's lr -- natural starting point from scratch
SECONDARY_LR = 2e-4    # C3's fine-tune lr, recorded for comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-dataset models from random init (C4 control).")
    p.add_argument("--model", choices=["cnn", "lstm"], required=True)
    p.add_argument("--dataset-names", type=str, nargs="+", required=True)
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    p.add_argument("--joint-results-dir", type=str, default="artifacts/results",
                    help="Where the joint run's per-dataset test accuracy lives, for delta_vs_joint.")
    p.add_argument("--results-dir", type=str, default="artifacts/results")
    p.add_argument("--lr-mode", choices=["primary", "secondary", "both"], default="both",
                    help=f"primary={PRIMARY_LR} (joint lr, recommended), secondary={SECONDARY_LR} "
                         "(matches C3's fine-tune lr). Default 'both' reports each rather than "
                         "picking the more favorable one silently.")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 63, 84, 105, 126, 147])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def scratch_one(
    model_name: str, seed: int, dataset_name: str, args: argparse.Namespace, device: str,
    num_classes_by_dataset: dict[str, int], base_test_acc: float | None, lr: float, lr_label: str,
) -> dict:
    x_train, y_train, x_test, y_test = load_ucr_dataset(args.dataset_root, dataset_name)
    x_tr, x_val, y_tr, y_val = split_train_validation(x_train, y_train, args.val_size, seed)
    in_channels = int(x_tr.shape[1])

    loader_kwargs = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    train_loader = make_loader(x_tr, y_tr, args.batch_size, shuffle=True, seed=seed, **loader_kwargs)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False, seed=seed, **loader_kwargs)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=seed, **loader_kwargs)

    # Random init -- unlike finetune.py, there is no joint checkpoint to load
    # and nothing is frozen. torch's own seeding (set by the caller/run_all.py
    # convention) governs the init; this function does not reseed internally
    # so seed sweeps produce genuinely different random inits, same as the
    # joint run's own seed handling.
    torch.manual_seed(seed)
    model = _build_model(model_name, in_channels, num_classes_by_dataset).to(device)
    trainable = list(model.parameters())

    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(y_tr, num_classes_by_dataset[dataset_name]).to(device))
    optimizer = AdamW(trainable, lr=lr, weight_decay=args.weight_decay)

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
        raise RuntimeError(f"Scratch training produced no valid state for {dataset_name} (seed {seed}, lr {lr}).")
    model.load_state_dict(best_state)

    test_metrics = evaluate_dataset(model, test_loader, criterion, device, dataset_name)
    delta = None if base_test_acc is None else test_metrics["acc"] - base_test_acc

    return {
        "model": model_name, "lr": lr, "lr_label": lr_label, "seed": seed, "dataset": dataset_name,
        "epochs_trained": epochs_trained, "best_epoch": best_epoch,
        "test_acc": test_metrics["acc"], "test_f1_macro": test_metrics["f1_macro"],
        "base_test_acc": base_test_acc, "delta_vs_joint": delta,
    }


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    results_dir = Path(args.results_dir)
    lr_variants = (
        [("primary", PRIMARY_LR)] if args.lr_mode == "primary" else
        [("secondary", SECONDARY_LR)] if args.lr_mode == "secondary" else
        [("primary", PRIMARY_LR), ("secondary", SECONDARY_LR)]
    )

    joint_baseline = load_joint_baseline(Path(args.joint_results_dir), args.model)

    num_classes_by_dataset: dict[str, int] = {}
    for name in args.dataset_names:
        nc, _ch = _dataset_num_classes_and_channels(args.dataset_root, name)
        num_classes_by_dataset[name] = nc

    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    summary_path = results_dir / "scratch_runs_summary.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() else None

    # Rows for any (model, lr_label) this invocation isn't touching --
    # preserved as-is and rewritten alongside all_results on every flush, so
    # a run for one model/lr_label never erases another's results in this
    # shared file.
    touched_labels = [label for label, _lr in lr_variants]
    untouched = (
        existing[~((existing["model"] == args.model) & (existing["lr_label"].isin(touched_labels)))]
        if existing is not None and len(existing) > 0 else None
    )

    def _flush() -> None:
        # Written after every newly-computed result, not just once at the
        # end -- a Colab disconnect mid-run used to lose the entire
        # invocation's progress since nothing was written until run()
        # returned. Cheap: at most a few hundred short rows.
        this_run_df = pd.DataFrame(all_results)
        runs_df = pd.concat([untouched, this_run_df], ignore_index=True) if untouched is not None else this_run_df
        runs_df.to_csv(summary_path, index=False)

    for lr_label, lr in lr_variants:
        for seed in args.seeds:
            for dataset_name in args.dataset_names:
                if existing is not None and not args.force:
                    already = existing[(existing["model"] == args.model) & (existing["lr_label"] == lr_label) &
                                        (existing["seed"] == seed) & (existing["dataset"] == dataset_name)]
                    if len(already) > 0:
                        all_results.append(already.iloc[0].to_dict())
                        continue

                result = scratch_one(
                    args.model, seed, dataset_name, args, device,
                    num_classes_by_dataset, joint_baseline.get(dataset_name), lr, lr_label,
                )
                all_results.append(result)
                print(f"[{args.model} scratch-{lr_label}] seed={seed} dataset={dataset_name} "
                      f"test_acc={result['test_acc']:.4f}")
                _flush()

    if not all_results:
        raise RuntimeError(
            f"No results for model={args.model} -- check --dataset-names and --seeds "
            f"were passed correctly; nothing was actually run."
        )

    _flush()
    this_model_df = pd.DataFrame(all_results)  # all_results is already only args.model's rows

    for lr_label, _lr in lr_variants:
        lr_df = this_model_df[this_model_df["lr_label"] == lr_label]
        per_dataset = (
            lr_df.groupby("dataset")
            .agg(acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
                 acc_min=("test_acc", "min"), acc_max=("test_acc", "max"),
                 f1_mean=("test_f1_macro", "mean"), f1_std=("test_f1_macro", "std"),
                 delta_vs_joint_mean=("delta_vs_joint", "mean"), n_seeds=("seed", "nunique"))
            .reset_index()
        )
        # Primary lr keeps the plain filename (scratch_cnn.csv); secondary
        # gets an explicit suffix so neither silently overwrites the other.
        suffix = "" if lr_label == "primary" else "_lr2e-4"
        out_path = results_dir / f"scratch_{args.model}{suffix}.csv"
        per_dataset.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    print(f"Wrote {summary_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
