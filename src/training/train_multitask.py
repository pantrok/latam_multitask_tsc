from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from src.data.ucr_io import class_weights_from_labels, load_ucr_dataset, make_loader, split_train_validation
from src.models.cnn1d_multiscale_residual import CNN1DMultiescalaResidualMultiHead
from src.models.lstm_baseline import LSTMMultitarea

MODEL_BUILDERS = {"cnn": CNN1DMultiescalaResidualMultiHead, "lstm": LSTMMultitarea}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one shared backbone (CNN or LSTM) with one head per UCR dataset.")
    parser.add_argument("--model", choices=["cnn", "lstm"], required=True)
    parser.add_argument("--dataset-names", type=str, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Defaults to artifacts/runs/<model>")
    parser.add_argument("--results-dir", type=str, default="artifacts/results")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler", choices=["onecycle", "cosine", "none"], default="cosine")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="Max grad norm; <=0 disables clipping.")
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 63, 84, 105, 126, 147])
    parser.add_argument("--deterministic", action="store_true",
                         help="torch.use_deterministic_algorithms(True); may error on ops without a deterministic kernel.")
    parser.add_argument("--force", action="store_true", help="Retrain seeds even if a checkpoint already exists.")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1,
                         help="Save a resumable mid-training checkpoint every N epochs (default: every epoch). "
                              "A Colab disconnect mid-seed resumes from the last saved epoch, not from epoch 1.")
    parser.add_argument("--max-epochs-profile", type=int, default=None,
                         help="Override epochs for a single seed, used by run_all.py --profile.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(model_name: str, in_channels: int, num_classes_by_dataset: dict[str, int], dropout: float):
    if model_name == "cnn":
        return CNN1DMultiescalaResidualMultiHead(
            in_channels=in_channels, num_classes_by_dataset=num_classes_by_dataset, dropout=dropout,
        )
    return LSTMMultitarea(in_channels=in_channels, num_classes_by_dataset=num_classes_by_dataset, head_dropout=dropout)


@torch.no_grad()
def evaluate_dataset(model: nn.Module, loader, criterion: nn.Module, device: str, dataset_name: str) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb, dataset_name)
        loss = criterion(logits, yb)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(yb.cpu().tolist())

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": accuracy_score(all_targets, all_preds),
        "f1_macro": f1_score(all_targets, all_preds, average="macro"),
    }


def interleaved_batches(loaders_by_dataset: dict[str, torch.utils.data.DataLoader]):
    iters = {name: iter(loader) for name, loader in loaders_by_dataset.items()}
    active = list(loaders_by_dataset.keys())
    while active:
        random.shuffle(active)
        next_active = []
        for name in active:
            it = iters[name]
            try:
                batch = next(it)
                yield name, batch
                next_active.append(name)
            except StopIteration:
                continue
        active = next_active


def build_data_for_seed(args: argparse.Namespace, seed: int, loader_kwargs: dict) -> dict:
    data: dict[str, dict] = {}
    in_channels: int | None = None

    for dataset_name in args.dataset_names:
        x_train, y_train, x_test, y_test = load_ucr_dataset(args.dataset_root, dataset_name)
        x_tr, x_val, y_tr, y_val = split_train_validation(x_train, y_train, args.val_size, seed)

        if in_channels is None:
            in_channels = int(x_tr.shape[1])
        elif int(x_tr.shape[1]) != in_channels:
            raise ValueError("All datasets must have the same number of channels.")

        # train union test, not just y_train: load_ucr_dataset remaps class
        # labels to indices based on the two combined, so the model head and
        # class-weight tensor must be sized off the same union or a class
        # seen only in test would index out of range.
        num_classes = int(np.unique(np.concatenate([y_train, y_test])).shape[0])

        data[dataset_name] = {
            "train_loader": make_loader(x_tr, y_tr, args.batch_size, shuffle=True, seed=seed, **loader_kwargs),
            "val_loader": make_loader(x_val, y_val, args.batch_size, shuffle=False, seed=seed, **loader_kwargs),
            "test_loader": make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=seed, **loader_kwargs),
            "class_weights": class_weights_from_labels(y_tr, num_classes),
            "num_classes": num_classes,
        }

    if in_channels is None:
        raise RuntimeError("No datasets were loaded.")

    return {"datasets": data, "in_channels": in_channels}


def train_one_seed(args: argparse.Namespace, seed: int, run_dir: Path) -> dict:
    seed_everything(seed)
    epochs = args.max_epochs_profile or args.epochs

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and args.device == "cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    non_blocking = bool(loader_kwargs["pin_memory"])
    payload = build_data_for_seed(args, seed, loader_kwargs)
    datasets = payload["datasets"]
    in_channels = payload["in_channels"]

    num_classes_by_dataset = {name: info["num_classes"] for name, info in datasets.items()}
    model = build_model(args.model, in_channels, num_classes_by_dataset, args.dropout).to(args.device)

    criteria = {
        name: nn.CrossEntropyLoss(weight=info["class_weights"].to(args.device), label_smoothing=args.label_smoothing)
        for name, info in datasets.items()
    }

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    steps_per_epoch = int(sum(len(info["train_loader"]) for info in datasets.values()))
    if args.scheduler == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch)
        step_per_batch = True
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        step_per_batch = False
    else:
        scheduler = None
        step_per_batch = False

    checkpoint_path = run_dir / "best_model.pt"
    resume_path = run_dir / "checkpoint.pt"

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0
    start_epoch = 1
    elapsed_before_resume = 0.0

    if resume_path.exists() and not args.force:
        ckpt = torch.load(resume_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt["history"]
        best_val_loss = ckpt["best_val_loss"]
        best_epoch = ckpt["best_epoch"]
        best_state = ckpt["best_state"]
        patience_counter = ckpt["patience_counter"]
        start_epoch = ckpt["epoch"] + 1
        elapsed_before_resume = ckpt["elapsed_seconds"]
        random.setstate(ckpt["rng_random"])
        np.random.set_state(ckpt["rng_numpy"])
        # torch.set_rng_state() strictly requires a CPU ByteTensor, but
        # map_location above just moved every tensor in the checkpoint --
        # this one included -- onto args.device (cuda). Move it back.
        torch.set_rng_state(ckpt["rng_torch"].cpu())
        print(f"[{args.model} seed={seed}] resuming from epoch {start_epoch} "
              f"(checkpoint found at {resume_path}, {elapsed_before_resume:.0f}s already elapsed)")

    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for dataset_name, (xb, yb) in interleaved_batches({k: v["train_loader"] for k, v in datasets.items()}):
            xb = xb.to(args.device, non_blocking=non_blocking)
            yb = yb.to(args.device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, dataset_name)
            loss = criteria[dataset_name](logits, yb)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None and step_per_batch:
                scheduler.step()

            bs = yb.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        if scheduler is not None and not step_per_batch:
            scheduler.step()

        train_loss = running_loss / max(n_samples, 1)

        val_metrics_by_dataset: dict[str, dict] = {}
        for dataset_name, info in datasets.items():
            val_metrics_by_dataset[dataset_name] = evaluate_dataset(
                model, info["val_loader"], criteria[dataset_name], args.device, dataset_name,
            )

        val_loss_mean = float(np.mean([m["loss"] for m in val_metrics_by_dataset.values()]))
        val_acc_mean = float(np.mean([m["acc"] for m in val_metrics_by_dataset.values()]))
        val_f1_mean = float(np.mean([m["f1_macro"] for m in val_metrics_by_dataset.values()]))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_mean": val_loss_mean,
            "val_acc_mean": val_acc_mean,
            "val_f1_macro_mean": val_f1_mean,
        })

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            best_epoch = epoch
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{args.model} seed={seed}] epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss_mean={val_loss_mean:.4f} val_acc_mean={val_acc_mean:.4f} val_f1_mean={val_f1_mean:.4f}"
        )

        if epoch % args.checkpoint_every_epochs == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "history": history,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "patience_counter": patience_counter,
                "elapsed_seconds": elapsed_before_resume + (time.time() - t0),
                "rng_random": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
            }, resume_path)

        if patience_counter >= args.patience:
            print(f"[{args.model} seed={seed}] early stopping at epoch {epoch} (patience={args.patience})")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_path)
    elapsed_s = elapsed_before_resume + (time.time() - t0)
    resume_path.unlink(missing_ok=True)

    test_by_dataset: dict[str, dict] = {}
    for dataset_name, info in datasets.items():
        test_by_dataset[dataset_name] = evaluate_dataset(
            model, info["test_loader"], criteria[dataset_name], args.device, dataset_name,
        )

    test_acc_mean = float(np.mean([m["acc"] for m in test_by_dataset.values()]))
    test_f1_mean = float(np.mean([m["f1_macro"] for m in test_by_dataset.values()]))

    result = {
        "model": args.model,
        "seed": seed,
        "device": args.device,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_loss_mean": best_val_loss,
        "test_acc_mean": test_acc_mean,
        "test_f1_macro_mean": test_f1_mean,
        "test_by_dataset": test_by_dataset,
        "elapsed_seconds": elapsed_s,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def write_result_csvs(results: list[dict], model_name: str, results_dir: Path, dataset_names: list[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        for dataset_name in dataset_names:
            m = r["test_by_dataset"][dataset_name]
            rows.append({
                "model": model_name,
                "seed": r["seed"],
                "dataset": dataset_name,
                "epochs_trained": r["epochs_trained"],
                "best_epoch": r["best_epoch"],
                "test_acc": m["acc"],
                "test_f1_macro": m["f1_macro"],
                "elapsed_seconds": r["elapsed_seconds"],
            })
    runs_df = pd.DataFrame(rows)

    summary_path = results_dir / "runs_summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[~((existing["model"] == model_name))]
        runs_df = pd.concat([existing, runs_df], ignore_index=True)
    runs_df.sort_values(["model", "seed", "dataset"]).to_csv(summary_path, index=False)

    per_dataset = (
        runs_df[runs_df["model"] == model_name]
        .groupby("dataset")
        .agg(
            acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
            acc_min=("test_acc", "min"), acc_max=("test_acc", "max"),
            f1_mean=("test_f1_macro", "mean"), f1_std=("test_f1_macro", "std"),
            f1_min=("test_f1_macro", "min"), f1_max=("test_f1_macro", "max"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    out_name = "per_dataset_results.csv" if model_name == "cnn" else f"per_dataset_{model_name}.csv"
    per_dataset.to_csv(results_dir / out_name, index=False)
    print(f"Wrote {results_dir / out_name} and {summary_path}")


def run(args: argparse.Namespace) -> None:
    args.device = resolve_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/runs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.model}] device={args.device} datasets={len(args.dataset_names)} seeds={args.seeds}")

    results = []
    for seed in args.seeds:
        run_dir = output_dir / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"[{args.model}] seed={seed} already trained, skipping (metrics.json found). Use --force to redo.")
            with metrics_path.open("r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        results.append(train_one_seed(args, seed, run_dir))

    write_result_csvs(results, args.model, Path(args.results_dir), args.dataset_names)

    accs = np.array([r["test_acc_mean"] for r in results])
    f1s = np.array([r["test_f1_macro_mean"] for r in results])
    print(f"\n[{args.model}] Global ACC mean={accs.mean():.4f} std={accs.std(ddof=1) if len(accs) > 1 else 0.0:.5f} "
          f"min={accs.min():.4f} max={accs.max():.4f}")
    print(f"[{args.model}] Global F1  mean={f1s.mean():.4f} std={f1s.std(ddof=1) if len(f1s) > 1 else 0.0:.5f} "
          f"min={f1s.min():.4f} max={f1s.max():.4f}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
