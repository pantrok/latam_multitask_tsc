from __future__ import annotations

"""
Bounded Optuna search for the shared LSTM backbone, over the same tunable
dimensions CNN's configs/final.yaml came from: learning rate, weight decay,
dropout, scheduler, gradient clipping, label smoothing. Architecture
(hidden size, layers, bidirectionality) is not searched, matching how the
CNN's own search never varied channel counts or kernel sizes either.

The CNN/LSTM contrast in this repo has a confound: the CNN was tuned for
this exact protocol and the LSTM never was, so any claim that the CNN
adapts better than the LSTM partly reflects that asymmetry rather than an
architectural difference. This search exists to remove that confound.

Objective: validation loss of the joint (C2) training, at a reduced proxy
protocol -- one seed, roughly half the epoch budget -- not the full 8-seed
run. A 30-trial search at the full 8-seed budget is not tractable;
cheap-HPO-then-verify-at-full-budget is the standard workaround. Whatever
config wins here still needs a full 8-seed run before it replaces anything
in configs/final.yaml -- this script only picks a candidate.
"""

import argparse
import json
import shutil
from pathlib import Path

import optuna

from src.training import train_multitask

N_TRIALS_DEFAULT = 30
SEARCH_SEED = 7
SEARCH_EPOCHS = 8
SEARCH_PATIENCE = 8  # >= SEARCH_EPOCHS -- no early stop within the proxy budget


def _train_args(trial: optuna.Trial, dataset_names: list[str], dataset_root: str, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        model="lstm", dataset_names=dataset_names, dataset_root=dataset_root,
        epochs=SEARCH_EPOCHS, batch_size=48,
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        val_size=0.2, patience=SEARCH_PATIENCE,
        scheduler=trial.suggest_categorical("scheduler", ["cosine", "onecycle", "none"]),
        grad_clip=trial.suggest_categorical("grad_clip", [0.0, 0.5, 1.0]),
        label_smoothing=trial.suggest_categorical("label_smoothing", [0.0, 0.02, 0.05]),
        device=device, num_workers=0, pin_memory=False,
        deterministic=False, force=True,
        checkpoint_every_epochs=SEARCH_EPOCHS + 1,  # trial is disposable, skip mid-run checkpoint I/O
        max_epochs_profile=None,
    )


def objective(trial: optuna.Trial, dataset_names: list[str], dataset_root: str,
              device: str, search_dir: Path) -> float:
    args = _train_args(trial, dataset_names, dataset_root, device)
    run_dir = search_dir / f"trial_{trial.number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = train_multitask.train_one_seed(args, SEARCH_SEED, run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)  # only the trial's score matters, not the checkpoint
    return result["best_val_loss_mean"]


def run(dataset_names: list[str], dataset_root: str, device: str, n_trials: int,
        search_dir: Path, out_path: Path) -> optuna.Study:
    search_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, dataset_names, dataset_root, device, search_dir),
                   n_trials=n_trials)

    print(f"Best trial: {study.best_trial.number}  val_loss={study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "search_epochs": SEARCH_EPOCHS,
        "search_seed": SEARCH_SEED,
    }, indent=2))
    print(f"Wrote {out_path}")

    trials_csv = out_path.with_suffix(".csv")
    study.trials_dataframe().to_csv(trials_csv, index=False)
    print(f"Wrote {trials_csv}")
    return study


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bounded Optuna hyperparameter search for the shared LSTM.")
    p.add_argument("--dataset-names", type=str, nargs="+", required=True)
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    p.add_argument("--search-dir", type=Path, default=Path("artifacts/hparam_search/lstm"))
    p.add_argument("--out", type=Path, default=Path("artifacts/analysis/lstm_hparam_search.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = train_multitask.resolve_device(args.device)
    run(args.dataset_names, args.dataset_root, device, args.n_trials, args.search_dir, args.out)


if __name__ == "__main__":
    main()
