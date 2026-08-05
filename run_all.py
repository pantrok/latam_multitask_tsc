#!/usr/bin/env python3
"""
run_all.py -- single entry point for the whole pipeline, meant to run inside
Colab after cloning this repo. See colab.ipynb for the minimal notebook that
calls this.

Stages (comma-separated, or "all"):
  download, eda, geometry, train-cnn, train-lstm, finetune-cnn, finetune-lstm,
  random-backbone, bn-only, head-updates, bn-trainsize, condition-dispersion,
  paired-tests, c2-convergence, baselines, baselines-local, correlation,
  report, figures

  bn-only is CNN-only -- the LSTM backbone has no BatchNorm layers, so
  "recalibrate normalization statistics" is undefined for it (running it
  would just reproduce the joint checkpoint's own accuracy unchanged, since
  nothing in the model would move at all). See src/training/bn_only.py.

  scratch-cnn/scratch-lstm are NOT part of "all" -- the fourth condition
  (train each dataset from random init, same epoch budget/seeds as
  fine-tuning, no shared backbone at all), invoked explicitly since it
  roughly doubles cost by default (--scratch-lr-mode both runs two lr
  variants). See src/training/scratch_baseline.py.

  hparam-search-lstm is also NOT part of "all" -- a bounded Optuna search
  over the LSTM's training hyperparameters, mirroring the search the CNN's
  configs/final.yaml already came from. Optional and costly (default 30
  trials); see src/training/hparam_search_lstm.py.

Examples
--------
  python run_all.py --stage all
  python run_all.py --stage download,eda,geometry
  python run_all.py --stage train-cnn --seeds 7 21
  python run_all.py --stage figures
  python run_all.py --profile
  python run_all.py --stage all --dry-run
  python run_all.py --stage all --artifacts-dir /content/drive/MyDrive/latam_run

Each stage detects its own output artifact and skips re-running unless
--force is passed. train-cnn/train-lstm additionally checkpoint per seed
inside src/training/train_multitask.py, so a stage that dies mid-run resumes
at the next untrained seed on the next invocation, without --force.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import dataset_list, download as data_download  # noqa: E402
from src.analysis import baselines as baselines_mod  # noqa: E402
from src.analysis import dataset_geometry, eda as eda_mod  # noqa: E402
from src.analysis import geometry_structure_check  # noqa: E402
from src.training import train_multitask, evaluate as evaluate_mod  # noqa: E402
from src.training import finetune as finetune_mod  # noqa: E402
from src.training import scratch_baseline as scratch_baseline_mod  # noqa: E402
from src.training import random_backbone as random_backbone_mod  # noqa: E402
from src.training import bn_only as bn_only_mod  # noqa: E402
from src.training import hparam_search_lstm as hparam_search_lstm_mod  # noqa: E402
from src.analysis import head_update_counts as head_update_counts_mod  # noqa: E402
from src.analysis import bn_effect_vs_trainsize as bn_effect_vs_trainsize_mod  # noqa: E402
from src.analysis import paired_tests as paired_tests_mod  # noqa: E402
from src.analysis import condition_dispersion as condition_dispersion_mod  # noqa: E402
from src.analysis import c2_convergence as c2_convergence_mod  # noqa: E402

STAGES = ["download", "eda", "geometry", "train-cnn", "train-lstm", "finetune-cnn", "finetune-lstm",
          "random-backbone", "bn-only", "head-updates", "bn-trainsize", "condition-dispersion",
          "paired-tests", "c2-convergence", "baselines", "baselines-local", "correlation", "report", "figures"]
EXPLORATORY_STAGES = ["scratch-cnn", "scratch-lstm", "hparam-search-lstm"]


def _load_final_config(path: Path = REPO_ROOT / "configs" / "final.yaml") -> dict:
    """configs/final.yaml is the single source of truth for hyperparameters --
    CLI flags below only override it per-invocation, they never fork from it."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_CFG = _load_final_config()
DEFAULT_SEEDS = _CFG["seeds"]


def _model_default(model: str, key: str):
    """CNN and LSTM each have their own hyperparameter defaults in
    configs/final.yaml -- there is no shared global default. A CLI flag, if
    passed, still overrides both models uniformly; this is only the
    fallback."""
    return _CFG[model][key]


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple MPS"
    return "CPU"


def capture_reproducibility_snapshot(artifacts_dir: Path, device: str) -> None:
    """Written once at the very start of main(), into --artifacts-dir rather
    than a repo-relative path -- so it persists to Drive across Colab
    disconnects instead of being lost with the ephemeral clone, and is
    captured before a run that dies partway through rather than after.
    Best-effort: a missing git history or pip should not abort the run,
    only leave that field marked unavailable."""
    out_path = artifacts_dir / "repro_snapshot.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as e:
        commit = f"unavailable ({e})"

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout
        relevant = [
            line for line in freeze.splitlines()
            if re.search(r"^(torch|numpy|pandas|scipy|aeon|scikit-learn|optuna)==", line, re.I)
        ]
    except Exception as e:
        relevant = [f"pip freeze unavailable ({e})"]

    lines = [
        "# Reproducibility snapshot",
        f"captured (UTC): {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"git commit: {commit}",
        f"device: {device} ({gpu_name(device)})",
        f"python: {sys.version.split()[0]}",
        "pip freeze (filtered to packages this pipeline depends on):",
        *(f"  {line}" for line in relevant),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote reproducibility snapshot to {out_path}")


def log_runtime(runtime_md: Path, stage: str, elapsed_s: float, device: str, note: str = "") -> None:
    runtime_md.parent.mkdir(parents=True, exist_ok=True)
    header = "| timestamp (UTC) | stage | elapsed | device | note |\n|---|---|---|---|---|\n"
    if not runtime_md.exists() or header.split("\n")[0] not in runtime_md.read_text(encoding="utf-8", errors="ignore"):
        runtime_md.write_text("# Runtime log\n\n" + header, encoding="utf-8")
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    m, s = divmod(elapsed_s, 60)
    h, m = divmod(int(m), 60)
    elapsed_str = f"{h:02d}:{m:02d}:{int(s):02d}"
    with runtime_md.open("a", encoding="utf-8") as f:
        f.write(f"| {ts} | {stage} | {elapsed_str} | {device} | {note} |\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=str, default="all", help="Comma-separated stage names, or 'all'.")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=None, help="Overrides both models' configs/final.yaml default if passed.")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--scheduler", type=str, default=None, choices=["cosine", "onecycle", "none"])
    p.add_argument("--num-workers", type=int, default=None, help="Default: 4 on cuda/mps, 0 on cpu.")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--checkpoint-every-epochs", type=int, default=1,
                    help="train-cnn/train-lstm: save a resumable mid-training checkpoint every N epochs.")
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    p.add_argument("--artifacts-dir", type=str, default="artifacts", help="e.g. /content/drive/MyDrive/.../artifacts")
    p.add_argument("--force", action="store_true", help="Re-run stages even if their artifact already exists.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and estimate, execute nothing.")
    p.add_argument("--profile", action="store_true",
                    help="Time 1 seed x 2 epochs per training stage requested and extrapolate to the full config.")
    p.add_argument("--finetune-mode", choices=["full", "head", "both", "head_bn_eval"], default="full",
                    help="full: unfreeze whole model. head: freeze all but that dataset's head. "
                         "both: run each separately (roughly doubles finetune-cnn/lstm cost). "
                         "head_bn_eval: same as head, but BatchNorm stays in eval() throughout "
                         "(ablation on whether frozen-backbone BN running stats drifting matters).")
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--finetune-patience", type=int, default=5)
    p.add_argument("--finetune-lr", type=float, default=2e-4,
                    help="Default is 1/10th of the joint training lr (2e-3) -- standard fine-tuning practice.")
    p.add_argument("--hparam-search-trials", type=int, default=30,
                    help="Number of Optuna trials for --stage hparam-search-lstm.")
    p.add_argument("--scratch-lr-mode", choices=["primary", "secondary", "both"], default="both",
                    help="primary=1.617e-3 (joint lr, recommended), secondary=2e-4 (matches C3's "
                         "fine-tune lr). Default 'both' reports each rather than picking the more "
                         "favorable one silently.")
    return p.parse_args()


def expand_stages(stage_arg: str) -> list[str]:
    if stage_arg == "all":
        return list(STAGES)
    requested = [s.strip() for s in stage_arg.split(",") if s.strip()]
    valid = STAGES + EXPLORATORY_STAGES
    unknown = [s for s in requested if s not in valid]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {unknown}. Valid: {valid}")
    return requested


def figures_output_dir(artifacts_dir: Path) -> Path:
    """Sits next to artifacts_dir rather than always under the repo's cwd, so
    pointing --artifacts-dir at Drive also persists the regenerated figures
    across Colab disconnects, not just the CSVs/JSON that feed them."""
    return artifacts_dir.parent / "figures"


def _all_seeds_trained(artifacts_dir: Path, model: str, seeds: list[int]) -> bool:
    """Not just 'does runs_summary.csv exist' -- a stale or --profile-only CSV
    (fewer/different seeds) must NOT be mistaken for a complete run."""
    summary_path = artifacts_dir / "results" / "runs_summary.csv"
    if not summary_path.exists():
        return False
    import pandas as pd
    df = pd.read_csv(summary_path)
    trained = set(df.loc[df["model"] == model, "seed"].unique().tolist())
    return set(seeds).issubset(trained)


def _all_seeds_finetuned(artifacts_dir: Path, model: str, seeds: list[int], modes: list[str]) -> bool:
    """Same reasoning as _all_seeds_trained: a partial or --force-free rerun
    with fewer seeds/modes must not read as 'already complete'."""
    summary_path = artifacts_dir / "results" / "finetune_runs_summary.csv"
    if not summary_path.exists():
        return False
    import pandas as pd
    df = pd.read_csv(summary_path)
    sub = df[df["model"] == model]
    for mode in modes:
        trained = set(sub.loc[sub["mode"] == mode, "seed"].unique().tolist())
        if not set(seeds).issubset(trained):
            return False
    return True


def _all_seeds_random_backbone(artifacts_dir: Path, model: str, seeds: list[int]) -> bool:
    """Same reasoning as _all_seeds_trained: a partial run with fewer seeds
    must not read as 'already complete'."""
    summary_path = artifacts_dir / "results" / "random_backbone_runs_summary.csv"
    if not summary_path.exists():
        return False
    import pandas as pd
    df = pd.read_csv(summary_path)
    trained = set(df.loc[df["model"] == model, "seed"].unique().tolist())
    return set(seeds).issubset(trained)


def _all_seeds_bn_only(artifacts_dir: Path, model: str, seeds: list[int]) -> bool:
    """Same reasoning as _all_seeds_trained: a partial run with fewer seeds
    must not read as 'already complete'."""
    summary_path = artifacts_dir / "results" / "bn_only_runs_summary.csv"
    if not summary_path.exists():
        return False
    import pandas as pd
    df = pd.read_csv(summary_path)
    trained = set(df.loc[df["model"] == model, "seed"].unique().tolist())
    return set(seeds).issubset(trained)


def _all_seeds_scratch_trained(artifacts_dir: Path, model: str, seeds: list[int], lr_labels: list[str]) -> bool:
    """Same reasoning as _all_seeds_finetuned: a partial run at only one of
    the two lr variants must not read as 'already complete'."""
    summary_path = artifacts_dir / "results" / "scratch_runs_summary.csv"
    if not summary_path.exists():
        return False
    import pandas as pd
    df = pd.read_csv(summary_path)
    sub = df[df["model"] == model]
    for lr_label in lr_labels:
        trained = set(sub.loc[sub["lr_label"] == lr_label, "seed"].unique().tolist())
        if not set(seeds).issubset(trained):
            return False
    return True


def artifact_exists(stage: str, artifacts_dir: Path, seeds: list[int] | None = None,
                     finetune_modes: list[str] | None = None, scratch_lr_labels: list[str] | None = None) -> bool:
    modes = finetune_modes or ["full"]
    lr_labels = scratch_lr_labels or ["primary", "secondary"]
    checks = {
        "download": lambda: not data_download.verify(Path("data/raw"), dataset_list.get_sensor_datasets("assets/DataSummary.csv")),
        "eda": lambda: (artifacts_dir / "eda" / "dataset_profiles.csv").exists(),
        "geometry": lambda: (artifacts_dir / "eda" / "cid_distance_matrix.csv").exists(),
        "train-cnn": lambda: (artifacts_dir / "results" / "per_dataset_results.csv").exists() and _all_seeds_trained(artifacts_dir, "cnn", seeds or []),
        "train-lstm": lambda: (artifacts_dir / "results" / "per_dataset_lstm.csv").exists() and _all_seeds_trained(artifacts_dir, "lstm", seeds or []),
        "finetune-cnn": lambda: _all_seeds_finetuned(artifacts_dir, "cnn", seeds or [], modes),
        "finetune-lstm": lambda: _all_seeds_finetuned(artifacts_dir, "lstm", seeds or [], modes),
        "random-backbone": lambda: (_all_seeds_random_backbone(artifacts_dir, "cnn", seeds or []) and
                                     _all_seeds_random_backbone(artifacts_dir, "lstm", seeds or [])),
        "bn-only": lambda: _all_seeds_bn_only(artifacts_dir, "cnn", seeds or []),
        "scratch-cnn": lambda: _all_seeds_scratch_trained(artifacts_dir, "cnn", seeds or [], lr_labels),
        "scratch-lstm": lambda: _all_seeds_scratch_trained(artifacts_dir, "lstm", seeds or [], lr_labels),
        "hparam-search-lstm": lambda: (artifacts_dir / "analysis" / "lstm_hparam_search.json").exists(),
        "head-updates": lambda: (artifacts_dir / "results" / "head_update_counts.csv").exists(),
        "bn-trainsize": lambda: (artifacts_dir / "results" / "bn_effect_vs_trainsize.json").exists(),
        "condition-dispersion": lambda: (artifacts_dir / "results" / "condition_dispersion.csv").exists(),
        "paired-tests": lambda: (artifacts_dir / "results" / "paired_tests.json").exists(),
        "c2-convergence": lambda: (artifacts_dir / "results" / "c2_training_curves.csv").exists(),
        "baselines": lambda: (artifacts_dir / "results" / "baselines_reference.csv").exists(),
        "baselines-local": lambda: (artifacts_dir / "results" / "baselines_local_minirocket.csv").exists(),
        "correlation": lambda: (artifacts_dir / "analysis" / "geometry_gap_correlation.json").exists(),
        "figures": lambda: (figures_output_dir(artifacts_dir) / "fig_four_conditions.pdf").exists(),
        "report": lambda: (artifacts_dir / "results" / "summary_report.json").exists(),
    }
    try:
        return checks[stage]()
    except Exception:
        return False


def default_num_workers() -> int:
    """0 by default, on every device -- not just the original Mac runs.

    Each dataset gets its own train/val/test DataLoader (30 datasets x 3
    splits = 90 loaders per seed), and train_multitask.py builds all of them
    upfront with persistent_workers=True, so a nonzero num_workers here means
    that many background processes PER LOADER, alive simultaneously for the
    whole seed. On a free Colab instance (2 vCPUs) even num_workers=4 means
    up to ~120 worker processes contending for 2 cores -- confirmed in
    practice by PyTorch's own "suggested max number of workers is 2" warning
    firing on every one of the 90 loaders. Separately, each dataset is
    already a fully in-memory TensorDataset (ucr_io.py loads the whole
    .tsv via np.loadtxt once) with no per-item preprocessing, so background
    workers buy no real prefetch benefit here to begin with -- unlike the
    typical case (e.g. loading+decoding images from disk per item) where
    num_workers>0 helps. Override with --num-workers if you want to test
    otherwise; do not silently raise this default again.
    """
    return 0


def run_stage(stage: str, args: argparse.Namespace, device: str, artifacts_dir: Path) -> None:
    dataset_names = dataset_list.get_sensor_datasets("assets/DataSummary.csv")
    num_workers = args.num_workers if args.num_workers is not None else default_num_workers()

    if stage == "download":
        data_download.run(Path(args.dataset_root), force=args.force)

    elif stage == "eda":
        eda_mod.run(args.dataset_root, artifacts_dir / "eda")

    elif stage == "geometry":
        dataset_geometry.run(args.dataset_root, artifacts_dir / "eda")
        geometry_structure_check.run(
            artifacts_dir / "eda" / "dtw_distance_matrix.csv",
            artifacts_dir / "eda" / "tsne_dtw_coords.csv",
            artifacts_dir / "eda" / "geometry_structure_check.json",
        )

    elif stage in ("train-cnn", "train-lstm"):
        model = "cnn" if stage == "train-cnn" else "lstm"
        seeds = args.seeds
        epochs = args.epochs if args.epochs is not None else _model_default(model, "epochs")
        patience = args.patience if args.patience is not None else _model_default(model, "patience")
        learning_rate = args.learning_rate if args.learning_rate is not None else _model_default(model, "learning_rate")
        weight_decay = args.weight_decay if args.weight_decay is not None else _model_default(model, "weight_decay")
        dropout = args.dropout if args.dropout is not None else _model_default(model, "dropout")
        batch_size = args.batch_size if args.batch_size is not None else _model_default(model, "batch_size")
        grad_clip = args.grad_clip if args.grad_clip is not None else _model_default(model, "grad_clip")
        label_smoothing = args.label_smoothing if args.label_smoothing is not None else _model_default(model, "label_smoothing")
        scheduler = args.scheduler if args.scheduler is not None else _model_default(model, "scheduler")
        # --profile writes to an entirely separate artifacts/profile/ subtree.
        # It must never share a path with the real run's output -- otherwise a
        # profiling pass (1 seed x 2 epochs) gets mistaken later for a
        # complete run and silently short-circuits real training.
        base_dir = artifacts_dir
        if args.profile:
            seeds = seeds[:1]
            epochs = 2
            base_dir = artifacts_dir / "profile"
        train_args = argparse.Namespace(
            model=model, dataset_names=dataset_names, dataset_root=args.dataset_root,
            output_dir=str(base_dir / "runs" / model), results_dir=str(base_dir / "results"),
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, dropout=dropout, val_size=_CFG["val_size"], patience=patience,
            scheduler=scheduler, grad_clip=grad_clip, label_smoothing=label_smoothing,
            device=device, num_workers=num_workers, pin_memory=(device == "cuda"), seeds=seeds,
            deterministic=args.deterministic, force=args.force, max_epochs_profile=None,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
        )
        train_multitask.run(train_args)

    elif stage in ("finetune-cnn", "finetune-lstm"):
        model = "cnn" if stage == "finetune-cnn" else "lstm"
        finetune_args = argparse.Namespace(
            model=model, dataset_names=dataset_names, dataset_root=args.dataset_root,
            joint_runs_dir=str(artifacts_dir / "runs" / model), joint_results_dir=str(artifacts_dir / "results"),
            output_dir=str(artifacts_dir / "runs" / f"{model}_finetuned"), results_dir=str(artifacts_dir / "results"),
            mode=args.finetune_mode, seeds=args.seeds, epochs=args.finetune_epochs, patience=args.finetune_patience,
            learning_rate=args.finetune_lr, weight_decay=_model_default(model, "weight_decay"),
            batch_size=args.batch_size if args.batch_size is not None else _model_default(model, "batch_size"),
            val_size=_CFG["val_size"], device=device, force=args.force,
        )
        finetune_mod.run(finetune_args)

    elif stage == "random-backbone":
        # Same epoch budget/lr/patience as C3-head (args.finetune_*, not
        # separate flags) -- the two conditions must share an exact protocol
        # for the comparison to isolate what it's meant to isolate.
        for model in ("cnn", "lstm"):
            rb_args = argparse.Namespace(
                model=model, dataset_names=dataset_names, dataset_root=args.dataset_root,
                joint_results_dir=str(artifacts_dir / "results"), results_dir=str(artifacts_dir / "results"),
                seeds=args.seeds, epochs=args.finetune_epochs, patience=args.finetune_patience,
                learning_rate=args.finetune_lr, weight_decay=_model_default(model, "weight_decay"),
                batch_size=args.batch_size if args.batch_size is not None else _model_default(model, "batch_size"),
                val_size=_CFG["val_size"], device=device, force=args.force,
            )
            random_backbone_mod.run(rb_args)

    elif stage == "bn-only":
        # CNN only -- the LSTM has no BatchNorm layers to recalibrate.
        # Same forward-pass budget/patience as C3-head (args.finetune_*),
        # for a like-for-like comparison.
        bn_args = argparse.Namespace(
            model="cnn", dataset_names=dataset_names, dataset_root=args.dataset_root,
            joint_runs_dir=str(artifacts_dir / "runs" / "cnn"), joint_results_dir=str(artifacts_dir / "results"),
            results_dir=str(artifacts_dir / "results"), seeds=args.seeds,
            epochs=args.finetune_epochs, patience=args.finetune_patience,
            batch_size=args.batch_size if args.batch_size is not None else _model_default("cnn", "batch_size"),
            val_size=_CFG["val_size"], device=device, force=args.force,
        )
        bn_only_mod.run(bn_args)

    elif stage == "head-updates":
        head_update_counts_mod.run(
            dataset_names, args.dataset_root,
            artifacts_dir / "results" / "per_dataset_results.csv",
            artifacts_dir / "results" / "runs_summary.csv",
            artifacts_dir / "results" / "finetune_head_cnn.csv",
            artifacts_dir / "results" / "finetune_runs_summary.csv",
            artifacts_dir / "results", artifacts_dir / "analysis", model="cnn",
        )

    elif stage == "bn-trainsize":
        bn_effect_vs_trainsize_mod.run(
            artifacts_dir / "results" / "finetune_head_cnn.csv",
            artifacts_dir / "results" / "finetune_head_bn_eval_cnn.csv",
            artifacts_dir / "results" / "head_update_counts.csv",
            artifacts_dir / "results" / "bn_effect_vs_trainsize.json",
        )

    elif stage == "condition-dispersion":
        condition_dispersion_mod.run(artifacts_dir / "results", artifacts_dir / "results" / "condition_dispersion.csv")

    elif stage == "paired-tests":
        paired_tests_mod.run(
            artifacts_dir / "results" / "finetune_runs_summary.csv",
            artifacts_dir / "results" / "bn_only_runs_summary.csv",
            artifacts_dir / "results" / "paired_tests.json",
        )

    elif stage == "c2-convergence":
        c2_convergence_mod.run(
            artifacts_dir / "runs" / "cnn", args.seeds, artifacts_dir / "results" / "c2_training_curves.csv",
        )

    elif stage in ("scratch-cnn", "scratch-lstm"):
        # Same epoch budget and patience as C3 (args.finetune_epochs/patience,
        # not separate flags -- reusing the same args keeps them from
        # accidentally drifting apart).
        model = "cnn" if stage == "scratch-cnn" else "lstm"
        scratch_args = argparse.Namespace(
            model=model, dataset_names=dataset_names, dataset_root=args.dataset_root,
            joint_results_dir=str(artifacts_dir / "results"), results_dir=str(artifacts_dir / "results"),
            lr_mode=args.scratch_lr_mode, seeds=args.seeds, epochs=args.finetune_epochs,
            patience=args.finetune_patience, weight_decay=_model_default(model, "weight_decay"),
            batch_size=args.batch_size if args.batch_size is not None else _model_default(model, "batch_size"),
            val_size=_CFG["val_size"], device=device, force=args.force,
        )
        scratch_baseline_mod.run(scratch_args)

    elif stage == "hparam-search-lstm":
        hparam_search_lstm_mod.run(
            dataset_names, args.dataset_root, device, args.hparam_search_trials,
            artifacts_dir / "hparam_search" / "lstm", artifacts_dir / "analysis" / "lstm_hparam_search.json",
        )

    elif stage == "baselines":
        baselines_mod.run(dataset_names, out_dir=artifacts_dir / "results")

    elif stage == "baselines-local":
        baselines_mod.fit_minirocket_local(dataset_names, args.dataset_root, artifacts_dir / "results")

    elif stage == "correlation":
        import subprocess
        cmd = [
            sys.executable, "-m", "src.analysis.geometry_gap_correlation",
            "--distance-matrix", str(artifacts_dir / "eda" / "dtw_distance_matrix.csv"),
            "--results", str(artifacts_dir / "results" / "per_dataset_results.csv"),
            "--baseline", str(artifacts_dir / "results" / "hc2_baseline.csv"),
            "--metadata", "assets/DataSummary.csv", "--metadata-ucr",
            "--out-dir", str(artifacts_dir / "analysis"),
        ]
        subprocess.run(cmd, check=True)

    elif stage == "figures":
        # Paper figure set (F1-F4 + graphical abstract). fig_tsne_dtw and
        # fig_distance_matrices are kept but out of the paper -- repo-only
        # diagnostics.
        import matplotlib.pyplot as plt

        from src.figures import (
            fig_adaptability_slope, fig_c2_convergence, fig_control_decomposition, fig_distance_matrices,
            fig_factorial_interaction, fig_four_conditions, fig_geometry_null, fig_graphical_abstract,
            fig_tsne_dtw,
        )
        fig_dir = figures_output_dir(artifacts_dir)
        plt.close(fig_tsne_dtw.run(artifacts_dir / "eda" / "tsne_dtw_coords.csv", fig_dir))
        plt.close(fig_distance_matrices.run(artifacts_dir / "eda", fig_dir))

        plt.close(fig_four_conditions.run(artifacts_dir / "results" / "per_dataset_results.csv",
                                           artifacts_dir / "results" / "finetune_head_cnn.csv",
                                           artifacts_dir / "results" / "scratch_cnn.csv",
                                           artifacts_dir / "results" / "baselines_reference.csv",
                                           artifacts_dir / "results" / "baselines_local_minirocket.csv", fig_dir))
        plt.close(fig_control_decomposition.run(artifacts_dir / "results" / "per_dataset_results.csv",
                                                 artifacts_dir / "results" / "finetune_head_cnn.csv",
                                                 artifacts_dir / "results" / "finetune_full_cnn.csv",
                                                 artifacts_dir / "results" / "scratch_cnn.csv",
                                                 artifacts_dir / "results" / "baselines_reference.csv",
                                                 artifacts_dir / "results" / "baselines_local_minirocket.csv", fig_dir))
        plt.close(fig_adaptability_slope.run(artifacts_dir / "results" / "three_condition_table.csv", fig_dir))
        plt.close(fig_c2_convergence.run(artifacts_dir / "results" / "c2_training_curves.csv", fig_dir))
        plt.close(fig_factorial_interaction.run(artifacts_dir / "results" / "per_dataset_results.csv",
                                                 artifacts_dir / "results" / "finetune_head_bn_eval_cnn.csv",
                                                 artifacts_dir / "results" / "bn_only_cnn.csv",
                                                 artifacts_dir / "results" / "finetune_head_cnn.csv",
                                                 artifacts_dir / "results" / "finetune_full_cnn.csv",
                                                 artifacts_dir / "results" / "random_backbone_cnn.csv",
                                                 artifacts_dir / "results" / "baselines_local_minirocket.csv",
                                                 fig_dir, two_panel=True))
        plt.close(fig_geometry_null.run(artifacts_dir / "analysis" / "geometry_gap_correlation.json", fig_dir))
        plt.close(fig_graphical_abstract.run(artifacts_dir / "results" / "per_dataset_results.csv",
                                              artifacts_dir / "results" / "baselines_reference.csv",
                                              artifacts_dir / "results" / "three_condition_table.csv", fig_dir))

    elif stage == "report":
        report = evaluate_mod.build_report(artifacts_dir / "results")
        import json
        out_path = artifacts_dir / "results" / "summary_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))

        three_cond = evaluate_mod.build_three_condition_table(artifacts_dir / "results")
        three_cond_path = artifacts_dir / "results" / "three_condition_table.csv"
        three_cond.to_csv(three_cond_path, index=False)
        print(f"\nWrote {three_cond_path}")
        print(three_cond.to_string(index=False))


def print_plan(stages: list[str], args: argparse.Namespace, device: str) -> None:
    print("Plan:")
    for s in stages:
        note = ""
        if s in ("train-cnn", "train-lstm"):
            model = "cnn" if s == "train-cnn" else "lstm"
            seeds = args.seeds[:1] if args.profile else args.seeds
            epochs = 2 if args.profile else (args.epochs if args.epochs is not None else _model_default(model, "epochs"))
            note = f"seeds={seeds} epochs={epochs} device={device}"
        print(f"  - {s}{'  (' + note + ')' if note else ''}")
    print(f"artifacts-dir = {args.artifacts_dir}")
    print(f"device        = {device} ({gpu_name(device)})")


def _warn_if_ephemeral_on_colab(artifacts_dir: Path) -> None:
    """An invocation without --artifacts-dir on Colab writes to ephemeral
    local disk instead of the Drive-mounted path and is lost on disconnect.
    /content only existing on Colab is the only signal available without an
    explicit flag."""
    on_colab = Path("/content").exists()
    if on_colab and not artifacts_dir.is_absolute():
        print(f"[warn] --artifacts-dir not set (defaulting to '{artifacts_dir}') while running on "
              f"what looks like Colab -- this writes to EPHEMERAL local disk, not your Drive-mounted "
              f"path, and will be lost on disconnect. Pass --artifacts-dir \"$ARTIFACTS_DIR\" explicitly.")


def main() -> None:
    args = parse_args()
    stages = expand_stages(args.stage)
    device = resolve_device()
    artifacts_dir = Path(args.artifacts_dir)
    _warn_if_ephemeral_on_colab(artifacts_dir)

    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    if args.dry_run:
        print_plan(stages, args, device)
        return

    print_plan(stages, args, device)
    capture_reproducibility_snapshot(artifacts_dir, device)
    runtime_md = Path("docs/RUNTIME.md")

    finetune_modes = ["full", "head"] if args.finetune_mode == "both" else [args.finetune_mode]
    scratch_lr_labels = ["primary", "secondary"] if args.scratch_lr_mode == "both" else [args.scratch_lr_mode]
    for stage in stages:
        if not args.force and not args.profile and artifact_exists(stage, artifacts_dir, args.seeds, finetune_modes, scratch_lr_labels):
            print(f"\n=== [{stage}] artifact already present, skipping (use --force to redo) ===")
            continue

        print(f"\n=== [{stage}] starting ===")
        t0 = time.time()
        run_stage(stage, args, device, artifacts_dir)
        elapsed = time.time() - t0
        print(f"=== [{stage}] done in {elapsed:.1f}s ===")

        note = ""
        if args.profile and stage in ("train-cnn", "train-lstm"):
            estimated = elapsed / 2 * args.epochs / 1 * len(DEFAULT_SEEDS)
            note = f"profiled: 1 seed x 2 epochs = {elapsed:.1f}s -> extrapolated {args.epochs} epochs x {len(DEFAULT_SEEDS)} seeds ~= {estimated / 3600:.2f}h"
            print(f"[profile] {note}")
        log_runtime(runtime_md, stage, elapsed, device, note)

    print("\nAll requested stages complete." if not args.profile else "\nProfiling complete. See docs/RUNTIME.md.")


if __name__ == "__main__":
    main()
