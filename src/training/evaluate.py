from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _seed_level_stats(runs_summary: pd.DataFrame, model: str) -> dict:
    sub = runs_summary[runs_summary["model"] == model]
    per_seed = sub.groupby("seed").agg(acc=("test_acc", "mean"), f1=("test_f1_macro", "mean"))
    acc, f1 = per_seed["acc"].to_numpy(), per_seed["f1"].to_numpy()
    n = len(acc)

    def stats(x: np.ndarray) -> dict:
        mean = float(x.mean())
        std = float(x.std(ddof=1)) if n > 1 else 0.0
        return {
            "mean": mean, "std": std, "min": float(x.min()), "max": float(x.max()),
            "cv_pct": 100.0 * std / mean if mean else 0.0,
            "ci95": 1.96 * std / np.sqrt(n) if n > 0 else 0.0,
        }

    return {"n_seeds": int(n), "acc": stats(acc), "f1": stats(f1)}


def _per_dataset_stats(runs_summary: pd.DataFrame, model: str) -> dict:
    sub = runs_summary[runs_summary["model"] == model].groupby("dataset").agg(
        acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
        acc_min=("test_acc", "min"), acc_max=("test_acc", "max"),
    )
    acc_mean = sub["acc_mean"]
    return {
        "acc_mean": float(acc_mean.mean()), "acc_std": float(acc_mean.std(ddof=1)),
        "acc_min": float(acc_mean.min()), "acc_max": float(acc_mean.max()),
    }


def build_report(results_dir: Path, baseline_name: str = "HC2") -> dict:
    runs_summary = pd.read_csv(results_dir / "runs_summary.csv")
    baselines = pd.read_csv(results_dir / "baselines_reference.csv")

    report: dict = {"models_present": sorted(runs_summary["model"].unique().tolist())}

    for model in ("cnn", "lstm"):
        if model not in runs_summary["model"].unique():
            continue
        report[model] = {
            "global": _seed_level_stats(runs_summary, model),
            "per_dataset": _per_dataset_stats(runs_summary, model),
        }

    if "cnn" in runs_summary["model"].unique():
        cnn_per_dataset = runs_summary[runs_summary["model"] == "cnn"].groupby("dataset")["test_acc"].mean()
        base = baselines.set_index("dataset")[baseline_name] if baseline_name in baselines.columns else None
        if base is not None:
            common = cnn_per_dataset.index.intersection(base.index)
            delta = (base.loc[common] - cnn_per_dataset.loc[common]).sort_values()
            report["gap_vs_baseline"] = {
                "baseline": baseline_name,
                "mean_gap": float(delta.mean()),
                "top5_smallest_gap": delta.head(5).round(4).to_dict(),
                "bottom5_largest_gap": delta.tail(5).round(4).to_dict(),
            }
        top5 = cnn_per_dataset.sort_values(ascending=False).head(5).round(4).to_dict()
        bottom5 = cnn_per_dataset.sort_values(ascending=True).head(5).round(4).to_dict()
        report["cnn_top5_bottom5"] = {"top5": top5, "bottom5": bottom5}

    return report


def _acc_stats(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    return {
        "acc_mean": float(s.mean()) if n else None,
        "acc_std": float(s.std(ddof=1)) if n > 1 else (0.0 if n == 1 else None),
        "acc_min": float(s.min()) if n else None,
        "acc_max": float(s.max()) if n else None,
        "n_datasets": n,
    }


def build_three_condition_table(results_dir: Path) -> pd.DataFrame:
    """Specialist / joint (C2) / fine-tuned (C3), each at scope=30 (all Sensor
    datasets) and scope=20 (the subset with a published HC2 reference, so the
    specialist row is comparable to the other two). MiniRocket-local rows are
    written as all-None, not omitted, if baselines_local_minirocket.csv
    doesn't exist yet -- makes it visible in the CSV that they're pending,
    not silently absent."""
    rows: list[dict] = []

    baselines = pd.read_csv(results_dir / "baselines_reference.csv")
    hc2 = baselines.set_index("dataset")["HC2"].dropna() if "HC2" in baselines.columns else pd.Series(dtype=float)
    subset20 = set(hc2.index)

    rows.append({"condition": "specialist", "model": "HC2", "scope": 20, **_acc_stats(hc2)})

    mr_path = results_dir / "baselines_local_minirocket.csv"
    if mr_path.exists():
        mr = pd.read_csv(mr_path).set_index("dataset")["acc"]
        rows.append({"condition": "specialist", "model": "MiniRocket_local", "scope": 30, **_acc_stats(mr)})
        rows.append({"condition": "specialist", "model": "MiniRocket_local", "scope": 20,
                     **_acc_stats(mr.loc[mr.index.intersection(subset20)])})
    else:
        rows.append({"condition": "specialist", "model": "MiniRocket_local", "scope": 30,
                     "acc_mean": None, "acc_std": None, "acc_min": None, "acc_max": None, "n_datasets": 0})
        rows.append({"condition": "specialist", "model": "MiniRocket_local", "scope": 20,
                     "acc_mean": None, "acc_std": None, "acc_min": None, "acc_max": None, "n_datasets": 0})

    model_files = [
        ("cnn", "per_dataset_results.csv", "finetune_full_cnn.csv", "finetune_head_cnn.csv", "scratch_cnn.csv"),
        ("lstm", "per_dataset_lstm.csv", "finetune_full_lstm.csv", "finetune_head_lstm.csv", "scratch_lstm.csv"),
    ]
    for model, joint_file, ft_file, head_file, scratch_file in model_files:
        joint_path = results_dir / joint_file
        if joint_path.exists():
            joint = pd.read_csv(joint_path).set_index("dataset")["acc_mean"]
            rows.append({"condition": "joint", "model": model, "scope": 30, **_acc_stats(joint)})
            rows.append({"condition": "joint", "model": model, "scope": 20,
                         **_acc_stats(joint.loc[joint.index.intersection(subset20)])})

        ft_path = results_dir / ft_file
        if ft_path.exists():
            ft = pd.read_csv(ft_path).set_index("dataset")["acc_mean"]
            rows.append({"condition": "finetuned", "model": model, "scope": 30, **_acc_stats(ft)})
            rows.append({"condition": "finetuned", "model": model, "scope": 20,
                         **_acc_stats(ft.loc[ft.index.intersection(subset20)])})

        # C3-head control: backbone frozen, only that dataset's head fine-tuned.
        head_path = results_dir / head_file
        if head_path.exists():
            head = pd.read_csv(head_path).set_index("dataset")["acc_mean"]
            rows.append({"condition": "head", "model": model, "scope": 30, **_acc_stats(head)})
            rows.append({"condition": "head", "model": model, "scope": 20,
                         **_acc_stats(head.loc[head.index.intersection(subset20)])})

        # C4 control: trained from random init, no shared backbone at all.
        scratch_path = results_dir / scratch_file
        if scratch_path.exists():
            scratch = pd.read_csv(scratch_path).set_index("dataset")["acc_mean"]
            rows.append({"condition": "scratch", "model": model, "scope": 30, **_acc_stats(scratch)})
            rows.append({"condition": "scratch", "model": model, "scope": 20,
                         **_acc_stats(scratch.loc[scratch.index.intersection(subset20)])})

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate run artifacts into a machine-readable summary report.")
    p.add_argument("--results-dir", type=str, default="artifacts/results")
    p.add_argument("--baseline-name", type=str, default="HC2")
    p.add_argument("--out", type=str, default="artifacts/results/summary_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(Path(args.results_dir), args.baseline_name)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
