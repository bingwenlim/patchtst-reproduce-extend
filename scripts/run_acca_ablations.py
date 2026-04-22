"""ACCA ablation runner (Wang Yuxiao).

Extends scripts/run_benchmarks.py with the ablation matrix described in the
project plan. Only runs combinations that scripts/run_benchmarks.py has not
already covered:

- attention variant across all four datasets, at pre_head and post_head.
- linear variant at post_head on ETTh1 + FX (the other linear combos were
  already produced by run_benchmarks.py).
- alpha_mode=fixed_one ablation on ETTh1 + FX for both acca_type values
  (forces the gate fully open to test whether the learned gate is the only
  reason ACCA is silent).

Each run invokes train.py with `--run_name <id>` so a per-epoch alpha / MSE
trace is written to scripts/traces/<id>_trace.json.

CLI:
    # Run the full 14-run matrix
    uv run python scripts/run_acca_ablations.py

    # Run only ETTh1 ablations (fast on CPU, ~5h total)
    uv run python scripts/run_acca_ablations.py --datasets ETTh1

    # Run only FX and ETTh1 (most relevant when compute is limited)
    uv run python scripts/run_acca_ablations.py --datasets ETTh1 fx

Outputs (aggregated across runs; filtering applies to the tables too):
    scripts/acca_ablation_results.json
    scripts/acca_ablation_results.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paper ETTh1 config: aligns with scripts/run_benchmarks.py so that the ACCA
# numbers here are comparable to the baseline table Ziming already produced.
BASE_ARGS = [
    "--d_model", "16",
    "--n_heads", "4",
    "--d_ff", "128",
    "--dropout", "0.3",
]

DATASETS = ["ETTh1", "traffic", "air", "fx"]


def cfg(name, dataset, extra):
    """Build a run specification.

    The --run_name is a stable slug that both scripts/traces/ filenames and the
    aggregated results tables key on.
    """
    return {
        "name": name,
        "dataset": dataset,
        "args": BASE_ARGS + extra + ["--run_name", name],
    }


RUNS = []

# Main ablation: attention variant, pre_head, learned gate
for ds in DATASETS:
    RUNS.append(cfg(
        f"acca_attn_pre_learned_{ds}",
        ds,
        ["--use_acca", "--acca_type", "attention",
         "--acca_placement", "pre_head", "--alpha_mode", "learned"],
    ))

# Placement ablation: attention variant, post_head, learned gate
for ds in DATASETS:
    RUNS.append(cfg(
        f"acca_attn_post_learned_{ds}",
        ds,
        ["--use_acca", "--acca_type", "attention",
         "--acca_placement", "post_head", "--alpha_mode", "learned"],
    ))

# Placement ablation: linear post_head on ETTh1 + FX (ETTh1 lets us compare to
# pre_head run from run_benchmarks.py; FX is the correlated stress-test).
for ds in ["ETTh1", "fx"]:
    RUNS.append(cfg(
        f"acca_lin_post_learned_{ds}",
        ds,
        ["--use_acca", "--acca_type", "linear",
         "--acca_placement", "post_head", "--alpha_mode", "learned"],
    ))

# Alpha-mode ablation: force the gate fully open (alpha=1) so ACCA always
# writes through. This isolates whether the gate is the reason ACCA is silent.
for ds in ["ETTh1", "fx"]:
    for acca_type in ["attention", "linear"]:
        RUNS.append(cfg(
            f"acca_{acca_type[:4]}_pre_fixedone_{ds}",
            ds,
            ["--use_acca", "--acca_type", acca_type,
             "--acca_placement", "pre_head", "--alpha_mode", "fixed_one"],
        ))


def parse_summary(proc_stdout):
    """Parse train.py's Final Summary lines to extract metrics.

    train.py prints the summary as `key: value` pairs after its `Final Summary`
    header; the keys that matter for the aggregate table are test_mse,
    test_mae, best_epoch, total_training_time, and the final alpha values.
    """
    metrics = {
        "test_mse": None,
        "test_mae": None,
        "best_epoch": None,
        "total_training_time": None,
        "final_alpha_raw": None,
        "final_alpha_effective": None,
    }
    for line in proc_stdout.splitlines():
        for key in metrics:
            prefix = f"{key}:"
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                metrics[key] = value
    return metrics


def run_one(spec):
    print(f"\n--- Running {spec['name']} on {spec['dataset']} ---", flush=True)
    cmd = [
        "uv", "run", "python", "train.py",
        "--model", "PatchTST",
        "--dataset", spec["dataset"],
        *spec["args"],
    ]
    print("cmd:", " ".join(cmd), flush=True)

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=REPO_ROOT,
    )

    captured = []
    for line in process.stdout:
        print(line, end="")
        captured.append(line)
    process.wait()

    metrics = parse_summary("".join(captured))
    return {
        "name": spec["name"],
        "dataset": spec["dataset"],
        "returncode": process.returncode,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of datasets to run (default: all).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print planned runs without executing them.",
    )
    args = parser.parse_args()

    active = RUNS
    if args.datasets:
        active = [r for r in RUNS if r["dataset"] in args.datasets]
        if not active:
            raise SystemExit(f"No runs match datasets={args.datasets}")
    print(f"Planned {len(active)} run(s):")
    for r in active:
        print(f"  - {r['name']} (dataset={r['dataset']})")
    if args.dry_run:
        return

    results = [run_one(spec) for spec in active]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_path = REPO_ROOT / "scripts" / "acca_ablation_results.json"
    md_path = REPO_ROOT / "scripts" / "acca_ablation_results.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)

    grouped = defaultdict(list)
    for r in results:
        grouped[r["dataset"]].append(r)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# ACCA Ablation Results\n\nGenerated: {timestamp}\n\n")
        f.write(
            "Each row is a `PatchTST --use_acca` run with the paper's ETTh1 "
            "config (`--d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3`). "
            "`name` is the `--run_name` slug; the per-epoch alpha/MSE trace "
            "lives at `scripts/traces/<name>_trace.json`.\n\n"
        )
        for dataset, rows in grouped.items():
            f.write(f"## {dataset}\n\n")
            f.write("| Run | Test MSE | Test MAE | Best Epoch | "
                    "alpha_raw | alpha_eff | Time |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for row in rows:
                time_s = row.get("total_training_time")
                time_fmt = f"{float(time_s):.1f}s" if time_s else "N/A"
                f.write(
                    f"| `{row['name']}` | {row['test_mse']} | {row['test_mae']} | "
                    f"{row['best_epoch']} | {row['final_alpha_raw']} | "
                    f"{row['final_alpha_effective']} | {time_fmt} |\n"
                )
            f.write("\n")

    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved Markdown summary to {md_path}")


if __name__ == "__main__":
    main()
