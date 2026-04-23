"""Generate the figures used by the ACCA extension section of the report.

Produces three PDF figures in report/figures/:
    alpha_trace.pdf      - per-epoch alpha_effective across ACCA ablation runs
    mse_delta.pdf        - test-MSE change of each ACCA run vs. base PatchTST
    attention_heatmap.pdf - [C x C] attention heatmap from a chosen run
                            (only generated when the .npy file exists).

Inputs:
    scripts/traces/<run_name>_trace.json     (from train.py --run_name)
    scripts/benchmark_results.json           (baseline numbers from run_benchmarks.py)
    scripts/acca_ablation_results.json       (ACCA ablation numbers)
    scripts/traces/*_attn.npy                (optional, from extract_attention.py)

Outputs are self-contained so the figures can be rebuilt without re-running
any experiments once the traces/results JSON files exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "report" / "figures"
TRACE_DIR = REPO / "scripts" / "traces"


def load_traces():
    """Return a dict of {run_name: trace_dict} for every *_trace.json found."""
    traces = {}
    if not TRACE_DIR.exists():
        return traces
    for path in sorted(TRACE_DIR.glob("*_trace.json")):
        with path.open(encoding="utf-8") as f:
            traces[path.stem.removesuffix("_trace")] = json.load(f)
    return traces


def plot_alpha_trace(traces, out_path):
    """Line plot of alpha_effective vs. epoch, one line per ACCA run.

    Lets the reader see whether any configuration opened the gate. Runs with
    fixed-mode alpha (e.g. fixed_one) show a flat horizontal line; learned
    runs converge toward some equilibrium value.
    """
    acca_traces = {
        name: t for name, t in traces.items()
        if t.get("config", {}).get("use_acca")
    }
    if not acca_traces:
        print("[plot_alpha_trace] no ACCA traces found; skipping")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # Deterministic colour assignment keyed by run name so the figure is
    # reproducible even when traces are added incrementally. Runs with an
    # empty per_epoch array (e.g. the second-batch stubs) are silently
    # skipped to avoid ghost legend entries.
    for name in sorted(acca_traces):
        trace = acca_traces[name]
        epochs = [p["epoch"] for p in trace["per_epoch"]]
        alpha = [p["alpha_effective"] for p in trace["per_epoch"]]
        if not epochs or any(a is None for a in alpha):
            continue
        style = "--" if "fixedone" in name else "-"
        ax.plot(epochs, alpha, style, label=name, linewidth=1.4, alpha=0.85)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\alpha_{\mathrm{effective}} = \sigma(\alpha_{\mathrm{raw}})$")
    ax.set_title(r"ACCA gate evolution across ablation runs")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_alpha_trace] wrote {out_path}")


def _load_baseline():
    """Load scripts/benchmark_results.json -> {dataset: {model: (mse, mae)}}.

    The benchmark JSON produced by Ziming's scripts/run_benchmarks.py only
    covers traffic/air/fx; ETTh1 lives in the reproduction-validation section
    of the report and is hard-coded here so the plot still has a reference
    point for ETTh1 bars.
    """
    path = REPO / "scripts" / "benchmark_results.json"
    out = {"ETTh1": {"PatchTST": (0.381, 0.403)}}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for row in data.get("results", []):
        ds = row["dataset"]
        out.setdefault(ds, {})[row["model"]] = (
            float(row["test_mse"]) if row["test_mse"] else None,
            float(row["test_mae"]) if row["test_mae"] else None,
        )
    return out


def _load_acca_results():
    """Load scripts/acca_ablation_results.json -> [{name,dataset,test_mse,...}]."""
    path = REPO / "scripts" / "acca_ablation_results.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data.get("results", []):
        try:
            mse = float(r["test_mse"]) if r["test_mse"] else None
            mae = float(r["test_mae"]) if r["test_mae"] else None
        except (TypeError, ValueError):
            mse = mae = None
        rows.append({
            "name": r["name"],
            "dataset": r["dataset"],
            "mse": mse,
            "mae": mae,
            "alpha_effective": (
                float(r["final_alpha_effective"])
                if r.get("final_alpha_effective") else None
            ),
        })
    return rows


def plot_mse_delta(out_path):
    """Grouped bar chart: MSE delta vs. base PatchTST, per (dataset, ACCA run)."""
    baseline = _load_baseline()
    acca = _load_acca_results()
    if not baseline or not acca:
        print("[plot_mse_delta] missing baseline or ACCA results; skipping")
        return

    datasets = sorted({r["dataset"] for r in acca})
    runs_by_dataset = {ds: [r for r in acca if r["dataset"] == ds] for ds in datasets}
    max_runs = max(len(v) for v in runs_by_dataset.values())
    if max_runs == 0:
        print("[plot_mse_delta] no ACCA runs to plot")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    width = 0.8 / max_runs
    x = np.arange(len(datasets))

    # For each dataset we draw up to max_runs bars; deltas are (ACCA - base)/base.
    # Labels come from the run whose configuration is most representative of
    # the slot, chosen as the first dataset that has a run in that slot (since
    # the matrix is built with the same ordering per dataset).
    for i in range(max_runs):
        deltas = []
        slot_label = None
        for ds in datasets:
            rows = runs_by_dataset[ds]
            if i >= len(rows) or rows[i]["mse"] is None:
                deltas.append(np.nan)
                continue
            base = baseline.get(ds, {}).get("PatchTST", (None, None))[0]
            if base is None:
                deltas.append(np.nan)
                continue
            deltas.append(100.0 * (rows[i]["mse"] - base) / base)
            if slot_label is None:
                slot_label = rows[i]["name"].replace("acca_", "").rsplit("_", 1)[0]
        positions = x + (i - max_runs / 2) * width + width / 2
        ax.bar(positions, deltas, width, label=slot_label or f"run{i}")

    ax.axhline(0.0, linewidth=0.8, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r"$\Delta$ Test MSE vs. PatchTST base (%)")
    ax.set_title("ACCA test-MSE change relative to the channel-independent baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_mse_delta] wrote {out_path}")


def plot_attention_heatmap(npy_path, out_path, title=None):
    """Draw a [C x C] cross-channel attention heatmap.

    Typical input comes from scripts/extract_attention.py run on a trained
    attention-ACCA checkpoint. Larger off-diagonal mass means the module is
    actively mixing channels; a ~uniform matrix means it collapsed to an
    average; a near-identity matrix means it preserves channel independence.
    """
    if not Path(npy_path).exists():
        print(f"[plot_attention_heatmap] {npy_path} not found; skipping")
        return
    attn = np.load(npy_path)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key channel")
    ax.set_ylabel("Query channel")
    ax.set_title(title or f"Average ACCA attention ({Path(npy_path).stem})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_attention_heatmap] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_npy", type=str, default=None,
                        help="Optional .npy file from extract_attention.py.")
    parser.add_argument("--attn_title", type=str, default=None)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    traces = load_traces()

    plot_alpha_trace(traces, FIG_DIR / "alpha_trace.pdf")
    plot_mse_delta(FIG_DIR / "mse_delta.pdf")

    if args.attn_npy:
        plot_attention_heatmap(
            args.attn_npy,
            FIG_DIR / "attention_heatmap.pdf",
            title=args.attn_title,
        )


if __name__ == "__main__":
    main()
