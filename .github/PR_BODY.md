# ACCA ablation + extension analysis (Wang Yuxiao)

Closes my portion of the proposal deliverables: ACCA ablation experiments, new report figures, and the Extension / Conclusion rewrite.

**Status of the branch.** This branch was forked from `origin/Run-result` (Ziming's in-flight documentation branch) because that was the latest code that had the ACCA module, the four-dataset benchmark results, and the dataset prose together. `main` has since diverged via PR #14 (dataset write-up). Please **merge Run-result into `main` first** (resolving conflicts with PR #14), then I can rebase this branch onto the new `main` and open the final PR. Attempting the rebase now produces conflicts in `report/patchtst.tex` / `report/patchtst.{aux,out,pdf}` / `.DS_Store` that are essentially Run-result vs. PR #14 and should be resolved by whoever merges Run-result.

## Summary

- **`train.py`**: logs a per-epoch JSON trace (`scripts/traces/<run_name>_trace.json`) with `alpha_raw`, `alpha_effective`, `train_mse`, `val_mse`, LR and epoch time. Adds `--run_name` and `--trace_dir` CLI flags; checkpoint filename now uses `run_name` so parallel ablation runs no longer clobber each other.
- **`models/PatchTST.py`**: `AdaptiveCrossChannelAttention` now caches `last_attn_weights` when `record_attn=True` and the module is in eval mode, enabling the cross-channel attention heatmap figure.
- **`scripts/run_acca_ablations.py`**: the 14-run ACCA ablation matrix (attention vs linear × pre-head vs post-head × learned vs fixed_one, across ETTh1 / Traffic / Air / FX). Supports `--datasets` filter and `--dry_run`. Aggregates into `scripts/acca_ablation_results.{json,md}`.
- **`scripts/extract_attention.py`**: standalone script that re-loads a trained attention-ACCA checkpoint, runs the test set with `record_attn=True`, and dumps the `[C x C]` mean attention matrix as `.npy`.
- **`scripts/plot_acca.py`**: produces the three Section 5 figures:
  - `report/figures/alpha_trace.pdf` — per-epoch α evolution across ablations (symlog).
  - `report/figures/mse_delta.pdf` — ΔMSE vs. CI baseline, grouped by dataset.
  - `report/figures/attention_heatmap.pdf` — optional, from the `.npy`.
- **`report/patchtst.tex`**: Section 5 completely rewritten into 7 subsections (Motivation → Architecture with formal $\mathbf{H}^{\text{attn}}/\mathbf{H}^{\text{lin}}$ definitions → Experimental Setup → Main Results table across 4 datasets → Ablations (placement + gate-mode + operator) with trace figure → Discussion → Limitations/Future Work). Section 6 (Conclusion) reframes the negative ACCA result as a positive empirical contribution about CI robustness.
- **References**: +3 (Crossformer, iTransformer, TSMixer) cited throughout Sections 5–6.

## Main takeaway (full 14-run matrix)

The learnable gate converges to $\alpha_{\text{effective}} \in [0.0098, 0.0132]$ on every learnable run across ETTh1 / Traffic / Air / FX — gradient descent actively chooses not to open the cross-channel pathway, even on FX with its 42 strongly correlated currency pairs. But the variants aren't all equivalent:

- **Linear, pre-head, learned** is the one truly neutral configuration (ΔMSE within ±0.4% everywhere).
- **Attention** leaks enough cross-channel gradient noise through the nearly-closed gate to degrade MSE by **+2.3% (Traffic), +10.1% (Air), +7.5% (FX)** — the extra channel-conditioned parameters are the likely culprit.
- **`fixed_one`** (gate forced open) collapses MSE, with damage that scales with channel count:
  - linear: **+53% on ETTh1 (C=7)** → **+244% on FX (C=42)**
  - attention: **+4.4% on ETTh1** → **+23.7% on FX** (the MHA residual + LayerNorm absorbs most of the damage)
- **Placement** (pre-head vs. post-head) makes no measurable difference on any dataset.

Net: under PatchTST's CI regime, a local late-stage cross-channel mixer is not merely redundant — it is *net harmful*. The learned gate correctly identifies this and stays closed. Consistent with but stronger than DLinear's observation; complements the backbone-level redesigns in Crossformer / iTransformer / TSMixer.

## How to reproduce

```bash
# Full 14-run matrix (takes several hours; use a fast machine)
uv run python scripts/run_acca_ablations.py

# ETTh1 only (feasible on CPU overnight)
uv run python scripts/run_acca_ablations.py --datasets ETTh1

# Regenerate the figures after runs complete
uv run python scripts/plot_acca.py

# (optional) extract + plot an attention heatmap
uv run python scripts/extract_attention.py \
  --checkpoint checkpoints/acca_attn_pre_learned_fx.pt \
  --dataset fx --out scripts/traces/fx_attn.npy
uv run python scripts/plot_acca.py --attn_npy scripts/traces/fx_attn.npy
```

## Notes for reviewers

- All 14 ablation runs are reported in Tables 5.2–5.4 — none are TBD. The runs were done in two batches: the ETTh1 batch has per-epoch traces in `scripts/traces/*_trace.json`; the Traffic / Air / FX batch's summary numbers are transcribed into `scripts/acca_ablation_results.{json,md}` verbatim from that batch's own aggregate MD, and its raw per-epoch trace JSONs are not checked in (this is noted in the MD/JSON header).
- `report/figures/alpha_trace.png` and `report/figures/mse_delta.png` are the full 14-run plots from the second batch's `plot_acca.py` run.
- PDF was not re-compiled locally (no TeX toolchain on this machine). Please compile `report/patchtst.tex` via Overleaf or local MikTeX before merging.

## Not in scope

- Traffic / Air dataset prose (Huang Wenhui + Chu Shi Yuan).
- Presentation slides / video recording.
- ETTh1 baseline re-runs (already produced by `scripts/run_benchmarks.py`).
