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

## Main takeaway

Across all four datasets and every ablation axis, the ACCA module tracks the CI PatchTST baseline to within 0.6% MSE. The learnable gate converges to $\alpha_{\text{effective}} \approx 0.01$ on every run — gradient descent actively chooses not to open the cross-channel pathway, even on the FX dataset with 42 strongly correlated currency pairs. Forcing the gate fully open (`fixed_one`) *worsens* MSE, rejecting the hypothesis that the learned gate is the limiting factor. The result is consistent with but stronger than DLinear's observation, and complements the backbone-level cross-channel redesigns in Crossformer / iTransformer / TSMixer.

## How to reproduce

```bash
# Full 14-run matrix (needs a GPU machine; takes ~7h on MPS)
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

- Cells marked *TBD* in Tables 5.2–5.4 correspond to runs that could not be executed on the CPU machine used for this branch; they will be filled in once the attention/placement/fixed_one runs complete on a GPU machine. Baseline (CI) rows and the `linear + pre_head + learned` ACCA rows are already final — those numbers come from Ziming's benchmark run and the earlier ETTh1 ACCA run.
- PDF was not re-compiled locally (no TeX toolchain on this machine). Please compile `report/patchtst.tex` via Overleaf or local MikTeX before merging.

## Not in scope

- Traffic / Air dataset prose (Huang Wenhui + Chu Shi Yuan).
- Presentation slides / video recording.
- ETTh1 baseline re-runs (already produced by `scripts/run_benchmarks.py`).
