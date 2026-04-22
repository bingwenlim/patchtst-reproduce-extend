# ACCA Ablation Results

Generated: 2026-04-23 00:50:39

Each row is a `PatchTST --use_acca` run with the paper's ETTh1 config (`--d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3`). `name` is the `--run_name` slug; the per-epoch alpha/MSE trace lives at `scripts/traces/<name>_trace.json`.

## ETTh1

| Run | Test MSE | Test MAE | Best Epoch | alpha_raw | alpha_eff | Time |
| --- | --- | --- | --- | --- | --- | --- |
| `acca_attn_pre_learned_ETTh1` | 0.3809 | 0.4029 | 47 | -4.5675 | 0.0103 | 2341.3s |
| `acca_attn_post_learned_ETTh1` | 0.3812 | 0.4031 | 46 | -4.5779 | 0.0102 | 2040.3s |
| `acca_lin_post_learned_ETTh1` | 0.3813 | 0.4030 | 36 | -4.5674 | 0.0103 | 2368.8s |
| `acca_atte_pre_fixedone_ETTh1` | 0.3977 | 0.4106 | 47 | inf | 1.0000 | 6317.6s |
| `acca_line_pre_fixedone_ETTh1` | 0.5846 | 0.5219 | 52 | inf | 1.0000 | 5670.0s |

