# ACCA Ablation Results

Generated: 2026-04-23 18:10:27

Each row is a `PatchTST --use_acca` run with the paper's ETTh1 config (`--d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3`). `name` is the `--run_name` slug; the per-epoch alpha/MSE trace lives at `scripts/traces/<name>_trace.json`.

> **Note.** The 14 runs were executed in two batches. ETTh1 rows are the
> first batch (finished 2026-04-22); their full per-epoch traces live in
> `scripts/traces/`. Traffic / Air / FX rows are the second batch (finished
> 2026-04-23); their `scripts/traces/<name>_trace.json` files carry only
> the `run_name` / `config` / `summary` blocks transcribed verbatim from
> the originals (with `per_epoch: []`), and the aggregate numbers in this
> file and `acca_ablation_results.json` come from the second batch's own
> aggregate output.

## ETTh1

| Run | Test MSE | Test MAE | Best Epoch | alpha_raw | alpha_eff | Time |
| --- | --- | --- | --- | --- | --- | --- |
| `acca_attn_pre_learned_ETTh1` | 0.3809 | 0.4029 | 47 | -4.5675 | 0.0103 | 2341.3s |
| `acca_attn_post_learned_ETTh1` | 0.3812 | 0.4031 | 46 | -4.5779 | 0.0102 | 2040.3s |
| `acca_lin_post_learned_ETTh1` | 0.3813 | 0.4030 | 36 | -4.5674 | 0.0103 | 2368.8s |
| `acca_atte_pre_fixedone_ETTh1` | 0.3977 | 0.4106 | 47 | inf | 1.0000 | 6317.6s |
| `acca_line_pre_fixedone_ETTh1` | 0.5846 | 0.5219 | 52 | inf | 1.0000 | 5670.0s |

## traffic

| Run | Test MSE | Test MAE | Best Epoch | alpha_raw | alpha_eff | Time |
| --- | --- | --- | --- | --- | --- | --- |
| `acca_attn_pre_learned_traffic` | 0.5431 | 0.4666 | 45 | -4.4324 | 0.0117 | 7799.2484s |
| `acca_attn_post_learned_traffic` | 0.5425 | 0.4612 | 49 | -4.3146 | 0.0132 | 2276.2533s |

## air

| Run | Test MSE | Test MAE | Best Epoch | alpha_raw | alpha_eff | Time |
| --- | --- | --- | --- | --- | --- | --- |
| `acca_attn_pre_learned_air` | 0.2444 | 0.2307 | 39 | -4.4215 | 0.0119 | 1612.2697s |
| `acca_attn_post_learned_air` | 0.2458 | 0.2329 | 54 | -4.5347 | 0.0106 | 1675.9078s |

## fx

| Run | Test MSE | Test MAE | Best Epoch | alpha_raw | alpha_eff | Time |
| --- | --- | --- | --- | --- | --- | --- |
| `acca_attn_pre_learned_fx` | 0.0957 | 0.1928 | 86 | -4.6116 | 0.0098 | 1568.5006s |
| `acca_attn_post_learned_fx` | 0.0958 | 0.1930 | 69 | -4.6134 | 0.0098 | 846.5207s |
| `acca_lin_post_learned_fx` | 0.0962 | 0.1933 | 83 | -4.5886 | 0.0101 | 981.2979s |
| `acca_atte_pre_fixedone_fx` | 0.1183 | 0.2211 | 73 | inf | 1.0000 | 1352.0344s |
| `acca_line_pre_fixedone_fx` | 0.3059 | 0.3673 | 67 | inf | 1.0000 | 839.4074s |
