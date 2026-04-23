# Reproduction Results

All timing columns are wall-clock. `Train Time` is the total training time reported by `run_training` as `total_training_time` (including early stopping). `Inference Time` is `test_inference_time`: the wall-clock time to run the best-epoch model over the full test loader on the same device. Both are printed by `train.py` in the final summary and captured by `scripts/run_benchmarks.py`.

## ETTh1 (pred_len=96)

### Summary

| Model           | Config           | MSE (Paper) | MSE (Reproduced) | MAE (Paper) | MAE (Reproduced) | Best Epoch | Time        |
| --------------- | ---------------- | ----------- | ---------- | ----------- | ---------- | ---------- | ----------- |
| PatchTST        | ETTh1            | 0.375       | 0.381      | 0.399       | 0.403      | 37         | 196s (MPS)  |
| PatchTST (ACCA) | ETTh1 (linear)   | ù?          | 0.381      | ù?          | 0.403      | 37         | 169s (MPS)  |
| PatchTST (ACCA) | default (linear) | ù?          | 0.377      | ù?          | 0.399      | 3          | 225s (MPS)  |
| DLinear         | default          | 0.375       | 0.374      | 0.399       | 0.397      | 43         | 9s (MPS)    |
| Autoformer      | default          | 0.435       | 0.528      | 0.446       | 0.491      | 6          | 1205s (MPS) |
| Autoformer      | ETTh1            | ù?          | 0.684      | ù?          | 0.556      | 11         | 146s (MPS)  |

### Experiment configs

1) **ETTh1** ù?PatchTST's paper config for small datasets: `--d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3`
2) **default** ù?the baseline implementation configuration (paper general defaults): d_model=128, n_heads=16, e_layers=3, d_ff=256, dropout=0.2, seq_len=336

All runs share: lr=1e-4, batch_size=128, epochs=100, patience=10, seed=42, type3 LR schedule.

### PatchTST

```bash
uv run python train.py --model PatchTST --d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3
```

Notes:
- Paper uses seed 2021, patience 100. This implementation utilizes seed 42, patience 10.
- Paper reports dropout 0.2 in text (Appendix A.1.4), but etth1.sh uses 0.3. A rate of 0.3 was applied in this replication.

### PatchTST (ACCA)

#### ETTh1 Config
```bash
uv run python train.py \
  --model PatchTST \
  --dataset ETTh1 \
  --d_model 16 \
  --n_heads 4 \
  --d_ff 128 \
  --dropout 0.3 \
  --use_acca \
  --acca_type linear
```

Notes:
- Full run using ACCA with linear mapping (ETTh1 minimal config).
- Test MSE: `0.3813`, Test MAE: `0.4031`
- Best Epoch: `37` (Final Epoch 47)
- ACCA `alpha_raw` mean: `-4.5674` (at epoch 47)
- ACCA `alpha_effective` mean: `0.0103` (at epoch 47)

#### Default Config
```bash
uv run python train.py \
  --model PatchTST \
  --dataset ETTh1 \
  --use_acca \
  --acca_type linear
```

Notes:
- Full run using ACCA with linear mapping (larger default hyperparameters).
- Test MSE: `0.3774`, Test MAE: `0.3993`
- Best Epoch: `3` (Final Epoch 13)
- ACCA `alpha_raw` mean: `-4.5767` (at epoch 13)
- ACCA `alpha_effective` mean: `0.0102` (at epoch 13)

### DLinear

```bash
uv run python train.py --model DLinear
```

DLinear only uses seq_len, pred_len, enc_in, and moving_avg from configs ù?model-specific hyperparameters don't apply.

### Autoformer

```bash
# default config
uv run python train.py --model Autoformer

# ETTh1 config
uv run python train.py --model Autoformer --d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3 --seq_len 96
```

The paper's 0.435 was obtained by running Autoformer across 6 different seq_len values and picking the best. The reproduced baseline run (0.528) uses seq_len=336 only. The ETTh1 config (d_model=16) is too small for Autoformer's encoder-decoder architecture, resulting in 0.684. Autoformer code is from Time-Series-Library and was not tuned.

## Traffic (pred_len=96)

### Summary

| Model                        | Config                   | MSE    | MAE    | Best Epoch | Time    |
| ---------------------------- | ------------------------ | ------ | ------ | ---------- | ------- |
| PatchTST                     | paper config             | 0.531  | 0.461  | 58         | 2962.6s    |
| PatchTST (ACCA linear, pre)  | learned                  | 0.530  | 0.460  | 58         | 3051.2s    |
| PatchTST (ACCA attn,   pre)  | learned                  | 0.5431 | 0.4666 | 45         | 7799.2484s |
| PatchTST (ACCA attn,   post) | learned                  | 0.5425 | 0.4612 | 49         | 2276.2533s |
| DLinear                      | default                  | 0.581  | 0.517  | 15         | 20.5s      |
| Autoformer                   | paper config             | 0.685  | 0.539  | 10         | 526.3s     |


## Air Quality (pred_len=96)

### Summary

| Model                        | Config                   | MSE    | MAE    | Best Epoch | Time    |
| ---------------------------- | ------------------------ | ------ | ------ | ---------- | ------- |
| PatchTST                     | paper config             | 0.222  | 0.202  | 50         | 1668.3s    |
| PatchTST (ACCA linear, pre)  | learned                  | 0.221  | 0.201  | 50         | 1731.5s    |
| PatchTST (ACCA attn,   pre)  | learned                  | 0.2444 | 0.2307 | 39         | 1612.2697s |
| PatchTST (ACCA attn,   post) | learned                  | 0.2458 | 0.2329 | 54         | 1675.9078s |
| DLinear                      | default                  | 0.278  | 0.289  | 11         | 11.7s      |
| Autoformer                   | paper config             | 0.404  | 0.337  | 9          | 429.9s     |


## FX (pred_len=96)

### Summary

| Model                        | Config                   | MSE    | MAE    | Best Epoch | Time    |
| ---------------------------- | ------------------------ | ------ | ------ | ---------- | ------- |
| PatchTST                     | paper config             | 0.089  | 0.185  | 88         | 934.2s     |
| PatchTST (ACCA linear, pre)  | learned                  | 0.089  | 0.185  | 88         | 981.6s     |
| PatchTST (ACCA linear, post) | learned                  | 0.0962 | 0.1933 | 83         | 981.2979s  |
| PatchTST (ACCA attn,   pre)  | learned                  | 0.0957 | 0.1928 | 86         | 1568.5006s |
| PatchTST (ACCA attn,   post) | learned                  | 0.0958 | 0.1930 | 69         | 846.5207s  |
| PatchTST (ACCA attn,   pre)  | **fixed_one**            | 0.1183 | 0.2211 | 73         | 1352.0344s |
| PatchTST (ACCA linear, pre)  | **fixed_one**            | 0.3059 | 0.3673 | 67         | 839.4074s  |
| DLinear                      | default                  | 0.155  | 0.260  | 47         | 8.4s       |
| Autoformer                   | paper config             | 0.166  | 0.287  | 65         | 238.8s     |

## ACCA full ablation (pred_len=96)

The full 14-run ACCA ablation matrix lives in `scripts/acca_ablation_results.md` and `scripts/acca_ablation_results.json`. The ETTh1 batch's per-epoch traces are in `scripts/traces/`; the Traffic / Air / FX batch's summary numbers are transcribed into the JSON verbatim from its own aggregate MD. Three headline observations:

1. **The gate stays closed.** Every learnable run converges to $\alpha_{\text{effective}} \in [0.0098, 0.0132]$ on every dataset.
2. **Attention leaks noise when C is large.** On ETTh1 (C=7) attention and linear agree to $0.0001$ MSE; on Traffic / Air / FX the attention variant is $+2.3\% / +10.1\% / +7.5\%$ worse than the CI baseline even with $\alpha \approx 0.01$.
3. **Forcing `fixed_one` scales with C.** Linear `fixed_one` collapses $+53\%$ on ETTh1 vs. $+244\%$ on FX; attention `fixed_one` collapses $+4.4\%$ vs. $+23.7\%$. The MHA residual + LayerNorm in the attention variant absorbs most of the damage.

