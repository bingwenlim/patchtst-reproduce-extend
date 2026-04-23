# Reproduction Results

All timing columns are wall-clock. `Train Time` is the total training time reported by `run_training` as `total_training_time` (including early stopping). `Inference Time` is `test_inference_time`: the wall-clock time to run the best-epoch model over the full test loader on the same device. Both are printed by `train.py` in the final summary and captured by `scripts/run_benchmarks.py`.

## ETTh1 (pred_len=96)

### Summary

| Model           | Config           | MSE (Paper) | MSE (Reproduced) | MAE (Paper) | MAE (Reproduced) | Best Epoch | Time        |
| --------------- | ---------------- | ----------- | ---------- | ----------- | ---------- | ---------- | ----------- |
| PatchTST        | ETTh1            | 0.375       | 0.381      | 0.399       | 0.403      | 37         | 196s (MPS)  |
| PatchTST (ACCA) | ETTh1 (linear)   | —           | 0.381      | —           | 0.403      | 37         | 169s (MPS)  |
| PatchTST (ACCA) | default (linear) | —           | 0.377      | —           | 0.399      | 3          | 225s (MPS)  |
| DLinear         | default          | 0.375       | 0.374      | 0.399       | 0.397      | 43         | 9s (MPS)    |
| Autoformer      | default          | 0.435       | 0.528      | 0.446       | 0.491      | 6          | 1205s (MPS) |
| Autoformer      | ETTh1            | —           | 0.684      | —           | 0.556      | 11         | 146s (MPS)  |

### Experiment configs

1) **ETTh1** — PatchTST's paper config for small datasets: `--d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3`
2) **default** — the baseline implementation configuration (paper general defaults): d_model=128, n_heads=16, e_layers=3, d_ff=256, dropout=0.2, seq_len=336

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

DLinear only uses seq_len, pred_len, enc_in, and moving_avg from configs — model-specific hyperparameters don't apply.

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

| Model           | Config           | MSE (Ours) | MAE (Ours) | Best Epoch | Train Time  | Inference Time |
| --------------- | ---------------- | ---------- | ---------- | ---------- | ----------- | -------------- |
| PatchTST        | paper config     | 0.531      | 0.461      | 58         | 2846.3s     | 2.307s         |
| PatchTST (ACCA) | paper config     | 0.531      | 0.461      | 52         | 2842.0s     | 2.752s         |
| DLinear         | default          | 0.581      | 0.517      | 15         | 24.6s       | 0.131s         |
| Autoformer      | paper config     | 0.685      | 0.539      | 10         | 522.5s      | 0.809s         |


## Air Quality (pred_len=96)

### Summary

| Model           | Config           | MSE (Ours) | MAE (Ours) | Best Epoch | Train Time  | Inference Time |
| --------------- | ---------------- | ---------- | ---------- | ---------- | ----------- | -------------- |
| PatchTST        | paper config     | 0.222      | 0.202      | 50         | 1828.0s     | 1.602s         |
| PatchTST (ACCA) | paper config     | 0.221      | 0.201      | 47         | 1794.4s     | 1.907s         |
| DLinear         | default          | 0.278      | 0.289      | 11         | 13.7s       | 0.115s         |
| Autoformer      | paper config     | 0.404      | 0.337      | 9          | 428.8s      | 0.955s         |


## FX (pred_len=96)

### Summary

| Model           | Config           | MSE (Ours) | MAE (Ours) | Best Epoch | Train Time  | Inference Time |
| --------------- | ---------------- | ---------- | ---------- | ---------- | ----------- | -------------- |
| PatchTST        | paper config     | 0.089      | 0.185      | 88         | 1047.0s     | 0.549s         |
| PatchTST (ACCA) | paper config     | 0.089      | 0.185      | 89         | 1044.8s     | 0.542s         |
| DLinear         | default          | 0.155      | 0.260      | 47         | 8.1s        | 0.025s         |
| Autoformer      | paper config     | 0.166      | 0.287      | 65         | 238.0s      | 0.104s         |

