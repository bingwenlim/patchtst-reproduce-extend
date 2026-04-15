# Reproduction Results

## ETTh1 (pred_len=96)

### Summary

| Model | Config | MSE (Paper) | MSE (Ours) | MAE (Paper) | MAE (Ours) | Best Epoch | Time |
|-------|--------|-------------|------------|-------------|------------|------------|------|
| PatchTST | ETTh1 | 0.375 | 0.378 | 0.399 | 0.400 | 48 | 262s (MPS) |
| DLinear | default | 0.375 | 0.374 | 0.399 | 0.397 | 43 | 9s (MPS) |
| Autoformer | default | 0.435 | 0.528 | 0.446 | 0.491 | 6 | 1205s (MPS) |
| Autoformer | ETTh1 | — | 0.684 | — | 0.556 | 11 | 146s (MPS) |

### Experiment configs

1) **ETTh1** — PatchTST's paper config for small datasets: `--d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3`
2) **default** — our train.py defaults (paper general defaults): d_model=128, n_heads=16, e_layers=3, d_ff=256, dropout=0.2, seq_len=336

All runs share: lr=1e-4, batch_size=128, epochs=100, patience=10, seed=42, type3 LR schedule.

### PatchTST

```bash
uv run python train.py --model PatchTST --d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3
```

Notes:
- Paper uses seed 2021, patience 100. We use seed 42, patience 10.
- Paper reports dropout 0.2 in text (Appendix A.1.4), but etth1.sh uses 0.3. We used 0.3.

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

The paper's 0.435 was obtained by running Autoformer across 6 different seq_len values and picking the best. Our default run (0.528) uses seq_len=336 only. The ETTh1 config (d_model=16) is too small for Autoformer's encoder-decoder architecture, resulting in 0.684. Autoformer code is from Time-Series-Library and was not tuned.
