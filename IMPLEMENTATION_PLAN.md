# PatchTST Implementation Guide

This guide walks through everything needed to understand, implement, and verify PatchTST from scratch. It is written for someone who has a basic understanding of PyTorch and neural networks but is new to time series forecasting and Transformers.

**Primary reference:** Nie et al. (2023), *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR 2023. https://arxiv.org/abs/2211.14730

---

## Step 1: Understand the Problem — Long-Term Time Series Forecasting (LTSF)

### What is the task?
Given a historical window of a multivariate time series, predict the next `pred_len` timesteps. For example, given the last 336 hours of weather sensor readings (temperature, humidity, wind speed, etc.), predict the next 96 hours.

Formally:
- Input: `X ∈ R^(seq_len × C)` where `seq_len` is the lookback window length and `C` is the number of variables (channels)
- Output: `X̂ ∈ R^(pred_len × C)` — the predicted future values

### Standard benchmark settings
The paper evaluates on prediction lengths of **96, 192, 336, and 720** timesteps. These are the four numbers you will see in every results table. You need to run all four to reproduce the paper's results.

### What makes this hard?
- Long sequences are expensive for Transformers due to O(T²) attention complexity
- Time series have local temporal patterns (trends, seasonality) that point-wise tokenization fails to capture well
- Multivariate series can have complex inter-variable dependencies, but naively modeling them leads to overfitting

### Important note on data format
In PyTorch, batched data is typically shaped `[B, seq_len, C]` where B is batch size. Some operations require transposing to `[B, C, seq_len]`. Keep track of dimension order carefully throughout the implementation — this is a common source of bugs.

---

## Step 2: Understand the Transformer Encoder

### Why only the encoder?
PatchTST is an encoder-only model. Unlike sequence-to-sequence models (e.g., Autoformer) that use both an encoder and decoder, PatchTST directly maps the encoded patch representations to the forecast via a linear head. This simplifies the architecture significantly.

### Key components of a Transformer encoder layer
Each encoder layer consists of:

1. **Multi-Head Self-Attention (MHSA):** Each token (patch) attends to all other tokens. The attention scores determine how much each patch should "look at" other patches when building its representation. With `h` heads and model dimension `d_model`, each head operates on dimension `d_model / h`.

2. **Feed-Forward Network (FFN):** A two-layer MLP applied independently to each token after attention. Typically expands to `4 * d_model` in the hidden layer, then projects back down.

3. **Layer Normalization:** Applied before each sub-layer (Pre-LN variant used in PatchTST). Stabilizes training.

4. **Residual Connections:** The input to each sub-layer is added back to its output. Prevents vanishing gradients.

### Implementation note
You do **not** need to implement attention from scratch. PyTorch's `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` implement all of the above. PatchTST uses these directly. The key hyperparameters are:
- `d_model` — embedding dimension (default: 128)
- `nhead` — number of attention heads (default: 16)
- `num_layers` — number of stacked encoder layers (default: 3)
- `dim_feedforward` — FFN hidden dimension (default: 256)
- `dropout` — dropout rate (default: 0.2)

### Important: norm_first=True
PatchTST uses Pre-Layer Normalization (norm applied before attention, not after). When using `nn.TransformerEncoderLayer`, set `norm_first=True` to match the paper.

---

## Step 3: Understand the Patching Mechanism

### The core idea
Instead of treating each individual timestep as a token (which gives a sequence of length `seq_len` to the Transformer), PatchTST divides the time series into overlapping fixed-length segments called **patches**, and treats each patch as a token. This is directly analogous to how Vision Transformers (ViT) divide an image into patches.

### How patching works
Given a univariate time series of length `seq_len`:

1. **Padding:** Append `patch_len - stride` zeros (or replicate the last value) to the end of the series so the last patch is complete. The paper uses replication padding.

2. **Unfolding:** Use a sliding window of size `patch_len` with step `stride` to extract patches. PyTorch's `Tensor.unfold(dimension, size, step)` does this in one line.

3. **Result:** You get `N` patches, each of length `patch_len`, where:
   ```
   N = floor((seq_len - patch_len) / stride) + 2
   ```
   With default settings (`seq_len=336`, `patch_len=16`, `stride=8`): N = 42 patches.

4. **Linear projection:** Each patch of length `patch_len` is projected to `d_model` via a learned linear layer (no bias). This is the patch embedding.

5. **Positional encoding:** Standard sinusoidal positional encoding is added to the patch embeddings so the model knows the order of patches.

### Why this helps
- Reduces the sequence length from `seq_len` (e.g., 336) to `N` (e.g., 42), reducing attention complexity from O(336²) to O(42²) — roughly 64× fewer operations.
- Each patch contains `patch_len` consecutive timesteps, giving the model local temporal context per token rather than a single point.
- The paper shows this inductive bias significantly improves forecasting accuracy.

### Important: stride < patch_len means overlap
With `patch_len=16` and `stride=8`, consecutive patches overlap by 8 timesteps. This is intentional — it ensures no information is lost at patch boundaries.

---

## Step 4: Understand Channel Independence (CI)

### What it means
In a multivariate time series with `C` variables, the Channel-Independent approach processes each variable completely separately through the same shared Transformer backbone. There is no explicit cross-variable attention.

### How it is implemented
The trick is a simple reshape:
- Input: `[B, seq_len, C]`
- Transpose to: `[B, C, seq_len]`
- Reshape to: `[B*C, seq_len]` — treat each channel of each sample as an independent sequence
- Run through the Transformer
- Reshape output back to: `[B, C, pred_len]`
- Transpose to: `[B, pred_len, C]`

This means a batch of size 32 with 7 channels is processed as a batch of 224 independent univariate sequences. The Transformer weights are shared across all channels.

### Why this works better than channel mixing
Naively concatenating all channels and running them through a shared Transformer (Channel-Dependent / CD) tends to overfit because the model tries to learn cross-variable relationships that may not generalize. CI acts as a strong regularizer. The paper shows CI consistently outperforms CD on most benchmarks.

### Important note for the extension
The architectural extension in this project (Cross-Channel Attention Module) is specifically designed to address the limitation of CI — it adds cross-variable interaction *after* the CI Transformer encoder, rather than replacing it.

---

## Step 5: Full Architecture Data Flow

This is the complete forward pass, with tensor shapes at each stage. Use this as a checklist when implementing.

```
Input:          [B, seq_len, C]

# Channel Independence reshape
Transpose:      [B, C, seq_len]
Reshape:        [B*C, seq_len]

# Patching
Pad:            [B*C, seq_len + padding]
Unfold:         [B*C, N, patch_len]

# Patch Embedding
Linear:         [B*C, N, d_model]
+ Pos Encoding: [B*C, N, d_model]
Dropout:        [B*C, N, d_model]

# Transformer Encoder (num_layers stacked)
Encoder:        [B*C, N, d_model]

# Prediction Head
Flatten:        [B*C, N * d_model]
Linear:         [B*C, pred_len]

# Reshape back
Reshape:        [B, C, pred_len]
Transpose:      [B, pred_len, C]

Output:         [B, pred_len, C]
```

Read Figure 1 of the paper alongside this to see the visual representation.

---

## Step 6: Implementation Order

All PatchTST code goes in `models/PatchTST.py`. Implement in this order:

### 6.1 PatchEmbedding
Takes a batch of univariate sequences `[B*C, seq_len]` and returns patch embeddings `[B*C, N, d_model]`. Internally: pad → unfold → linear projection → add positional encoding → dropout.

### 6.2 PatchTSTEncoder
Wraps `nn.TransformerEncoder`. Takes `[B*C, N, d_model]`, returns `[B*C, N, d_model]`. Remember `norm_first=True` and `batch_first=True` in `nn.TransformerEncoderLayer`.

### 6.3 PredictionHead
Takes `[B*C, N, d_model]`, flattens to `[B*C, N*d_model]`, applies a linear layer to get `[B*C, pred_len]`. Optionally add a dropout before the linear layer.

### 6.4 PatchTST (main model)
Combines the above. Handles the channel independence reshape at the start and the reshape back at the end. The `forward` method signature should match the baselines: `forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None)` — the extra arguments are unused but needed for a consistent training loop interface.

### Sanity check after each sub-component
After implementing each piece, test it with a dummy tensor to verify the output shape is correct before moving on. For example:
```python
x = torch.randn(32, 336, 7)  # B=32, seq_len=336, C=7
model = PatchTST(seq_len=336, pred_len=96, enc_in=7)
out = model(x)
assert out.shape == (32, 96, 7)
```

---

## Step 7: Training Setup and Verification

### Dataset: ETTh1
ETTh1 (Electricity Transformer Temperature, hourly) is the standard sanity-check dataset. It has 7 variables and ~17,000 timesteps. Download from: https://github.com/zhouhaoyi/ETDataset

Standard train/val/test split: 12/4/4 months (roughly 8640/2880/2880 timesteps).

### Data preprocessing
- Normalize each channel independently using the **training set** mean and standard deviation. Do not use val/test statistics — this would be data leakage.
- Use a sliding window to create samples: each sample is `(x, y)` where `x` is `seq_len` timesteps and `y` is the next `pred_len` timesteps.

### Training loop essentials
- Optimizer: Adam with `lr=1e-4`
- Loss: MSE between predicted and actual values
- Batch size: 128
- Train for up to 100 epochs with early stopping (patience=10) based on validation MSE

### Evaluation metrics
- **MSE** (Mean Squared Error): `mean((y_pred - y_true)^2)`
- **MAE** (Mean Absolute Error): `mean(|y_pred - y_true|)`
- Compute on the **normalized** scale (same as the paper)

### Expected results on ETTh1 (from paper, Table 1)

**PatchTST**
| pred_len | MSE   | MAE   |
|----------|-------|-------|
| 96       | 0.370 | 0.400 |
| 192      | 0.413 | 0.429 |
| 336      | 0.422 | 0.440 |
| 720      | 0.447 | 0.468 |

**DLinear**
| pred_len | MSE   | MAE   |
|----------|-------|-------|
| 96       | 0.386 | 0.400 |
| 192      | 0.437 | 0.432 |
| 336      | 0.481 | 0.459 |
| 720      | 0.456 | 0.482 |

**Autoformer**
| pred_len | MSE   | MAE   |
|----------|-------|-------|
| 96       | 0.449 | 0.459 |
| 192      | 0.500 | 0.482 |
| 336      | 0.521 | 0.496 |
| 720      | 0.514 | 0.512 |

If your numbers are in the same ballpark (within ~5%), the implementation is correct. Exact reproduction is difficult due to random seeds and hardware differences. Always verify these numbers against the actual paper tables — the above are from memory and may have minor discrepancies.

---

## What to Skip for Now
- **Self-supervised pretraining (Paper Section 4):** The paper also proposes a masked autoencoder pretraining scheme. This is not part of the project scope.
- **Autoformer internals:** The Autoformer code was copied directly from Time-Series-Library. You do not need to understand its AutoCorrelation mechanism in depth — just know it is an encoder-decoder Transformer variant with O(T log T) complexity.
- **DLinear internals:** Simple decomposition + linear model. The code is self-explanatory if you read it.
