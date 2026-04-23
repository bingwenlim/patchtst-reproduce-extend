# patchtst-reproduce-extend
Replication and extension of PatchTST by Group 23 for DSA5106.

## Extending PatchTST with ACCA

This repository extends the original Channel Independent PatchTST architecture with an **Adaptive Cross-Channel Attention (ACCA)** module. ACCA is inserted after the Transformer encoder and before the prediction head. It mixes the per-channel patch representations with a linear map across the channel axis, then blends the mixed signal back into the original representation via a learned sigmoid gate `alpha`. The gate is initialized to `sigmoid(-4.6) ~= 0.01`, so the model starts as a faithful copy of PatchTST and can open the cross-channel pathway through gradient descent if it is useful.

### ACCA Configuration

*   `--use_acca`: Enable the ACCA module.
*   `--alpha_init`: Initial value of the raw alpha gate. Default: `-4.6` (`sigmoid(-4.6) ~= 0.01`).

```bash
uv run python train.py \
  --model PatchTST \
  --dataset ETTh1 \
  --d_model 16 \
  --n_heads 4 \
  --d_ff 128 \
  --dropout 0.3 \
  --use_acca
```

This configuration achieves Test MSE: `0.3813` and Test MAE: `0.4031`.

## Setup

The project leverages [uv](https://docs.astral.sh/uv/) for dependency management. It enforces dependency resolution and environment reproducibility automatically.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Subsequent to installation, restart the terminal (or execute `source ~/.zshrc` / `source ~/.bashrc`) to initialize the `uv` binary.

### 2. Clone the repo

```bash
git clone git@github.com:bingwenlim/patchtst-reproduce-extend.git
cd patchtst-reproduce-extend
```

### 3. Install dependencies

```bash
uv sync
```

This automates the provisioning of a Python 3.12 virtual environment (`.venv/`) and the installation of all required dependencies.

### 4. Dataset

Target datasets must reside within the `data/` directory. The following benchmark datasets are supported and registered:
- `ETTh1` (`data/ett.csv`)
- `traffic` (`data/processed_traffic.csv`)
- `air` (`data/processed_air.csv`)
- `fx` (`data/fx_cleaned.csv`)

## Training

Execution requires prepending commands with `uv run` to invoke the isolated environment.

```bash
# PatchTST (paper defaults)
uv run python train.py --model PatchTST

# PatchTST (ETTh1-specific config from paper)
uv run python train.py --model PatchTST --d_model 16 --n_heads 4 --d_ff 128 --dropout 0.3

# DLinear
uv run python train.py --model DLinear

# Autoformer
uv run python train.py --model Autoformer
```

### CLI arguments

| Argument       | Default     | Description                                       |
| -------------- | ----------- | ------------------------------------------------- |
| `--model`      | PatchTST    | Model name: PatchTST, DLinear, Autoformer         |
| `--dataset`    | ETTh1       | Dataset name: ETTh1, traffic, air (or registered) |
| `--epochs`     | 100         | Max training epochs                               |
| `--batch_size` | 128         | Batch size                                        |
| `--patience`   | 10          | Early stopping patience                           |
| `--lr`         | 1e-4        | Learning rate                                     |
| `--seq_len`    | 336         | Input sequence length                             |
| `--label_len`  | 48          | Decoder label length (Autoformer)                 |
| `--pred_len`   | 96          | Prediction horizon                                |
| `--d_model`    | 128         | Transformer latent dimension                      |
| `--n_heads`    | 16          | Number of attention heads                         |
| `--e_layers`   | 3           | Number of encoder layers                          |
| `--d_ff`       | 256         | Feed-forward hidden dimension                     |
| `--dropout`    | 0.2         | Dropout rate                                      |
| `--patch_len`  | 16          | Patch length (PatchTST)                           |
| `--stride`     | 8           | Patch stride (PatchTST)                           |
| `--seed`       | 42          | Random seed                                       |
| `--save_dir`   | checkpoints | Checkpoint save directory                         |

### Output

Model checkpoints saved to `checkpoints/`. Training logs printed to stdout.

## Adding a New Dataset

Integrating a novel dataset requires formatting as a CSV and defining a corresponding loader function within `data_provider.py`.

### 1. CSV format

The CSV format necessitates:
- A `date` column (parseable by `pd.to_datetime`) for temporal features
- One or more numeric columns (these become the forecast channels)

```
date,feature1,feature2,feature3
2020-01-01 00:00:00,1.2,3.4,5.6
2020-01-01 01:00:00,1.3,3.5,5.7
...
```

The file should be located in `data/`, e.g., `data/weather.csv`.

### 2. Write a loader function

A data loader function must be implemented in `data_provider.py`. It is required to:
- Load the CSV
- Define train/val/test split borders
- Standardize using train-only statistics
- Extract time features from the date column
- Return `(datasets_dict, enc_in)` where `datasets_dict` has keys `"train"`, `"val"`, `"test"`

The `_get_ett_datasets()` function may serve as a structural template. Essential modifications include the file path and training/validation/test indices.

```python
def _get_weather_datasets(file_path, seq_len, pred_len, label_len):
    df = pd.read_csv(file_path)
    raw_data = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    marks = time_features(df["date"])

    # Define dataset split indices
    train_end = ...
    val_end = ...
    test_end = ...

    border1s = [0, train_end - seq_len, val_end - seq_len]
    border2s = [train_end, val_end, test_end]

    # Standardize on train only
    train_portion = raw_data[:train_end]
    mean = train_portion.mean(axis=0, keepdims=True)
    std = train_portion.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scaled = (raw_data - mean) / std

    enc_in = raw_data.shape[1]
    datasets = {}
    for i, name in enumerate(["train", "val", "test"]):
        b1, b2 = border1s[i], border2s[i]
        datasets[name] = TimeSeriesDataset(
            scaled[b1:b2], seq_len, pred_len, label_len, marks[b1:b2],
        )
    return datasets, enc_in
```

### 3. Register in get_dataset()

Register the new dataset loader within the `get_dataset()` dispatch router:

```python
def get_dataset(name, seq_len, pred_len, label_len):
    if name == "ETTh1":
        return _get_ett_datasets(...)
    if name == "Weather":
        return _get_weather_datasets(
            file_path="data/weather.csv",
            seq_len=seq_len, pred_len=pred_len, label_len=label_len,
        )
    raise ValueError(f"Unknown dataset: {name}")
```

### 4. Run

```bash
uv run python train.py --dataset Weather --model PatchTST
```

## Project Structure

```
models/          # Model implementations (PatchTST, DLinear, Autoformer)
layers/          # Shared building blocks (Autoformer layers, embeddings)
data_provider.py # Dataset loading, splitting, and preprocessing
train.py         # Training script with CLI
data/            # Dataset CSV files
checkpoints/     # Saved model weights
RESULTS.md       # Reproduction benchmark results
```
