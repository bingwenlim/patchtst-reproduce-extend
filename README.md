# patchtst-reproduce-extend
Replication and extension of PatchTST for time series forecasting. For DSA5106 Group 23.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. It handles the Python version and all packages automatically.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal (or run `source ~/.zshrc` / `source ~/.bashrc`) so the `uv` command is available.

### 2. Clone the repo

```bash
git clone git@github.com:bingwenlim/patchtst-reproduce-extend.git
cd patchtst-reproduce-extend
```

### 3. Install dependencies

```bash
uv sync
```

This will automatically download Python 3.12 if you don't have it, create a virtual environment in `.venv/`, and install all required packages.

### 4. Dataset

Place datasets inside the `data/` folder. ETTh1 is included as `data/ett.csv`.

## Training

Prefix any Python command with `uv run` to use the project environment.

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

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | PatchTST | Model name: PatchTST, DLinear, Autoformer |
| `--dataset` | ETTh1 | Dataset name (must be registered in data_provider.py) |
| `--epochs` | 100 | Max training epochs |
| `--batch_size` | 128 | Batch size |
| `--patience` | 10 | Early stopping patience |
| `--lr` | 1e-4 | Learning rate |
| `--seq_len` | 336 | Input sequence length |
| `--label_len` | 48 | Decoder label length (Autoformer) |
| `--pred_len` | 96 | Prediction horizon |
| `--d_model` | 128 | Transformer latent dimension |
| `--n_heads` | 16 | Number of attention heads |
| `--e_layers` | 3 | Number of encoder layers |
| `--d_ff` | 256 | Feed-forward hidden dimension |
| `--dropout` | 0.2 | Dropout rate |
| `--patch_len` | 16 | Patch length (PatchTST) |
| `--stride` | 8 | Patch stride (PatchTST) |
| `--seed` | 42 | Random seed |
| `--save_dir` | checkpoints | Checkpoint save directory |

### Output

Model checkpoints saved to `checkpoints/`. Training logs printed to stdout.

## Adding a New Dataset

To add a new dataset, you need a CSV file and a loader function in `data_provider.py`.

### 1. CSV format

Your CSV must have:
- A `date` column (parseable by `pd.to_datetime`) for temporal features
- One or more numeric columns (these become the forecast channels)

```
date,feature1,feature2,feature3
2020-01-01 00:00:00,1.2,3.4,5.6
2020-01-01 01:00:00,1.3,3.5,5.7
...
```

Place it in `data/`, e.g. `data/weather.csv`.

### 2. Write a loader function

Add a `_get_<name>_datasets()` function in `data_provider.py`. It must:
- Load the CSV
- Define train/val/test split borders
- Standardize using train-only statistics
- Extract time features from the date column
- Return `(datasets_dict, enc_in)` where `datasets_dict` has keys `"train"`, `"val"`, `"test"`

Use `_get_ett_datasets()` as a template. The key things to change are the file path and split borders.

```python
def _get_weather_datasets(file_path, seq_len, pred_len, label_len):
    df = pd.read_csv(file_path)
    raw_data = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    marks = time_features(df["date"])

    # Define your split borders
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

Add a branch to the `get_dataset()` router:

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
