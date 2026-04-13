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

Place datasets inside the data/ folder.

Example:

```bash
data/
  ett.csv
  weather.csv
  electricity.csv
```

### 5. Run Training 
Prefix any Python command with `uv run` to use the project environment.

You do not need to manually activate the virtual environment.


#### Train models

```bash
# PatchTST
uv run python train.py --model PatchTST --data data/ett.csv

# DLinear
uv run python train.py --model DLinear --data data/ett.csv

# Autoformer
uv run python train.py --model Autoformer --data data/ett.csv
```


#### Optional arguments
```bash
uv run python train.py \
  --model PatchTST \
  --data data/ett.csv \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.0001 \
  --seq_len 336 \
  --pred_len 96
```
#### Output
Model checkpoints will be saved in:

```bash
checkpoints/
```

## Project Structure

```
models/         # Model implementations (DLinear, Autoformer, PatchTST)
layers/         # Shared building blocks used by models
data_provider.py # Dataset loading and preprocessing
train.py        # Main training script
data/           # Dataset files (not included in repo)
checkpoints/    # Saved model weights
```
