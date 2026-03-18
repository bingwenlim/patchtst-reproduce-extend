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
git clone git@github.com:BingWen-Fazz/patchtst-reproduce-extend.git
cd patchtst-reproduce-extend
```

### 3. Install dependencies

```bash
uv sync
```

This will automatically download Python 3.12 if you don't have it, create a virtual environment in `.venv/`, and install all required packages.

### 4. Run code

Prefix any Python command with `uv run` to use the project environment:

```bash
uv run python main.py
```

You do not need to manually activate the virtual environment.

## Project Structure

```
models/         # Model implementations (DLinear, Autoformer, PatchTST)
layers/         # Shared building blocks used by models
```
