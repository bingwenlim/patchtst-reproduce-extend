import argparse
import copy
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.DLinear import Model as DLinear
from models.Autoformer import Model as Autoformer
from models.PatchTST import Model as PatchTST
from data_provider import TimeSeriesDataset, load_csv_dataset, standardize_train_val_test


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_configs(
    enc_in: int,
    seq_len: int = 336,
    label_len: int = 48,
    pred_len: int = 96,
    d_model: int = 128,
    n_heads: int = 8,
    e_layers: int = 2,
    d_layers: int = 1,
    d_ff: int = 256,
    factor: int = 1,
    dropout: float = 0.1,
    patch_len: int = 16,
    stride: int = 8,
):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        moving_avg=25,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        factor=factor,
        dropout=dropout,
        embed="timeF",
        freq="h",
        activation="gelu",
        patch_len=patch_len,
        stride=stride,
    )


def get_model(model_name: str, configs):
    if model_name == "DLinear":
        return DLinear(configs)
    if model_name == "Autoformer":
        return Autoformer(configs)
    if model_name == "PatchTST":
        return PatchTST(configs)
    raise ValueError(f"Unknown model_name: {model_name}")


def build_decoder_input(batch_size: int, configs, dtype, device):
    return torch.zeros(
        batch_size,
        configs.label_len + configs.pred_len,
        configs.dec_in,
        dtype=dtype,
        device=device,
    )


def evaluate(model, loader, criterion, configs, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for x_enc, y_true in loader:
            x_enc = x_enc.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            x_dec = build_decoder_input(
                batch_size=x_enc.size(0),
                configs=configs,
                dtype=x_enc.dtype,
                device=device,
            )

            y_pred = model(x_enc, None, x_dec, None)

            mse = criterion(y_pred, y_true)
            mae = torch.mean(torch.abs(y_pred - y_true))

            total_mse += mse.item()
            total_mae += mae.item()

    return total_mse / len(loader), total_mae / len(loader)


def train_one_epoch(model, loader, optimizer, criterion, configs, device):
    model.train()
    total_mse = 0.0

    for x_enc, y_true in loader:
        x_enc = x_enc.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        optimizer.zero_grad()

        x_dec = build_decoder_input(
            batch_size=x_enc.size(0),
            configs=configs,
            dtype=x_enc.dtype,
            device=device,
        )

        y_pred = model(x_enc, None, x_dec, None)
        mse = criterion(y_pred, y_true)

        mse.backward()
        optimizer.step()

        total_mse += mse.item()

    return total_mse / len(loader)


def run_training(
    model_name="PatchTST",
    file_path="data/ett.csv",
    epochs=20,
    batch_size=32,
    patience=3,
    lr=1e-4,
    seq_len=336,
    label_len=48,
    pred_len=96,
    save_dir="checkpoints",
    seed=42,
    num_workers=0,
):
    set_seed(seed)

    raw_data = load_csv_dataset(file_path)

    n = len(raw_data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_data = raw_data[:train_end]
    val_data = raw_data[train_end:val_end]
    test_data = raw_data[val_end:]

    train_data, val_data, test_data, _, _ = standardize_train_val_test(
        train_data, val_data, test_data
    )

    enc_in = train_data.shape[1]
    configs = get_default_configs(
        enc_in=enc_in,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
    )

    train_dataset = TimeSeriesDataset(train_data, configs.seq_len, configs.pred_len)
    val_dataset = TimeSeriesDataset(val_data, configs.seq_len, configs.pred_len)
    test_dataset = TimeSeriesDataset(test_data, configs.seq_len, configs.pred_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = get_model(model_name, configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training {model_name} on {file_path}")
    print(f"Using device: {device}")
    print(
        f"seq_len={seq_len}, label_len={label_len}, pred_len={pred_len}, "
        f"batch_size={batch_size}, lr={lr}, patience={patience}, seed={seed}"
    )

    best_val_mse = float("inf")
    best_val_mae = None
    best_epoch = 0
    best_state_dict = None
    no_improve_count = 0

    total_train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_mse = train_one_epoch(model, train_loader, optimizer, criterion, configs, device)
        val_mse, val_mae = evaluate(model, val_loader, criterion, configs, device)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    total_train_time = time.time() - total_train_start

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / f"best_{model_name.lower()}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to: {model_save_path}")

    test_start = time.time()
    test_mse, test_mae = evaluate(model, test_loader, criterion, configs, device)
    test_inference_time = time.time() - test_start

    results = {
        "model": model_name,
        "data": file_path,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "best_val_mae": best_val_mae,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "total_training_time": total_train_time,
        "test_inference_time": test_inference_time,
    }

    print()
    print("Final Summary")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train forecasting baselines on a CSV dataset.")
    parser.add_argument("--model", type=str, default="PatchTST",
                        choices=["PatchTST", "DLinear", "Autoformer"])
    parser.add_argument("--data", type=str, default="data/ett.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        model_name=args.model,
        file_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        save_dir=args.save_dir,
        seed=args.seed,
        num_workers=args.num_workers,
    )