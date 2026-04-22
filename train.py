import argparse
import copy
import json
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
from data_provider import get_dataset

ENCODER_ONLY_MODELS = {"PatchTST", "DLinear"}
ENCODER_DECODER_MODELS = {"Autoformer"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def adjust_learning_rate(optimizer, epoch, base_lr):
    """Type3 scheduler: constant for epochs 0-2, then 0.9x decay per epoch."""
    if epoch < 3:
        lr = base_lr
    else:
        lr = base_lr * (0.9 ** (epoch - 2))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def evaluate(model, loader, configs, device, model_name):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            x_enc = batch[0].to(device, non_blocking=True)
            y_true = batch[1].to(device, non_blocking=True)

            if model_name in ENCODER_ONLY_MODELS:
                y_pred = model(x_enc)
            else:
                x_mark_enc = batch[2].to(device, non_blocking=True)
                x_mark_dec = batch[3].to(device, non_blocking=True)
                x_dec = build_decoder_input(
                    x_enc.size(0), configs, x_enc.dtype, device,
                )
                y_pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            preds.append(y_pred.cpu())
            trues.append(y_true.cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    mse = torch.mean((preds - trues) ** 2).item()
    mae = torch.mean(torch.abs(preds - trues)).item()
    return mse, mae


def train_one_epoch(model, loader, optimizer, criterion, configs, device, model_name):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x_enc = batch[0].to(device, non_blocking=True)
        y_true = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad()

        if model_name in ENCODER_ONLY_MODELS:
            y_pred = model(x_enc)
        else:
            x_mark_enc = batch[2].to(device, non_blocking=True)
            x_mark_dec = batch[3].to(device, non_blocking=True)
            x_dec = build_decoder_input(
                x_enc.size(0), configs, x_enc.dtype, device,
            )
            y_pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def run_training(
    model_name="PatchTST",
    dataset_name="ETTh1",
    epochs=100,
    batch_size=128,
    patience=10,
    lr=1e-4,
    seq_len=336,
    label_len=48,
    pred_len=96,
    d_model=128,
    n_heads=16,
    e_layers=3,
    d_ff=256,
    dropout=0.2,
    patch_len=16,
    stride=8,
    save_dir="checkpoints",
    seed=42,
    num_workers=0,
    use_acca=False,
    alpha_mode="learned",
    acca_type="attention",
    acca_placement="pre_head",
    acca_n_heads=16,
    alpha_init=-4.6,
    freeze_backbone_epochs=0,
    run_name=None,
    trace_dir="scripts/traces",
):
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    pin_memory = device.type == "cuda"

    datasets, enc_in = get_dataset(dataset_name, seq_len, pred_len, label_len)
    configs = SimpleNamespace(
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
        d_layers=1,
        d_ff=d_ff,
        factor=1,
        dropout=dropout,
        embed="timeF",
        freq="h",
        activation="gelu",
        patch_len=patch_len,
        stride=stride,
        use_acca=use_acca,
        alpha_mode=alpha_mode,
        acca_type=acca_type,
        acca_placement=acca_placement,
        acca_n_heads=acca_n_heads,
        alpha_init=alpha_init,
        freeze_backbone_epochs=freeze_backbone_epochs,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = get_model(model_name, configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training {model_name} on {dataset_name}")
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

    # Per-epoch trace: captures alpha + losses; written to trace_dir/{run_name}_trace.json at the end.
    epoch_trace = []

    for epoch in range(epochs):
        # 2-stage training: freeze/unfreeze backbone
        if configs.freeze_backbone_epochs > 0:
            freeze = epoch < configs.freeze_backbone_epochs
            if hasattr(model, 'patch_embedding'):
                for param in model.patch_embedding.parameters():
                    param.requires_grad = not freeze
            if hasattr(model, 'encoder'):
                for param in model.encoder.parameters():
                    param.requires_grad = not freeze

        current_lr = adjust_learning_rate(optimizer, epoch, lr)
        epoch_start = time.time()

        train_mse = train_one_epoch(
            model, train_loader, optimizer, criterion, configs, device, model_name,
        )
        val_mse, val_mae = evaluate(model, val_loader, configs, device, model_name)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        # ACCA logging
        alpha_raw_mean = None
        alpha_effective_mean = None
        if configs.use_acca and hasattr(model, 'acca'):
            raw = model.acca._alpha_raw.detach().cpu().numpy().flatten()
            eff = torch.sigmoid(model.acca._alpha_raw).detach().cpu().numpy().flatten()
            alpha_raw_mean = float(raw.mean())
            alpha_effective_mean = float(eff.mean())
            print(f"Epoch {epoch + 1} | ACCA alpha_raw (mean: {alpha_raw_mean:.4f}): {raw}")
            print(f"Epoch {epoch + 1} | ACCA alpha_effective (mean: {alpha_effective_mean:.4f}): {eff}")

            if epoch == 0:
                print(f"Epoch 1 Quick Check | ACCA alpha_raw.grad is hooked up: {model.acca._alpha_raw.grad is not None}")

        epoch_trace.append({
            "epoch": epoch + 1,
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "val_mae": float(val_mae),
            "lr": float(current_lr),
            "epoch_time_s": float(epoch_time),
            "alpha_raw": alpha_raw_mean,
            "alpha_effective": alpha_effective_mean,
        })

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
    # Include run_name in the checkpoint filename so parallel ablation runs
    # don't clobber each other and we can re-load a specific trained ACCA
    # model later (e.g. to extract attention weights for the report figures).
    ckpt_slug = run_name if run_name else f"best_{model_name.lower()}"
    model_save_path = save_dir / f"{ckpt_slug}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to: {model_save_path}")

    test_start = time.time()
    test_mse, test_mae = evaluate(model, test_loader, configs, device, model_name)
    test_inference_time = time.time() - test_start

    final_alpha_raw = None
    final_alpha_effective = None
    if configs.use_acca and hasattr(model, 'acca'):
        final_alpha_raw = float(
            model.acca._alpha_raw.detach().cpu().numpy().flatten().mean()
        )
        final_alpha_effective = float(
            torch.sigmoid(model.acca._alpha_raw).detach().cpu().numpy().flatten().mean()
        )

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "best_val_mae": best_val_mae,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "total_training_time": total_train_time,
        "test_inference_time": test_inference_time,
        "final_alpha_raw": final_alpha_raw,
        "final_alpha_effective": final_alpha_effective,
    }

    print()
    print("Final Summary")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if run_name:
        trace_path = Path(trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path / f"{run_name}_trace.json"
        with trace_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "config": {
                        "model": model_name,
                        "dataset": dataset_name,
                        "seq_len": seq_len,
                        "pred_len": pred_len,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "e_layers": e_layers,
                        "d_ff": d_ff,
                        "dropout": dropout,
                        "use_acca": use_acca,
                        "acca_type": acca_type,
                        "acca_placement": acca_placement,
                        "alpha_mode": alpha_mode,
                        "alpha_init": alpha_init,
                        "seed": seed,
                    },
                    "summary": results,
                    "per_epoch": epoch_trace,
                },
                f,
                indent=2,
            )
        print(f"Trace saved to: {trace_file}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train forecasting baselines on a CSV dataset.")
    parser.add_argument("--model", type=str, default="PatchTST",
                        choices=["PatchTST", "DLinear", "Autoformer"])
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # ACCA arguments
    parser.add_argument("--use_acca", action="store_true")
    parser.add_argument("--alpha_mode", type=str, default="learned")
    parser.add_argument("--acca_type", type=str, default="attention", choices=["attention", "linear"])
    parser.add_argument("--acca_placement", type=str, default="pre_head", choices=["pre_head", "post_head"])
    parser.add_argument("--acca_n_heads", type=int, default=16)
    parser.add_argument("--alpha_init", type=float, default=-4.6)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)

    # Run bookkeeping: if set, a per-epoch JSON trace is written to {trace_dir}/{run_name}_trace.json.
    parser.add_argument("--run_name", type=str, default=None,
                        help="Identifier for this run; used to name the trace file.")
    parser.add_argument("--trace_dir", type=str, default="scripts/traces")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        patch_len=args.patch_len,
        stride=args.stride,
        save_dir=args.save_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        use_acca=args.use_acca,
        alpha_mode=args.alpha_mode,
        acca_type=args.acca_type,
        acca_placement=args.acca_placement,
        acca_n_heads=args.acca_n_heads,
        alpha_init=args.alpha_init,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        run_name=args.run_name,
        trace_dir=args.trace_dir,
    )
