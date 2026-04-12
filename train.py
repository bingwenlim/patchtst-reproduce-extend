import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

from models.DLinear import Model as DLinear
from models.Autoformer import Model as Autoformer
from models.PatchTST import Model as PatchTST


def get_default_configs():
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=336,
        label_len=48,
        pred_len=96,
        enc_in=7,
        dec_in=7,
        c_out=7,
        moving_avg=25,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=1,
        dropout=0.1,
        embed="timeF",
        freq="h",
        activation="gelu",
        patch_len=16,
        stride=8,
    )


def create_dummy_dataloader(configs, num_samples=512, batch_size=32):
    x = torch.randn(num_samples, configs.seq_len, configs.enc_in)
    y = torch.randn(num_samples, configs.pred_len, configs.enc_in)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_model(model_name, configs):
    if model_name == "DLinear":
        return DLinear(configs)
    if model_name == "Autoformer":
        return Autoformer(configs)
    if model_name == "PatchTST":
        return PatchTST(configs)
    raise ValueError(f"Unknown model_name: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, configs):
    model.train()
    total_loss = 0.0

    for x_enc, y_true in loader:
        optimizer.zero_grad()

        batch_size = x_enc.size(0)

        # Autoformer expects decoder input shape [B, label_len + pred_len, dec_in]
        x_dec = torch.zeros(
            batch_size,
            configs.label_len + configs.pred_len,
            configs.dec_in
        )

        y_pred = model(
            x_enc,
            None,   # x_mark_enc
            x_dec,
            None    # x_mark_dec
        )

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def run_training(model_name="PatchTST", epochs=3):
    configs = get_default_configs()

    model = get_model(model_name, configs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loader = create_dummy_dataloader(configs)

    print(f"Training {model_name}...")

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, configs)
        print(f"Epoch {epoch + 1} | Loss: {loss:.4f}")


if __name__ == "__main__":
    run_training(model_name="Autoformer", epochs=3)