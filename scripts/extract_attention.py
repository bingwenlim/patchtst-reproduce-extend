"""Re-load a trained PatchTST+ACCA(attention) checkpoint, run the test set,
and dump the average cross-channel attention weight matrix to disk.

Usage:
    uv run python scripts/extract_attention.py \
        --checkpoint checkpoints/acca_attn_pre_learned_fx.pt \
        --dataset fx --placement pre_head --out scripts/traces/fx_attn.npy

The output .npy is a [C, C] matrix. Plotting is handled separately by
scripts/plot_acca.py so this script can be run on any machine that has the
trained weights without requiring matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.PatchTST import Model as PatchTST
from data_provider import get_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--placement", type=str, default="pre_head",
                   choices=["pre_head", "post_head"])
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--e_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=5,
                   help="Number of test batches to average the attention over.")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    datasets, enc_in = get_dataset(
        args.dataset, args.seq_len, args.pred_len, args.label_len,
    )

    configs = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        enc_in=enc_in, dec_in=enc_in, c_out=enc_in, moving_avg=25,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers,
        d_layers=1, d_ff=args.d_ff, factor=1, dropout=args.dropout,
        embed="timeF", freq="h", activation="gelu",
        patch_len=args.patch_len, stride=args.stride,
        use_acca=True, alpha_mode="learned",
        acca_type="attention", acca_placement=args.placement,
        acca_n_heads=args.n_heads, alpha_init=-4.6,
        freeze_backbone_epochs=0,
    )

    model = PatchTST(configs).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.acca.record_attn = True

    test_loader = DataLoader(
        datasets["test"], batch_size=args.batch_size, shuffle=False,
    )

    accumulated = None
    n_seen = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.num_batches:
                break
            x_enc = batch[0].to(device, non_blocking=True)
            _ = model(x_enc)
            w = model.acca.last_attn_weights
            if w is None:
                raise RuntimeError(
                    "ACCA did not populate last_attn_weights; confirm "
                    "record_attn=True and that acca_type is 'attention'."
                )
            accumulated = w.clone() if accumulated is None else accumulated + w
            n_seen += 1

    avg = (accumulated / max(1, n_seen)).numpy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, avg)
    print(f"Saved [{avg.shape[0]} x {avg.shape[1]}] attention matrix to {out_path}")


if __name__ == "__main__":
    main()
