import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, N, d_model]
        return self.pe[:, :x.size(1), :]


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = patch_len - stride

        self.patch_proj = nn.Linear(patch_len, d_model, bias=False)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B*C, seq_len]
        returns: [B*C, num_patches, d_model]
        """
        # Replication padding at the end
        if self.padding > 0:
            pad = x[:, -1:].repeat(1, self.padding)
            x = torch.cat([x, pad], dim=1)

        # Unfold into patches
        # x_unfold: [B*C, num_patches, patch_len]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Linear projection
        x = self.patch_proj(x)  # [B*C, num_patches, d_model]

        # Add positional encoding
        x = x + self.pos_encoding(x)

        return self.dropout(x)


class PredictionHead(nn.Module):
    def __init__(self, num_patches, d_model, pred_len, dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_patches * d_model, pred_len)

    def forward(self, x):
        """
        x: [B*C, num_patches, d_model]
        returns: [B*C, pred_len]
        """
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.patch_len = getattr(configs, "patch_len", 16)
        self.stride = getattr(configs, "stride", 8)
        self.d_model = getattr(configs, "d_model", 128)
        self.n_heads = getattr(configs, "n_heads", 16)
        self.e_layers = getattr(configs, "e_layers", 3)
        self.d_ff = getattr(configs, "d_ff", 256)
        self.dropout = getattr(configs, "dropout", 0.2)

        self.padding = self.patch_len - self.stride
        self.num_patches = ((self.seq_len - self.patch_len) // self.stride) + 2

        self.patch_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            dropout=self.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.e_layers,
        )

        self.head = PredictionHead(
            num_patches=self.num_patches,
            d_model=self.d_model,
            pred_len=self.pred_len,
            dropout=self.dropout,
        )

    def forecast(self, x_enc):
        """
        x_enc: [B, seq_len, C]
        returns: [B, pred_len, C]
        """
        B, L, C = x_enc.shape

        # Channel independence:
        # [B, seq_len, C] -> [B, C, seq_len]
        x = x_enc.permute(0, 2, 1)

        # [B, C, seq_len] -> [B*C, seq_len]
        x = x.reshape(B * C, L)

        # Patch embedding
        x = self.patch_embedding(x)  # [B*C, num_patches, d_model]

        # Transformer encoder
        x = self.encoder(x)  # [B*C, num_patches, d_model]

        # Prediction head
        x = self.head(x)  # [B*C, pred_len]

        # [B*C, pred_len] -> [B, C, pred_len]
        x = x.reshape(B, C, self.pred_len)

        # [B, C, pred_len] -> [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        return None