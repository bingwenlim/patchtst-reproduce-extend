import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, attn_dropout=0.0):
        super().__init__()
        self.scale = d_k ** -0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        # q, k, v: [B, H, N, d_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.W_Q = nn.Linear(d_model, self.d_k * n_heads, bias=True)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads, bias=True)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads, bias=True)
        self.W_O = nn.Linear(self.d_v * n_heads, d_model, bias=False)
        self.attn = ScaledDotProductAttention(self.d_k, attn_dropout)

    def forward(self, Q, K, V):
        B, N, _ = Q.shape
        # Project and reshape to [B, H, N, d_k]
        q = self.W_Q(Q).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(B, N, self.n_heads, self.d_v).transpose(1, 2)

        output = self.attn(q, k, v)

        # [B, H, N, d_v] -> [B, N, H*d_v]
        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.W_O(output)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.2):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )
        # BatchNorm via transpose trick (BN expects [B, C, L])
        self.norm_attn = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
        )
        self.norm_ffn = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention with residual + post-norm
        attn_out = self.self_attn(src, src, src)
        src = self.norm_attn(src + self.dropout_attn(attn_out))
        # FFN with residual + post-norm
        ff_out = self.ff(src)
        src = self.norm_ffn(src + self.dropout_ffn(ff_out))
        return src


class TSTEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, padding, num_patches, d_model, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.W_pos = nn.Parameter(torch.empty(num_patches, d_model))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
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

        # Unfold into patches: [B*C, num_patches, patch_len]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Linear projection + learnable positional encoding
        x = self.patch_proj(x)
        x = self.dropout(x + self.W_pos)

        return x


class AdaptiveCrossChannelAttention(nn.Module):
    def __init__(self, c_in, d_model_or_len, n_heads=4, dropout=0.1, alpha_mode='learned', acca_type='attention', acca_placement='pre_head', alpha_init=-4.6):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model_or_len
        self.acca_type = acca_type
        self.acca_placement = acca_placement
        self.alpha_mode = alpha_mode
        self._first_forward = True

        if alpha_mode == 'learned':
            self._alpha_raw = nn.Parameter(torch.tensor([float(alpha_init)], dtype=torch.float32).view(1, 1, 1))
        elif alpha_mode == 'fixed_zero':
            self.register_buffer('_alpha_raw', torch.tensor([-float('inf')], dtype=torch.float32).view(1, 1, 1))
        elif alpha_mode == 'fixed_one':
            self.register_buffer('_alpha_raw', torch.tensor([float('inf')], dtype=torch.float32).view(1, 1, 1))
        else:
            raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

        self.channel_embeddings = nn.Parameter(torch.randn(1, c_in, 1))

        if acca_type == 'attention':
            self.n_heads = min(n_heads, max(1, c_in // 2))

            # Ensure d_model is divisible by n_heads
            self.internal_dim = self.d_model
            if self.d_model % self.n_heads != 0:
                self.internal_dim = self.n_heads * (self.d_model // self.n_heads + 1)
                self.in_proj = nn.Linear(self.d_model, self.internal_dim)
                self.out_proj = nn.Linear(self.internal_dim, self.d_model)
            else:
                self.in_proj = nn.Identity()
                self.out_proj = nn.Identity()

            self.mha = nn.MultiheadAttention(embed_dim=self.internal_dim, num_heads=self.n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(self.internal_dim)
            self.dropout = nn.Dropout(dropout)
        elif acca_type == 'linear':
            self.linear = nn.Linear(c_in, c_in)
        else:
            raise ValueError(f"Unknown acca_type: {acca_type}")

        # Optional cache for the last attention weights (averaged over heads),
        # populated during eval when enabled via `record_attn`. Kept on CPU.
        self.record_attn = False
        self.last_attn_weights = None

    @property
    def alpha_raw(self):
        return self._alpha_raw.flatten()[0].item()

    @property
    def alpha_effective(self):
        return torch.sigmoid(self._alpha_raw).flatten()[0].item()

    def forward(self, x, b, c):
        alpha = torch.sigmoid(self._alpha_raw)
        
        if self.acca_placement == 'pre_head':
            # x is [B*C, P, D] where P is num_patches
            bc, p, d = x.shape
            
            x_reshaped = x.reshape(b, c, p, d)
            # attention across channels: [B*P, C, D]
            x_transposed = x_reshaped.permute(0, 2, 1, 3).reshape(b * p, c, d)
            
            if self.acca_type == 'attention':
                x_transposed_with_emb = x_transposed + self.channel_embeddings
                x_proj = self.in_proj(x_transposed_with_emb)
                need_w = self.record_attn and not self.training
                attn_out, attn_w = self.mha(
                    x_proj, x_proj, x_proj,
                    need_weights=need_w, average_attn_weights=True,
                )
                if need_w and attn_w is not None:
                    # attn_w: [B*P, C, C] -> average over the B*P axis for a stable heatmap.
                    self.last_attn_weights = attn_w.detach().mean(dim=0).cpu()
                h = self.out_proj(self.norm(x_proj + self.dropout(attn_out)))
            else: # linear
                h = self.linear(x_transposed.transpose(1, 2)).transpose(1, 2)

            out_transposed = x_transposed + alpha * (h - x_transposed)
            # Back to [B*C, P, D]
            out = out_transposed.reshape(b, p, c, d).permute(0, 2, 1, 3).reshape(bc, p, d)
            return out

        elif self.acca_placement == 'post_head':
            # x is [B*C, pred_len]
            bc, l = x.shape

            x_t = x.reshape(b, c, l)

            if self.acca_type == 'attention':
                x_t_with_emb = x_t + self.channel_embeddings
                x_proj = self.in_proj(x_t_with_emb)
                need_w = self.record_attn and not self.training
                attn_out, attn_w = self.mha(
                    x_proj, x_proj, x_proj,
                    need_weights=need_w, average_attn_weights=True,
                )
                if need_w and attn_w is not None:
                    # attn_w: [B, C, C] -> average over the batch axis.
                    self.last_attn_weights = attn_w.detach().mean(dim=0).cpu()
                h = self.out_proj(self.norm(x_proj + self.dropout(attn_out)))
            else:
                h = self.linear(x_t.transpose(1, 2)).transpose(1, 2)

            out = x_t + alpha * (h - x_t)
            return out.reshape(bc, l)


class PredictionHead(nn.Module):
    def __init__(self, num_patches, d_model, pred_len, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(num_patches * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [B*C, num_patches, d_model]
        returns: [B*C, pred_len]
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout

        self.num_patches = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1
        self.padding = self.patch_len + (self.num_patches - 1) * self.stride - self.seq_len

        self.patch_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.padding,
            num_patches=self.num_patches,
            d_model=self.d_model,
            dropout=self.dropout,
        )

        self.encoder = TSTEncoder([
            TSTEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.e_layers)
        ])

        self.head = PredictionHead(
            num_patches=self.num_patches,
            d_model=self.d_model,
            pred_len=self.pred_len,
            head_dropout=0.0,
        )

        self.use_acca = getattr(configs, 'use_acca', False)
        self.acca_placement = getattr(configs, 'acca_placement', 'pre_head')
        if self.use_acca:
            with torch.random.fork_rng():
                d_acca = self.d_model if self.acca_placement == 'pre_head' else self.pred_len
                self.acca = AdaptiveCrossChannelAttention(
                    c_in=self.enc_in,
                    d_model_or_len=d_acca,
                    n_heads=getattr(configs, 'acca_n_heads', self.n_heads),
                    dropout=self.dropout,
                    alpha_mode=getattr(configs, 'alpha_mode', 'learned'),
                    alpha_init=getattr(configs, 'alpha_init', -4.6),
                    acca_type=getattr(configs, 'acca_type', 'attention'),
                    acca_placement=self.acca_placement
                )

    def forecast(self, x_enc):
        """
        x_enc: [B, seq_len, C]
        returns: [B, pred_len, C]
        """
        # Instance normalization
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - mean) / stdev

        B, L, C = x_enc.shape

        # Channel independence: [B, seq_len, C] -> [B*C, seq_len]
        x = x_enc.permute(0, 2, 1).reshape(B * C, L)

        # Patch embedding
        x = self.patch_embedding(x)  # [B*C, num_patches, d_model]

        # Transformer encoder
        x = self.encoder(x)  # [B*C, num_patches, d_model]

        if self.use_acca and self.acca_placement == 'pre_head':
            x = self.acca(x, B, C)

        # Prediction head
        x = self.head(x)  # [B*C, pred_len]

        if self.use_acca and self.acca_placement == 'post_head':
            x = self.acca(x, B, C)

        # [B*C, pred_len] -> [B, pred_len, C]
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # Denormalize
        x = x * stdev + mean

        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        return None
