"""Per-asset temporal transformer encoder.

The original Cross-Asset Transformer treats each asset's 60-day return history
as a flat feature vector (just concatenated values). This loses any sequential
structure. A regime in March 2020 *looks like* a regime in October 2008 — but
the original architecture can't reference that because there's no notion of
time within the input.

TemporalEncoder fixes this. For each asset:
  - Input:  (T_hist, F_per_day)   per-day features over T_hist days
  - Output: (d_temp,)              fixed-dim asset representation

Architecture: a small Transformer with learned positional embeddings and a
[CLS] token whose final embedding is used as the asset representation.

Used INSIDE CrossAssetTransformer when `cfg.use_temporal_encoder=True`. The
output of the temporal encoder REPLACES the W concatenated daily returns in
the asset features. Static features (rolling vol, prev_w, OHLCV features)
are concatenated alongside.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class _TemporalBlock(nn.Module):
    """Pre-LN transformer block at the day-attention level."""

    def __init__(self, d_temp: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_temp)
        self.attn = nn.MultiheadAttention(d_temp, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_temp)
        self.mlp = nn.Sequential(
            nn.Linear(d_temp, d_temp * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_temp * ff_mult, d_temp),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.ln1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TemporalEncoder(nn.Module):
    """Process per-asset daily history into a fixed-dim representation.

    Args:
        per_day_dim : number of features per day (e.g. 1 for return-only,
                      5 for return + 4 OHLCV)
        d_temp      : hidden size of the temporal transformer
        n_heads     : number of attention heads
        n_layers    : depth
        max_seq_len : max history length (positional embeddings)
        dropout     : dropout for attention/MLP

    Forward signature:
        x : (B, N, T, F_per_day)   per-asset daily history
        returns : (B, N, d_temp)   fixed-dim asset embedding
    """

    def __init__(
        self,
        per_day_dim: int,
        d_temp: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 120,
        dropout: float = 0.10,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.d_temp = d_temp
        self.per_day_proj = nn.Linear(per_day_dim, d_temp)
        # Learned positional embedding over time positions (max_seq_len)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, d_temp) * 0.02)
        # Learned [CLS] token whose final embedding pools the sequence
        self.cls_token = nn.Parameter(torch.randn(d_temp) * 0.02)
        self.blocks = nn.ModuleList([
            _TemporalBlock(d_temp, n_heads, ff_mult, dropout) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_temp)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, T, F_per_day)
        B, N, T, F = x.shape
        x = x.reshape(B * N, T, F)
        x = self.per_day_proj(x) + self.pos_emb[:T].unsqueeze(0)         # (B*N, T, d_temp)
        cls = self.cls_token.expand(B * N, 1, -1)
        x = torch.cat([cls, x], dim=1)                                    # (B*N, T+1, d_temp)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        out = x[:, 0, :]                                                  # (B*N, d_temp)  CLS output
        return out.reshape(B, N, self.d_temp)
