"""Cross-Asset Transformer.

Sequence layout (per snapshot, per batch element):
    position 0       -> [CTX]  global macro / HMM regime
    positions 1..N   -> asset tokens (one per equity)

Two distinct linear projections embed CTX-raw and Asset-raw into a common
d_model space. Standard pre-LN multi-head self-attention + MLP blocks operate
over the (N+1) sequence with `key_padding_mask` honoring the dynamic universe.

Output heads (consumed by `paragon.model.cholesky` and friends):
    - cholesky_factor : (B, N, N) lower-triangular L  (diag positive)
    - mu              : (B, N) expected forward returns
The covariance is reconstructed as Sigma = L L^T outside this module so we keep
the model's tensor outputs raw and let the Cholesky helper enforce shape +
positivity semantics in one place.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .factor_head import FactorCholeskyHead


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class TransformerConfig:
    n_assets: int
    asset_in_dim: int
    ctx_in_dim: int
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    chol_min_diag: float = 1e-4    # enforced floor on diag of L (post softplus)
    head_type: str = "standard"    # "standard" (full lower-tri L) or "factor"
                                   # (Sigma = B F B^T + D, rank-k + diag).
                                   # Factor head has 3-4x fewer free params and
                                   # is well-suited to data-scarce regimes
                                   # where N is large relative to training set.
    n_factors: int = 4             # only used when head_type='factor'
    use_asset_pos: bool = True     # if False, the per-asset learned positional
                                   # embedding is dropped. This makes the model
                                   # truly permutation-invariant over assets,
                                   # which is required for universe-transfer
                                   # pretraining (train on 500 stocks, fine-tune
                                   # on 16). Asset identity then comes purely
                                   # from feature trajectories — the right
                                   # inductive bias for a cross-asset model
                                   # that should be a function of behavior, not
                                   # ticker name. v8+ default.


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class TokenEmbedding(nn.Module):
    """Two-headed projection: separate linear layers for CTX and asset rows.

    When `use_asset_pos=True`, asset tokens additionally receive a learned
    per-position embedding (breaks symmetry between AAPL and MSFT). When
    False, the model is permutation-invariant over assets — required for
    universe-transfer pretraining.
    """

    def __init__(
        self, n_assets: int, asset_in_dim: int, ctx_in_dim: int, d_model: int,
        use_asset_pos: bool = True,
    ):
        super().__init__()
        self.asset_proj = nn.Linear(asset_in_dim, d_model)
        self.ctx_proj = nn.Linear(ctx_in_dim, d_model)
        self.ctx_pos = nn.Parameter(torch.randn(d_model) * 0.02)
        if use_asset_pos:
            self.asset_pos = nn.Parameter(torch.randn(n_assets, d_model) * 0.02)
        else:
            self.register_parameter("asset_pos", None)

    def forward(self, asset_feats: Tensor, ctx_feats: Tensor) -> Tensor:
        # asset_feats: (B, N, F_a)   ctx_feats: (B, F_c)
        a = self.asset_proj(asset_feats)                                  # (B, N, d)
        if self.asset_pos is not None:
            a = a + self.asset_pos.unsqueeze(0)
        c = self.ctx_proj(ctx_feats) + self.ctx_pos                       # (B, d)
        seq = torch.cat([c.unsqueeze(1), a], dim=1)                       # (B, N+1, d)
        return seq


class TransformerBlock(nn.Module):
    """Pre-LN block: LN -> MHA -> +res -> LN -> MLP -> +res."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        y = self.ln1(x)
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #

class CholeskyHead(nn.Module):
    """Map per-asset embeddings to the entries of a lower-triangular L of size N.

    For each asset i we predict:
        - one diagonal scalar (passed through softplus + floor),
        - i off-diagonal scalars L[i, j] for j < i.
    Total params per asset = i + 1, so total = N(N+1)/2.

    We accomplish this by predicting a flat N-dim vector per asset and then
    masking with `tril_indices`. Concretely, each asset embedding maps to a
    small head producing N scalars; row i keeps the first (i+1) of them.
    """

    def __init__(self, n_assets: int, d_model: int, min_diag: float):
        super().__init__()
        self.n = n_assets
        self.min_diag = min_diag
        self.head = nn.Linear(d_model, n_assets)
        # Mask: row i keeps cols 0..i (lower triangular incl. diag).
        tril = torch.tril(torch.ones(n_assets, n_assets), diagonal=0)
        self.register_buffer("tril_mask", tril)

    def forward(self, asset_emb: Tensor) -> Tensor:
        """asset_emb: (B, N, d) -> L: (B, N, N) lower-triangular."""
        raw = self.head(asset_emb)                  # (B, N, N)
        L = raw * self.tril_mask                    # zero above diagonal
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_pos = torch.nn.functional.softplus(diag) + self.min_diag
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag_pos)
        return L


class MeanHead(nn.Module):
    """Per-asset scalar -> expected forward log return."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, asset_emb: Tensor) -> Tensor:
        return self.head(asset_emb).squeeze(-1)     # (B, N)


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #

class CrossAssetTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(
            cfg.n_assets, cfg.asset_in_dim, cfg.ctx_in_dim, cfg.d_model,
            use_asset_pos=cfg.use_asset_pos,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        if cfg.head_type == "factor":
            self.chol_head = FactorCholeskyHead(
                cfg.n_assets, cfg.d_model, n_factors=cfg.n_factors,
                min_diag=cfg.chol_min_diag,
            )
        else:
            self.chol_head = CholeskyHead(cfg.n_assets, cfg.d_model, cfg.chol_min_diag)
        self.mean_head = MeanHead(cfg.d_model)

    def forward(
        self, asset_feats: Tensor, ctx_feats: Tensor, mask: Tensor
    ) -> dict[str, Tensor]:
        """
        asset_feats : (B, N, F_a)
        ctx_feats   : (B, F_c)
        mask        : (B, N) bool, True = tradable. Will be inverted for
                      attention's key_padding_mask convention (True = ignore).

        Returns dict with:
          L     : (B, N, N) lower-triangular Cholesky factor
          mu    : (B, N) expected forward returns
        """
        seq = self.embed(asset_feats, ctx_feats)                   # (B, N+1, d)
        # CTX is always attendable; assets follow `mask`.
        B, N = mask.shape
        ctx_keep = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        keep = torch.cat([ctx_keep, mask], dim=1)                  # (B, N+1)
        kpm = ~keep                                                # True = ignore
        for blk in self.blocks:
            seq = blk(seq, key_padding_mask=kpm)
        seq = self.final_ln(seq)
        ctx_emb = seq[:, 0, :]                                     # (B, d)
        asset_emb = seq[:, 1:, :]                                  # (B, N, d)
        if self.cfg.head_type == "factor":
            L = self.chol_head(asset_emb, ctx_emb)                 # (B, N, N)
        else:
            L = self.chol_head(asset_emb)                          # (B, N, N)
        mu = self.mean_head(asset_emb)                             # (B, N)
        return {"L": L, "mu": mu}
