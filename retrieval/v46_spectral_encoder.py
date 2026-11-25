"""
retrieval/v46_spectral_encoder.py
=============================================================================
CASMI 2026 — V46: Independent timsTOF-Adapted Spectral Similarity Encoder

MS2DeepScore / DreaMS-inspired spectrum-to-spectrum structural similarity
encoder trained directly to predict pairwise Morgan-2048 Tanimoto similarity
between spectra across timsTOF and reference MS/MS libraries.

Orthogonal to V45's molecule-classification network: enables dense cosine
analog retrieval (`z_q @ Z_ref.T`) where spectral dot product approximates
molecular Tanimoto similarity.
=============================================================================
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectralSimilarityEncoder(nn.Module):
    """Transformer spectral encoder producing L2-normalized structural embeddings."""

    def __init__(self, width: int = 256, heads: int = 8, layers: int = 4, emb_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(10).float())
        # Peak features: 4 base + 20 sin + 20 cos = 44 dims
        self.peak_proj = nn.Linear(44, width)
        self.meta_proj = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.adduct_emb = nn.Embedding(11, width)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, emb_dim),
        )
        self.fp_head = nn.Linear(width, 2048)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, meta: torch.Tensor, adduct: torch.Tensor):
        # x[:, :, 0] = mz/1000, x[:, :, 3] = (prec - mz)/1000
        a = x[:, :, [0, 3]].unsqueeze(-1) * self.freq * torch.pi
        p = torch.cat([x, a.sin().flatten(2), a.cos().flatten(2)], dim=-1)
        tok = self.peak_proj(p)
        cls = self.cls_tok.expand(len(x), -1, -1) + self.meta_proj(meta)[:, None] + self.adduct_emb(adduct)[:, None]
        tok = torch.cat([cls, tok], dim=1)
        mask = torch.arange(x.shape[1] + 1, device=x.device)[None] > lengths[:, None]
        h = self.norm(self.encoder(tok, src_key_padding_mask=mask)[:, 0])
        z = F.normalize(self.proj(h), dim=-1)
        fp_logits = self.fp_head(h)
        return z, fp_logits
