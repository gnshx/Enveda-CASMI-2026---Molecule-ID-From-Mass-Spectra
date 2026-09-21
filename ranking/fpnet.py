"""
ranking/fpnet.py

Deep Residual Spectrum-to-Fingerprint Neural Network (FPNet) for CASMI 2026.
Predicts 2,048-bit Morgan (radius 2, ECFP4) + 166-bit MACCS keys from MS/MS spectra.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 2048, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FPNet(nn.Module):
    """
    FPNet: Binned MS/MS Spectrum -> 2,214-dimensional Fingerprint Logits
    """
    def __init__(
        self,
        in_dim: int = 4096 + 3,    # 4096 spectral bins + (precursor_mz, ce, ion_mode)
        hidden_dim: int = 2048,
        out_dim: int = 2048 + 166, # 2048 Morgan + 166 MACCS
        dropout: float = 0.2,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h)
        return logits

    @torch.no_grad()
    def predict_fingerprint(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities in [0, 1]."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)
