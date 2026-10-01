"""
models/autoencoder.py  —  Reconstruction-based MoE autoencoder (Section 2.4.4)

Each label-conditioned expert is reimplemented as a GRU encoder →
bottleneck → GRU decoder, reconstructing the 14-feature sequence per
time step.

Training: hard routing — only the declared-class expert receives gradient
(via masked MSE loss). The other experts' reconstructions are computed in
the forward pass for diagnostic use but are excluded from the loss by
construction (not just stop-grad, but literally not summed into the loss
term for those experts).

Inference: all 4 experts' reconstruction errors are computed for every
parcel, enabling the declared-vs-best error gap used as the verification
score (Section 2.5.5 variant) and the per-month/per-feature diagnostics
of Section 2.5.4.
"""

import torch
import torch.nn as nn
from .encoder import SensorFusionFrontend


class AutoencoderExpert(nn.Module):
    """
    GRU encoder → bottleneck linear → GRU decoder → output projection.
    Reconstructs the full 14-feature sequence at every time step.
    """

    def __init__(self, fused_dim: int = 48, hidden_size: int = 32,
                 output_dim: int = 14, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder    = nn.GRU(fused_dim, hidden_size, batch_first=True)
        self.bottleneck = nn.Linear(hidden_size, bottleneck_dim)
        self.decoder    = nn.GRU(bottleneck_dim, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, fused: torch.Tensor):
        # fused: (B, T, fused_dim)
        enc_out, _     = self.encoder(fused)              # (B, T, hidden)
        bottleneck_seq = self.bottleneck(enc_out)          # (B, T, bottleneck_dim)
        dec_out, _     = self.decoder(bottleneck_seq)      # (B, T, hidden)
        recon          = self.output_proj(dec_out)         # (B, T, output_dim)
        return recon, enc_out


class MoEAutoencoderModel(nn.Module):
    """
    Four AutoencoderExperts sharing the same SensorFusionFrontend.
    forward() runs ALL experts (needed for diagnostic comparisons at
    inference); the training loss only back-propagates through the
    declared-class expert (see MoEAutoencoderLightning).
    """

    def __init__(self, fused_dim: int = 48, expert_hidden: int = 32,
                 output_dim: int = 14, n_classes: int = 4, **frontend_kwargs):
        super().__init__()
        self.frontend = SensorFusionFrontend(fused_dim=fused_dim, **frontend_kwargs)
        self.experts  = nn.ModuleList([
            AutoencoderExpert(fused_dim, expert_hidden, output_dim)
            for _ in range(n_classes)
        ])

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor):
        fused = self.frontend(x_s1, x_s2)              # (B, T, fused_dim)
        reconstructions = []
        for expert in self.experts:
            recon, _ = expert(fused)
            reconstructions.append(recon)
        return torch.stack(reconstructions, dim=1)     # (B, n_classes, T, output_dim)


def masked_mse(recon: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    MSE over valid (non-NaN origin) positions only.
    mask: True where the original value was valid (same shape as target).
    """
    sq_err = (recon - target) ** 2
    return (sq_err * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def reconstruction_verification_score(all_errors: torch.Tensor,
                                       declared_label: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction-based verification score (Section 2.5.5 variant):
        score = error[declared_class] − min_e(error[e])
    Near zero  → declared expert fits as well as any other.
    Large      → another expert reconstructs the sequence better.

    all_errors: (B, n_classes) per-expert scalar MSE aggregated over T and D.
    """
    declared_err = all_errors.gather(1, declared_label.unsqueeze(1)).squeeze(1)
    best_err     = all_errors.min(dim=1).values
    return declared_err - best_err
