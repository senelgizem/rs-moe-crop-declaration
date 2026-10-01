"""
models/encoder.py  —  Shared sensor-fusion frontend (Section 2.4)

Used by UniversalModel, MoEModel, and GatedMoEModel. 
Sentinel-1 and Sentinel-2 features are encoded by independent single-layer GRUs
(hidden_size=32 each), concatenated per time step, and projected through
a linear + ReLU layer to a 48-dimensional fused representation.

Missing observations are handled by zero-filling before this module; the
binary validity masks (mask_s1, mask_s2) are consumed by the dataset/
dataloader and not passed here -- the GRU sees zeros for missing steps,
which the reconstruction model separately exploits via masking the loss.

Sensor ablation (Section 2.5.2): set use_s1=False or use_s2=False at
construction time. The missing branch's encoder is not instantiated, and
its contribution to the concatenation is replaced by a zero tensor of
the same width, so downstream code (FusionLayer, crop experts) never needs
to change shape for ablation runs.
"""

import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    """Single-layer GRU for one sensor's per-timestep feature vector."""

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)  →  (B, T, hidden_size)
        out, _ = self.gru(x)
        return out


class FusionLayer(nn.Module):
    """Concatenate S1 + S2 per-timestep encodings → linear + ReLU → fused_dim."""

    def __init__(self, s1_hidden: int, s2_hidden: int, fused_dim: int = 48):
        super().__init__()
        self.proj       = nn.Linear(s1_hidden + s2_hidden, fused_dim)
        self.activation = nn.ReLU()

    def forward(self, h_s1: torch.Tensor, h_s2: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(torch.cat([h_s1, h_s2], dim=-1)))


class SensorFusionFrontend(nn.Module):
    """
    s2_input_size=10: B5, B6, B7, B11, B12, NDVI, EVI, NDWI, NDMI, s2_valid_obs
    s1_input_size=4:  VV, VH, VH_VV, s1_valid_obs
    """

    def __init__(self, s1_input_size: int = 4, s2_input_size: int = 10,
                 s1_hidden: int = 32, s2_hidden: int = 32, fused_dim: int = 48,
                 use_s1: bool = True, use_s2: bool = True):
        super().__init__()
        assert use_s1 or use_s2, "At least one sensor must be enabled."
        self.use_s1 = use_s1
        self.use_s2 = use_s2
        self.s1_encoder = SensorEncoder(s1_input_size, s1_hidden) if use_s1 else None
        self.s2_encoder = SensorEncoder(s2_input_size, s2_hidden) if use_s2 else None
        self.s1_out     = s1_hidden if use_s1 else 0
        self.s2_out     = s2_hidden if use_s2 else 0
        self.fusion     = FusionLayer(self.s1_out, self.s2_out, fused_dim)

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor) -> torch.Tensor:
        B, T = x_s1.shape[0], x_s1.shape[1]
        dev  = x_s1.device
        h_s1 = self.s1_encoder(x_s1) if self.use_s1 else torch.zeros(B, T, 0, device=dev)
        h_s2 = self.s2_encoder(x_s2) if self.use_s2 else torch.zeros(B, T, 0, device=dev)
        return self.fusion(h_s1, h_s2)              # (B, T, fused_dim)
