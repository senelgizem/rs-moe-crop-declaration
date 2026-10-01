"""
models/universal.py  —  Universal model (Section 2.4.1)

Single shared GRU temporal backbone. Establishes the classification baseline
against which the MoE and Gated-MoE architectures are compared.
"""

import torch
import torch.nn as nn
from .encoder import SensorFusionFrontend


class UniversalModel(nn.Module):
    """
    One shared GRU (hidden_size=32) processes the fused sequence; its
    final hidden state is passed to a linear classification head.

    forward() returns (logits, pooled_repr) to keep the interface consistent
    with MoEModel, which also returns a secondary tensor.
    """

    def __init__(self, fused_dim: int = 48, backbone_hidden: int = 32,
                 n_classes: int = 4, **frontend_kwargs):
        super().__init__()
        self.frontend  = SensorFusionFrontend(fused_dim=fused_dim, **frontend_kwargs)
        self.backbone  = nn.GRU(fused_dim, backbone_hidden, batch_first=True)
        self.classifier = nn.Linear(backbone_hidden, n_classes)

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor):
        fused   = self.frontend(x_s1, x_s2)        # (B, T, fused_dim)
        _, h_n  = self.backbone(fused)              # h_n: (1, B, hidden)
        pooled  = h_n.squeeze(0)                    # (B, hidden)
        logits  = self.classifier(pooled)           # (B, n_classes)
        return logits, pooled
