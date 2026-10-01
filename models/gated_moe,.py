"""
models/gated_moe.py  —  Gated-MoE model (Section 2.4.3)

Same four GRU experts as MoEModel, but with a genuine learned gate: a
single linear layer (no hidden layers) mapping the mean-pooled 48-dim
fused representation to 4 gating weights (softmax-normalized), which
blend the four experts' final hidden states via weighted sum before a
shared linear classifier.

Added to test whether replacing the label-conditioned routing mechanism
with a classic input-dependent gate changes classification or verification
performance (Section 3.1 / Appendix A).
"""

import torch
import torch.nn as nn
from .encoder import SensorFusionFrontend


class GatedMoEModel(nn.Module):
    """
    Gate input:    mean-pooled fused sequence, 48-dim
    Gate network:  single linear layer (48 → n_classes) + softmax
    Combination:   single per-sequence gating decision blending experts'
                   final hidden states before a shared classifier
    """

    def __init__(self, fused_dim: int = 48, expert_hidden: int = 32,
                 n_classes: int = 4, **frontend_kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.frontend  = SensorFusionFrontend(fused_dim=fused_dim, **frontend_kwargs)
        self.experts   = nn.ModuleList([
            nn.GRU(fused_dim, expert_hidden, batch_first=True)
            for _ in range(n_classes)
        ])
        # Gate: input-only, no access to the declared label
        self.gate       = nn.Linear(fused_dim, n_classes)
        self.classifier = nn.Linear(expert_hidden, n_classes)

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor,
                return_gate_weights: bool = False):
        fused        = self.frontend(x_s1, x_s2)               # (B, T, fused_dim)
        gate_weights = torch.softmax(self.gate(fused.mean(1)), dim=-1)  # (B, n_classes)

        hiddens = []
        for expert in self.experts:
            _, h_n = expert(fused)
            hiddens.append(h_n.squeeze(0))                      # (B, hidden)
        hiddens  = torch.stack(hiddens, dim=1)                  # (B, n_classes, hidden)

        blended  = (gate_weights.unsqueeze(-1) * hiddens).sum(1)  # (B, hidden)
        logits   = self.classifier(blended)                     # (B, n_classes)

        if return_gate_weights:
            return logits, gate_weights
        return logits, None


def load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary loss discouraging gate collapse (all samples routed to one
    expert). Penalizes the coefficient of variation of mean per-expert gate
    usage across the batch; zero when usage is perfectly balanced.
    """
    mu = gate_weights.mean(dim=0)
    return mu.std() / (mu.mean() + 1e-8)
