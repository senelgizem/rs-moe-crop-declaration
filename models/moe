"""
models/moe.py  —  Label-conditioned MoE model (Section 2.4.2)

Four independent GRU experts, each producing a scalar fit-score. Fit-scores
are stacked into 4-way logits and trained with ordinary cross-entropy against
the declared label — the label itself provides routing; no separate gating
network is needed or used (softmax over fit-scores IS the gate).
"""

import torch
import torch.nn as nn
from .encoder import SensorFusionFrontend


class CropExpert(nn.Module):
    """One crop-specific GRU sub-network → scalar fit-score."""

    def __init__(self, fused_dim: int = 48, hidden_size: int = 32):
        super().__init__()
        self.gru      = nn.GRU(fused_dim, hidden_size, batch_first=True)
        self.fit_head = nn.Linear(hidden_size, 1)

    def forward(self, fused: torch.Tensor):
        _, h_n     = self.gru(fused)          # h_n: (1, B, hidden)
        h          = h_n.squeeze(0)            # (B, hidden)
        fit_score  = self.fit_head(h).squeeze(-1)   # (B,)
        return fit_score, h


class MoEModel(nn.Module):
    """
    Four independent crop-expert GRUs (wheat, barley, maize, sunflower).
    Each produces a scalar fit-score; scores are stacked into 4-way logits.
    No separate gating network: softmax(logits) performs routing.
    """

    def __init__(self, fused_dim: int = 48, expert_hidden: int = 32,
                 n_classes: int = 4, **frontend_kwargs):
        super().__init__()
        self.frontend = SensorFusionFrontend(fused_dim=fused_dim, **frontend_kwargs)
        self.experts  = nn.ModuleList([
            CropExpert(fused_dim, expert_hidden) for _ in range(n_classes)
        ])

    def forward(self, x_s1: torch.Tensor, x_s2: torch.Tensor,
                return_expert_reprs: bool = False):
        fused = self.frontend(x_s1, x_s2)

        fit_scores, reprs = [], []
        for expert in self.experts:
            score, h = expert(fused)
            fit_scores.append(score)
            reprs.append(h)

        logits = torch.stack(fit_scores, dim=1)         # (B, n_classes)

        if return_expert_reprs:
            return logits, torch.stack(reprs, dim=1)    # (B, n_classes, hidden)
        return logits, None


def verification_score(logits: torch.Tensor, declared_label: torch.Tensor) -> torch.Tensor:
    """
    Softmax-gap verification score (Section 2.5.5):
        score = max(p) − p[declared]
    Near zero  → declared class is the best fit  (no mismatch signal).
    Large      → another class fits better        (potential mismatch).
    """
    probs         = torch.softmax(logits, dim=-1)
    declared_conf = probs.gather(1, declared_label.unsqueeze(1)).squeeze(1)
    best_conf, _  = probs.max(dim=-1)
    return best_conf - declared_conf
