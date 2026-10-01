"""
training/lightning_module.py

PyTorch Lightning module for UniversalModel, MoEModel, and GatedMoEModel.

All three share the same training loop, optimizer, and scheduler.
The module dispatches to the correct model class based on model_type.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    N_CLASSES, LR, WEIGHT_DECAY, PATIENCE,
    S2_FEATURE_IDX, S1_FEATURE_IDX,
)
from models.universal import UniversalModel
from models.moe import MoEModel
from models.gated_moe import GatedMoEModel


MODEL_REGISTRY = {
    "universal":  UniversalModel,
    "moe":        MoEModel,
    "gated_moe":  GatedMoEModel,
}


class CropModelLightning(pl.LightningModule):
    """
    LightningModule wrapping UniversalModel / MoEModel / GatedMoEModel.

    Parameters
    ----------
    model_type    : "universal" | "moe" | "gated_moe"
    use_s2, use_s1: sensor ablation flags
    lr, weight_decay: AdamW hyperparameters
    patience      : early stopping patience (for ReduceLROnPlateau)
    lb_coeff      : load-balance loss coefficient (GatedMoE only)
    """

    def __init__(
        self,
        model_type: str = "moe",
        use_s2: bool = True,
        use_s1: bool = True,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
        patience: int = PATIENCE,
        lb_coeff: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        cls = MODEL_REGISTRY.get(model_type)
        if cls is None:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from {list(MODEL_REGISTRY)}."
            )
        self.model = cls(use_s2=use_s2, use_s1=use_s1)
        self.model_type = model_type
        self.lb_coeff   = lb_coeff

        # Metrics (macro average across 4 classes)
        mk = lambda: MulticlassAccuracy(num_classes=N_CLASSES, average="macro")
        f1 = lambda: MulticlassF1Score(num_classes=N_CLASSES, average="macro")
        self.train_acc = mk(); self.val_acc = mk(); self.test_acc = mk()
        self.train_f1  = f1(); self.val_f1  = f1(); self.test_f1  = f1()

    # ------------------------------------------------------------------
    def _unpack_batch(self, batch):
        x_s2, x_s1, mask_s2, mask_s1, labels, countries, uids = batch
        return x_s2, x_s1, mask_s2, mask_s1, labels

    def _forward(self, x_s2, x_s1, mask_s2, mask_s1):
        """Returns (logits, extra) where extra may be None or gate weights."""
        out = self.model(x_s2, x_s1, mask_s2, mask_s1)
        if isinstance(out, tuple):
            logits, extra = out
        else:
            logits, extra = out, None
        return logits, extra

    def _step(self, batch, stage: str):
        x_s2, x_s1, mask_s2, mask_s1, labels = self._unpack_batch(batch)
        logits, extra = self._forward(x_s2, x_s1, mask_s2, mask_s1)

        ce_loss = F.cross_entropy(logits, labels)
        loss = ce_loss

        # Load-balance penalty for GatedMoE
        if self.model_type == "gated_moe" and extra is not None:
            lb = self.model.load_balance_loss(extra)
            loss = loss + self.lb_coeff * lb
            self.log(f"{stage}/lb_loss", lb, prog_bar=False, on_epoch=True)

        preds = logits.argmax(dim=-1)
        accs  = getattr(self, f"{stage}_acc")
        f1s   = getattr(self, f"{stage}_f1")
        accs(preds, labels)
        f1s(preds, labels)

        self.log(f"{stage}/loss",     loss,           prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/ce_loss",  ce_loss,        prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc",      accs,           prog_bar=True,  on_epoch=True, on_step=False)
        self.log(f"{stage}/f1_macro", f1s,            prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        x_s2, x_s1, mask_s2, mask_s1, labels = self._unpack_batch(batch)
        logits, _ = self._forward(x_s2, x_s1, mask_s2, mask_s1)
        loss  = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.log("test/loss",     loss,          prog_bar=False, on_epoch=True)
        self.log("test/acc",      self.test_acc, prog_bar=True,  on_epoch=True)
        self.log("test/f1_macro", self.test_f1,  prog_bar=True,  on_epoch=True)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=self.hparams.patience // 2,
            factor=0.5, min_lr=1e-6,
        )
        return {
            "optimizer":  opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor":   "val/loss",
                "interval":  "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Inference helpers (used in experiments/)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, x_s2, x_s1, mask_s2, mask_s1) -> torch.Tensor:
        """Return softmax probabilities (N, C)."""
        logits, _ = self._forward(x_s2, x_s1, mask_s2, mask_s1)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def verification_score(
        self, x_s2, x_s1, mask_s2, mask_s1, declared_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Declaration-verification score.

        score = max_c p(c) − p(declared_c)

        Near 0 → declared crop matches model's top prediction.
        Large  → model sees a different crop as more likely.

        Defined for MoE and Universal; GatedMoE uses the same formula
        since it also produces a 4-way softmax.
        """
        proba = self.predict_proba(x_s2, x_s1, mask_s2, mask_s1)   # (N, C)
        top   = proba.max(dim=-1).values                              # (N,)
        p_dec = proba[torch.arange(len(declared_label)), declared_label]
        return top - p_dec
