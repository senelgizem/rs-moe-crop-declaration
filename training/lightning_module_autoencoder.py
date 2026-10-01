"""
training/lightning_module_autoencoder.py

PyTorch Lightning module for MoEAutoencoderModel.

The autoencoder is trained to reconstruct the input time series for each
crop-specific expert.  At inference time, the expert with the lowest
reconstruction error is used for declaration verification.

Loss: masked MSE summed over all 4 experts, each weighted by a soft
one-hot of the true label (during training) or the declared label
(during inference).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytorch_lightning as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import N_CLASSES, LR, WEIGHT_DECAY, PATIENCE
from models.autoencoder import MoEAutoencoderModel


class MoEAutoencoderLightning(pl.LightningModule):
    """
    Lightning wrapper for MoEAutoencoderModel.

    Training objective: minimise the masked reconstruction MSE for the
    expert that matches the parcel's true crop label.  All 4 experts run
    in parallel; only the labelled expert's loss updates its weights.

    Parameters
    ----------
    lr, weight_decay : AdamW hyperparameters
    """

    def __init__(
        self,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MoEAutoencoderModel()

    # ------------------------------------------------------------------
    def _unpack_batch(self, batch):
        x_s2, x_s1, mask_s2, mask_s1, labels, countries, uids = batch
        return x_s2, x_s1, mask_s2, mask_s1, labels

    def _step(self, batch, stage: str):
        x_s2, x_s1, mask_s2, mask_s1, labels = self._unpack_batch(batch)

        # reconstructions: list of 4 tensors, each (N, T, F_full)
        # errors        : (N, 4) masked MSE per expert per parcel
        recons, errors = self.model(x_s2, x_s1, mask_s2, mask_s1)

        # Select only the declared-crop expert error for training
        # labels: (N,) long  →  gather along expert axis
        labelled_errors = errors[torch.arange(len(labels)), labels]   # (N,)
        loss = labelled_errors.mean()

        self.log(f"{stage}/recon_loss", loss, prog_bar=True, on_epoch=True,
                 on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=PATIENCE // 2,
            factor=0.5, min_lr=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor":   "val/recon_loss",
                "interval":  "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Inference helpers (used in experiments/verify_declarations_recon.py)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruction_errors(
        self, x_s2, x_s1, mask_s2, mask_s1
    ) -> torch.Tensor:
        """
        Return per-expert reconstruction errors.

        Returns
        -------
        errors : (N, 4) float — masked MSE for each expert
        """
        _, errors = self.model(x_s2, x_s1, mask_s2, mask_s1)
        return errors

    @torch.no_grad()
    def reconstruction_verification_score(
        self, x_s2, x_s1, mask_s2, mask_s1, declared_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruction-based declaration verification score.

        score = error[declared] − min_e error[e]

        Near 0 → declared expert reconstructs as well as the best expert.
        Large  → a different expert reconstructs the parcel much better,
                 suggesting a crop-type mismatch.

        Parameters
        ----------
        declared_label : (N,) long — IACS-declared crop class index

        Returns
        -------
        score : (N,) float
        """
        errors   = self.reconstruction_errors(x_s2, x_s1, mask_s2, mask_s1)
        min_err  = errors.min(dim=-1).values
        dec_err  = errors[torch.arange(len(declared_label)), declared_label]
        return dec_err - min_err
