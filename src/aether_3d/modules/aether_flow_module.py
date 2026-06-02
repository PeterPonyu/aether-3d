"""
AetherFlowModule — Lightning training module for Aether3D multi-modal flow matching.

Handles the joint (spatial, gene, class) state and the weighted
multi-task loss (lambda_g, lambda_c).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch

from ..config.aether_config import Aether3DConfig
from ..flow import create_flow_transport
from ..models.aether_velocity_field import MultiModalVelocityField


class AetherFlowModule(pl.LightningModule):
    def __init__(self, config: Aether3DConfig, model: MultiModalVelocityField):
        super().__init__()
        self.save_hyperparameters(config.model_dump_for_checkpoint())
        self.cfg = config
        self.model = model
        self.ema_model = deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.transport = create_flow_transport(
            path=config.path_type,
            prediction=config.prediction,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _compute_multi_modal_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_x = torch.nn.functional.mse_loss(pred["vx"], target["vx"])
        loss_g = torch.nn.functional.mse_loss(pred["vg"], target["vg"])
        loss_c = torch.nn.functional.mse_loss(pred["vc"], target.get("vc", torch.zeros_like(pred["vc"])))

        total = loss_x + self.cfg.lambda_g * loss_g + self.cfg.lambda_c * loss_c
        return total, {"loss_x": loss_x, "loss_g": loss_g, "loss_c": loss_c}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # batch from SerialSliceTrajectoryDataset
        x0, g0, c0 = batch["x0"], batch["g0"], batch["c0"]
        x1, g1, c1 = batch["x1"], batch["g1"], batch["c1"]
        z0, z1 = batch["z0"], batch["z1"]
        delta_z = batch["delta_z"]

        # Route through the transport's shared sampler so this path and
        # FlowTransport.training_losses can never disagree on the time range,
        # and so transport.train_eps is honoured (issue #140).
        t = self.transport.sample_time(x0.shape[0], device=x0.device)

        # Interpolate states and get velocity targets using transport path planning
        _, xt, ux_t = self.transport.path.plan(t, x0, x1)
        _, gt, ug_t = self.transport.path.plan(t, g0, g1)
        _, ct, uc_t = self.transport.path.plan(t, c0, c1)
        _, zt, _ = self.transport.path.plan(t, z0, z1)

        # Condition on each source cell's encoded class rather than a constant
        # placeholder, so class-specific dynamics can influence the velocity.
        y = c0

        # Current state dictionary to pass to model
        state = {
            "x": xt,
            "g": gt,
            "c": ct,
            "z": zt,
            "delta_z": delta_z,
        }

        # Predict velocity
        pred = self.model(state, t, y)

        target = {
            "vx": ux_t,
            "vg": ug_t,
            "vc": uc_t,
        }

        loss, loss_dict = self._compute_multi_modal_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v)

        return loss

    @torch.no_grad()
    def _update_ema(self) -> None:
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.cfg.ema_decay).add_(p.data, alpha=1 - self.cfg.ema_decay)

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self._update_ema()
