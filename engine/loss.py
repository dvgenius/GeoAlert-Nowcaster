"""
Physics-Informed Weather Loss Module (PINN-Inspired)
Enforces thermodynamic consistency by penalizing high Cloudburst risk predictions
in pixels lacking adequate Integrated Water Vapor (IWV) or Cloud Top Temperature drop rate (-dCTT/dt).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedWeatherLoss(nn.Module):
    """
    Multi-task loss with thermodynamic physical regularization.
    Total Loss = L_bce(TS) + L_bce(CB) + L_bce(FF) + lambda_phys * L_physics

    Physics constraint:
    In atmospheric physics, extreme convective cloudburst precipitation is physically
    impossible without:
    1) High column water vapor (IWV)
    2) Rapid cloud-top cooling (-dCTT/dt > 0) indicating penetrative convective updrafts.
    Any predicted cloudburst probability that exceeds the joint support of (IWV, CTT_DROP)
    incurs a quadratic penalty.
    """
    def __init__(self, lambda_physics=2.5, bce_weights=None):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.bce = nn.BCELoss()
        self.weights = bce_weights or {"thunderstorm": 1.0, "cloudburst": 1.5, "flash_flood": 1.2}

    def compute_physics_penalty(self, prob_cloudburst, x_input):
        """
        Calculates thermodynamic violation penalty:
        x_input: [B, Channels=5, T=4, H=64, W=64]
        prob_cloudburst: [B, 1, H=64, W=64]
        """
        # Extract IWV and CTT_DROP at current observation time step T0
        # Channels: 0 = IWV, 3 = CTT_DROP
        iwv_norm = x_input[:, 0:1, -1, :, :]       # [B, 1, 64, 64] in [0, 1]
        ctt_drop_norm = x_input[:, 3:4, -1, :, :]  # [B, 1, 64, 64] in [0, 1]

        # Physical thermodynamic ceiling for cloudburst occurrence:
        # If either moisture or updraft velocity is near zero, cloudburst probability MUST be near zero.
        # Joint physical support modeled as smooth multiplicative or min envelope:
        physical_support = torch.sqrt(torch.clamp(iwv_norm * ctt_drop_norm, min=1e-6)) * 1.35
        physical_support = torch.clamp(physical_support, 0.0, 1.0)

        # Violation occurs when predicted probability strictly exceeds physical support
        violation = F.relu(prob_cloudburst - physical_support)
        penalty = torch.mean(violation ** 2)

        return penalty

    def forward(self, predictions, targets, x_input):
        """
        predictions: dict with 'thunderstorm', 'cloudburst', 'flash_flood' [B, 1, H, W]
        targets: dict with ground truth masks [B, 1, H, W]
        x_input: raw input tensor [B, 5, 4, H, W]
        """
        loss_ts = self.bce(predictions["thunderstorm"], targets["thunderstorm"])
        loss_cb = self.bce(predictions["cloudburst"], targets["cloudburst"])
        loss_ff = self.bce(predictions["flash_flood"], targets["flash_flood"])

        # Physics Penalty
        penalty_physics = self.compute_physics_penalty(predictions["cloudburst"], x_input)

        total_loss = (
            self.weights["thunderstorm"] * loss_ts +
            self.weights["cloudburst"] * loss_cb +
            self.weights["flash_flood"] * loss_ff +
            self.lambda_physics * penalty_physics
        )

        return {
            "loss": total_loss,
            "loss_ts": loss_ts,
            "loss_cb": loss_cb,
            "loss_ff": loss_ff,
            "penalty_physics": penalty_physics,
        }
