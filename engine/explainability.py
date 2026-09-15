"""
Explainable AI (XAI) Module using Captum Integrated Gradients
Quantifies hyper-local feature attribution across multivariate channels
(IWV, CAPE, CIN, CTT Drop Rate, DEM Topography) for transparent decision-making.
"""

import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients

CHANNEL_LABELS = {
    0: {"name": "IWV", "full_name": "Integrated Water Vapor (Atmospheric Moisture)"},
    1: {"name": "CAPE", "full_name": "Convective Available Potential Energy (Instability)"},
    2: {"name": "CIN", "full_name": "Convective Inhibition (Cap Breaking Energy)"},
    3: {"name": "CTT_DROP", "full_name": "Cloud Top Cooling Rate (Updraft Velocity)"},
    4: {"name": "DEM", "full_name": "Himalayan Topography / Valley Slope"},
}


class ModelHazardWrapper(nn.Module):
    """
    Subnet wrapper for Captum attribution targeting a specific hazard head.
    Outputs a scalar regional hazard index.
    """
    def __init__(self, model, hazard_name="cloudburst"):
        super().__init__()
        self.model = model
        self.hazard_name = hazard_name

    def forward(self, x):
        outputs = self.model(x)
        # Average risk across regional grid for target hazard
        return outputs[self.hazard_name].mean(dim=(1, 2, 3))


class WeatherExplainer:
    """
    Computes Integrated Gradients feature importance for severe weather nowcasts.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def attribute(self, input_tensor, target_hazard="cloudburst", n_steps=15):
        """
        Computes feature attribution breakdown for a single input or batch.
        input_tensor: [1, Channels=5, Time_Steps=4, H=64, W=64] or [5, 4, 64, 64]
        Returns:
          dict containing:
            - channel_percentages: dict of channel name -> percentage (0-100)
            - spatial_heatmap: [H, W] normalized spatial attribution map
            - raw_attributions: numpy array [5, 4, H, W]
        """
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        x = input_tensor.to(device).clone().detach().requires_grad_(True)

        # Baseline: neutral atmospheric background (zeros or mid-values)
        baseline = torch.zeros_like(x, device=device)

        wrapper = ModelHazardWrapper(self.model, hazard_name=target_hazard)
        ig = IntegratedGradients(wrapper)

        # Calculate Integrated Gradients
        attributions = ig.attribute(x, baselines=baseline, n_steps=n_steps)
        attr_np = attributions.detach().cpu().numpy()[0]  # [5, 4, H, W]

        # Aggregate absolute magnitude per channel across time and space
        channel_magnitudes = {}
        total_mag = 0.0

        for ch_idx in range(5):
            ch_data = np.abs(attr_np[ch_idx])
            mag = float(np.sum(ch_data))
            channel_magnitudes[ch_idx] = mag
            total_mag += mag

        # Prevent division by zero if all attributions are vanishing
        if total_mag < 1e-8:
            total_mag = 1.0

        # Percentages
        channel_breakdown = {}
        for ch_idx, mag in channel_magnitudes.items():
            pct = round((mag / total_mag) * 100.0, 2)
            meta = CHANNEL_LABELS[ch_idx]
            channel_breakdown[meta["name"]] = {
                "percentage": pct,
                "full_name": meta["full_name"],
                "channel_index": ch_idx
            }

        # Spatial attribution heatmap: sum across channels and time for grid visualization
        spatial_heatmap = np.sum(np.abs(attr_np), axis=(0, 1))  # [H, W]
        if spatial_heatmap.max() > 0:
            spatial_heatmap = spatial_heatmap / spatial_heatmap.max()

        return {
            "target_hazard": target_hazard,
            "channel_breakdown": channel_breakdown,
            "spatial_heatmap": spatial_heatmap,
            "raw_attributions": attr_np
        }
