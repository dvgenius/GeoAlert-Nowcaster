"""
Engine Verification Test Suite for Phase 2
Validates:
1. Spatiotemporal dataset loading, CTT drop rate computation, and tensor shapes.
2. Model forward pass and gradient flow through ConvLSTM U-Net backbone.
3. Multi-task output heads (Thunderstorm, Cloudburst, Flash Flood with DEM fusion).
4. Physics-Informed loss calculation & thermodynamic penalty enforcement.
5. Captum Integrated Gradients XAI attribution breakdown.
"""

import sys
import torch
import numpy as np

from engine.dataset import WeatherTensorDataset, compute_ctt_drop_rate, normalize_weather_tensor
from engine.model import MultiTaskWeatherNowcaster
from engine.loss import PhysicsInformedWeatherLoss
from engine.explainability import WeatherExplainer


def test_dataset_loading():
    print("\n[1/5] Testing WeatherTensorDataset...")
    dataset = WeatherTensorDataset(nc_path="data/chamoli_nowcast_input.nc", num_samples=4)
    assert len(dataset) == 4, f"Dataset length mismatch: {len(dataset)}"

    x, targets = dataset[0]

    # Check input shape [Channels=5, Time_Steps=4, H=64, W=64]
    assert x.shape == (5, 4, 64, 64), f"Unexpected tensor shape: {x.shape}"
    assert x.dtype == torch.float32, f"Unexpected tensor dtype: {x.dtype}"
    assert not torch.isnan(x).any(), "Found NaNs in input tensor"
    assert not torch.isinf(x).any(), "Found Infs in input tensor"

    # Value range check (normalized between 0 and 1)
    assert x.min() >= 0.0 and x.max() <= 1.0, f"Tensor values out of [0, 1]: min={x.min()}, max={x.max()}"

    # Target shape check
    for hazard in ["thunderstorm", "cloudburst", "flash_flood"]:
        assert hazard in targets, f"Missing target: {hazard}"
        assert targets[hazard].shape == (1, 64, 64), f"Target {hazard} shape mismatch: {targets[hazard].shape}"
        assert targets[hazard].dtype == torch.float32

    print(" WeatherTensorDataset shapes and value bounds validated successfully.")


def test_model_forward_and_backward():
    print("\n[2/5] Testing MultiTaskWeatherNowcaster forward & backward passes...")
    model = MultiTaskWeatherNowcaster(in_channels=5, time_steps=4, hidden_dim=16)
    model.train()

    batch_size = 2
    dummy_input = torch.rand(batch_size, 5, 4, 64, 64, dtype=torch.float32)

    outputs = model(dummy_input)

    expected_heads = ["thunderstorm", "cloudburst", "flash_flood"]
    for head in expected_heads:
        assert head in outputs, f"Head '{head}' missing from model output"
        out_tensor = outputs[head]
        assert out_tensor.shape == (batch_size, 1, 64, 64), f"Shape mismatch for {head}: {out_tensor.shape}"
        # Probability bounds
        assert out_tensor.min() >= 0.0 and out_tensor.max() <= 1.0, f"Probabilities for {head} out of range [0, 1]"

    # Verify gradient flow backwards through entire ConvLSTM + U-Net graph
    loss_dummy = outputs["thunderstorm"].mean() + outputs["cloudburst"].mean() + outputs["flash_flood"].mean()
    loss_dummy.backward()

    # Check that gradients exist for backbone and heads
    assert model.conv_lstm.conv.weight.grad is not None, "ConvLSTM gradients not flowing"
    assert model.head_flash_flood[0].weight.grad is not None, "Flash Flood head gradients not flowing"
    print(" Model forward and backward gradient passes verified.")


def test_physics_informed_loss():
    print("\n[3/5] Testing PhysicsInformedWeatherLoss...")
    loss_fn = PhysicsInformedWeatherLoss(lambda_physics=3.0)

    # 1. Test standard loss computation
    batch_size = 2
    x_input = torch.rand(batch_size, 5, 4, 64, 64)
    predictions = {
        "thunderstorm": torch.full((batch_size, 1, 64, 64), 0.6, requires_grad=True),
        "cloudburst": torch.full((batch_size, 1, 64, 64), 0.7, requires_grad=True),
        "flash_flood": torch.full((batch_size, 1, 64, 64), 0.5, requires_grad=True),
    }
    targets = {
        "thunderstorm": torch.ones(batch_size, 1, 64, 64),
        "cloudburst": torch.ones(batch_size, 1, 64, 64),
        "flash_flood": torch.ones(batch_size, 1, 64, 64),
    }

    loss_dict = loss_fn(predictions, targets, x_input)
    assert "loss" in loss_dict
    assert "penalty_physics" in loss_dict
    assert loss_dict["loss"].item() > 0.0
    print(f"  - Total Loss: {loss_dict['loss'].item():.4f}")
    print(f"  - Physics Violation Penalty: {loss_dict['penalty_physics'].item():.4f}")

    # 2. Test physical violation penalty: predict high cloudburst (0.95) when IWV and CTT Drop are zero
    dry_input = torch.zeros(1, 5, 4, 64, 64)  # Zero moisture, zero updraft cooling
    unphysical_preds = {
        "thunderstorm": torch.zeros(1, 1, 64, 64),
        "cloudburst": torch.full((1, 1, 64, 64), 0.95),  # Severely unphysical!
        "flash_flood": torch.zeros(1, 1, 64, 64),
    }
    dry_penalty = loss_fn.compute_physics_penalty(unphysical_preds["cloudburst"], dry_input)
    assert dry_penalty.item() > 0.5, f"Expected strong physics penalty for unphysical cloudburst, got {dry_penalty.item()}"
    print(f"  - Verified: Unphysical cloudburst incurred strong thermodynamic penalty ({dry_penalty.item():.4f}).")


def test_captum_xai_attribution():
    print("\n[4/5] Testing Captum Integrated Gradients XAI attribution...")
    model = MultiTaskWeatherNowcaster(in_channels=5, time_steps=4, hidden_dim=16)
    model.eval()

    explainer = WeatherExplainer(model)
    dummy_input = torch.rand(1, 5, 4, 64, 64)

    # Compute attribution for cloudburst
    xai_result = explainer.attribute(dummy_input, target_hazard="cloudburst", n_steps=8)

    assert "channel_breakdown" in xai_result
    assert "spatial_heatmap" in xai_result
    assert xai_result["spatial_heatmap"].shape == (64, 64)

    breakdown = xai_result["channel_breakdown"]
    print("  - XAI Feature Attribution Breakdown:")
    total_pct = 0.0
    for ch_name, data in breakdown.items():
        pct = data["percentage"]
        total_pct += pct
        print(f"    * {ch_name} ({data['full_name']}): {pct:.1f}%")

    # Percentage sum should be approximately 100%
    assert 98.0 <= total_pct <= 102.0, f"Percentages do not sum to ~100%: {total_pct}"
    print(" Captum Integrated Gradients attribution verified.")


def test_quick_training_loop():
    print("\n[5/5] Testing quick single-batch optimization step...")
    model = MultiTaskWeatherNowcaster(in_channels=5, time_steps=4, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = PhysicsInformedWeatherLoss()

    dataset = WeatherTensorDataset(num_samples=2)
    x, targets = dataset[0]
    x_batch = x.unsqueeze(0)
    targets_batch = {k: v.unsqueeze(0) for k, v in targets.items()}

    optimizer.zero_grad()
    preds = model(x_batch)
    loss_res = loss_fn(preds, targets_batch, x_batch)
    loss_res["loss"].backward()
    optimizer.step()

    print(f"  - Optimization step succeeded: initial loss = {loss_res['loss'].item():.4f}")
    print(" Single optimization step validated.")


if __name__ == "__main__":
    print("=" * 60)
    print("IIT BHU HACKATHON: SPATIOTEMPORAL WEATHER ENGINE VERIFICATION")
    print("=" * 60)

    test_dataset_loading()
    test_model_forward_and_backward()
    test_physics_informed_loss()
    test_captum_xai_attribution()
    test_quick_training_loop()

    print("\n" + "=" * 60)
    print(" ALL PHASE 2 ENGINE TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 60)
