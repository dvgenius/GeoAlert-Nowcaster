"""
Quick baseline model trainer to produce trained weights checkpoint for Phase 3/4.
"""

import os
import torch
from torch.utils.data import DataLoader
from engine.dataset import WeatherTensorDataset
from engine.model import MultiTaskWeatherNowcaster
from engine.loss import PhysicsInformedWeatherLoss


def train_baseline(epochs=5, batch_size=4, lr=1e-3, save_path="weights/nowcaster_baseline.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training baseline model on {device}...")

    dataset = WeatherTensorDataset(num_samples=32, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiTaskWeatherNowcaster(in_channels=5, time_steps=4, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = PhysicsInformedWeatherLoss(lambda_physics=2.5)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_phys = 0.0
        for x, targets in dataloader:
            x = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            preds = model(x)
            loss_dict = loss_fn(preds, targets, x)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_phys += loss_dict["penalty_physics"].item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        avg_phys = total_phys / len(dataset)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Physics Penalty: {avg_phys:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f" Saved trained baseline checkpoint to {save_path}")


if __name__ == "__main__":
    train_baseline()
