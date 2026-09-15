"""
Multi-Task Spatiotemporal Weather Nowcaster Architecture
Backbone: Spatiotemporal ConvLSTM / U-Net Feature Extractor
Heads: 3 distinct hazard risk probability maps (Thunderstorm, Cloudburst, Flash Flood)
Topographic Fusion: Explicitly fuses high-resolution DEM elevation into Flash Flood head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    2D Convolutional LSTM Cell for capturing spatiotemporal dynamics across time steps.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        h, w = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_channels, h, w, device=device),
            torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        )


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskWeatherNowcaster(nn.Module):
    """
    Spatiotemporal Multi-Task Deep Learning Model for Mountainous Nowcasting.
    Input: [B, Channels=5, Time_Steps=4, H=64, W=64]
    Outputs:
      - 'thunderstorm': [B, 1, 64, 64] in [0, 1]
      - 'cloudburst':   [B, 1, 64, 64] in [0, 1]
      - 'flash_flood':  [B, 1, 64, 64] in [0, 1] (DEM Topography Fused)
    """
    def __init__(self, in_channels=5, time_steps=4, hidden_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim

        # 1. Temporal Processor: ConvLSTM Cell
        self.conv_lstm = ConvLSTMCell(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            kernel_size=3
        )

        # 2. Spatial U-Net Encoder-Decoder on the accumulated temporal state
        # Encoder
        self.enc1 = DoubleConv(hidden_dim, 32)      # 64x64
        self.pool1 = nn.MaxPool2d(2)                # -> 32x32
        self.enc2 = DoubleConv(32, 64)              # 32x32
        self.pool2 = nn.MaxPool2d(2)                # -> 16x16
        self.bottleneck = DoubleConv(64, 128)       # 16x16

        # Decoder with Skip Connections
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # -> 32x32
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # -> 64x64
        self.dec1 = DoubleConv(64, 32)

        # 3. Multi-Task Heads
        # Head 1: Thunderstorm Risk Head
        self.head_thunderstorm = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head 2: Cloudburst Risk Head
        self.head_cloudburst = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head 3: Flash Flood Risk Head (Fuses latent feature + DEM channel + Cloudburst feature)
        # in_channels: 32 (latent) + 1 (DEM elevation) + 1 (Cloudburst risk prior) = 34
        self.head_flash_flood = nn.Sequential(
            nn.Conv2d(32 + 1 + 1, 24, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(12, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        """
        Passes sequence through ConvLSTM and U-Net backbone.
        x: [B, C=5, T=4, H=64, W=64]
        Returns: latent feature map [B, 32, 64, 64]
        """
        batch_size, _, time_steps, h, w = x.shape
        h_t, c_t = self.conv_lstm.init_hidden(batch_size, (h, w), x.device)

        # Sequential temporal rollout
        for t in range(time_steps):
            x_t = x[:, :, t, :, :]  # [B, C, H, W]
            h_t, c_t = self.conv_lstm(x_t, h_t, c_t)

        # U-Net pass on final accumulated hidden state
        e1 = self.enc1(h_t)             # [B, 32, 64, 64]
        p1 = self.pool1(e1)             # [B, 32, 32, 32]
        e2 = self.enc2(p1)              # [B, 64, 32, 32]
        p2 = self.pool2(e2)             # [B, 64, 16, 16]
        b = self.bottleneck(p2)         # [B, 128, 16, 16]

        d2 = self.up2(b)                # [B, 64, 32, 32]
        d2 = torch.cat([d2, e2], dim=1) # [B, 128, 32, 32]
        d2 = self.dec2(d2)              # [B, 64, 32, 32]

        d1 = self.up1(d2)               # [B, 32, 64, 64]
        d1 = torch.cat([d1, e1], dim=1) # [B, 64, 64, 64]
        latent = self.dec1(d1)          # [B, 32, 64, 64]
        return latent

    def forward(self, x):
        """
        Forward pass producing 3 hazard risk surfaces.
        """
        # Ensure 5D input [B, C, T, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(0)

        latent = self.extract_features(x)

        # Head 1: Thunderstorm
        prob_thunderstorm = self.head_thunderstorm(latent)

        # Head 2: Cloudburst
        prob_cloudburst = self.head_cloudburst(latent)

        # Head 3: Flash Flood with explicit DEM Topography Fusion
        # DEM channel is index 4 at the current time step (last time step)
        dem_channel = x[:, 4:5, -1, :, :]  # [B, 1, 64, 64]
        fused_flood_input = torch.cat([latent, dem_channel, prob_cloudburst], dim=1)  # [B, 34, 64, 64]
        prob_flash_flood = self.head_flash_flood(fused_flood_input)

        return {
            "thunderstorm": prob_thunderstorm,
            "cloudburst": prob_cloudburst,
            "flash_flood": prob_flash_flood
        }

    def forward_single_hazard(self, x, hazard="cloudburst"):
        """Helper for XAI attribution targeting a specific hazard map mean."""
        out = self.forward(x)
        return out[hazard].mean(dim=(1, 2, 3))
