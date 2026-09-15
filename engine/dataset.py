"""
Weather Tensor Dataset & Geospatial Preprocessing
Computes temporal CTT drop rate (-dCTT/dt), applies domain normalization,
and stacks multivariate channels into [Channels=5, Time_Steps=4, H=64, W=64] tensors.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import netCDF4 as nc
import rasterio

# Channel Indices
CH_IWV = 0
CH_CAPE = 1
CH_CIN = 2
CH_CTT_DROP = 3
CH_DEM = 4
CHANNEL_NAMES = ["IWV", "CAPE", "CIN", "CTT_DROP_RATE", "DEM"]


def compute_ctt_drop_rate(ctt_series, delta_t_hours=0.25):
    """
    Computes the temporal drop rate of Cloud Top Temperature (-dCTT/dt in K/hr).
    ctt_series: numpy array of shape [T=4, H=64, W=64]
    Returns: drop_rate of shape [T=4, H=64, W=64]
    Positive drop rate indicates rapid cloud-top cooling and vigorous convective updraft.
    """
    time_steps, h, w = ctt_series.shape
    drop_rate = np.zeros_like(ctt_series, dtype=np.float32)

    # For t > 0: backward difference (cooling rate)
    for t in range(1, time_steps):
        # -dCTT/dt = (CTT_{t-1} - CTT_t) / delta_t
        drop_rate[t] = (ctt_series[t - 1] - ctt_series[t]) / delta_t_hours

    # For t=0: extrapolate forward difference to preserve continuity
    if time_steps > 1:
        drop_rate[0] = (ctt_series[0] - ctt_series[1]) / delta_t_hours
    
    # Clip negative cooling (warming) to slight negative/zero, max to 90 K/hr
    drop_rate = np.clip(drop_rate, -10.0, 90.0)
    return drop_rate.astype(np.float32)


def normalize_weather_tensor(iwv, cape, cin, ctt_drop, dem):
    """
    Applies physical domain normalization to [0, 1] range:
    - IWV: (iwv - 15.0) / 55.0
    - CAPE: cape / 3500.0
    - CIN: (cin + 150.0) / 150.0 (where 0 is capped, 1 is free convection)
    - CTT_DROP: ctt_drop / 60.0
    - DEM: (dem - 1200.0) / 4000.0
    """
    iwv_norm = np.clip((iwv - 15.0) / 55.0, 0.0, 1.0)
    cape_norm = np.clip(cape / 3500.0, 0.0, 1.0)
    cin_norm = np.clip((cin + 150.0) / 150.0, 0.0, 1.0)
    ctt_drop_norm = np.clip(ctt_drop / 60.0, 0.0, 1.0)
    dem_norm = np.clip((dem - 1200.0) / 4000.0, 0.0, 1.0)

    return iwv_norm, cape_norm, cin_norm, ctt_drop_norm, dem_norm


def build_spatiotemporal_tensor(iwv, cape, cin, ctt_series, dem, time_steps=4):
    """
    Constructs a 4D tensor [Channels=5, Time_Steps=4, H=64, W=64].
    """
    ctt_drop_series = compute_ctt_drop_rate(ctt_series)
    h, w = dem.shape

    # Normalize individual fields
    iwv_n, cape_n, cin_n, ctt_drop_n, dem_n = normalize_weather_tensor(
        iwv, cape, cin, ctt_drop_series, dem
    )

    tensor = np.zeros((5, time_steps, h, w), dtype=np.float32)

    for t in range(time_steps):
        # Moisture, CAPE, and CIN can undergo temporal progression or advection
        # Simulating slight diurnal/synoptic ramp towards T0
        t_weight = 0.85 + (t / (time_steps - 1)) * 0.15
        tensor[CH_IWV, t] = np.clip(iwv_n * t_weight, 0.0, 1.0)
        tensor[CH_CAPE, t] = np.clip(cape_n * t_weight, 0.0, 1.0)
        tensor[CH_CIN, t] = np.clip(cin_n * t_weight, 0.0, 1.0)
        tensor[CH_CTT_DROP, t] = ctt_drop_n[t]
        tensor[CH_DEM, t] = dem_n  # Static topography

    return torch.from_numpy(tensor).float()


class WeatherTensorDataset(Dataset):
    """
    PyTorch Dataset loading Chamoli spatiotemporal meteorological cubes.
    Outputs:
      x: [5, 4, 64, 64] tensor (Channels: IWV, CAPE, CIN, CTT_DROP, DEM)
      targets: dict of 3 hazard maps [1, 64, 64] for:
        - thunderstorm
        - cloudburst
        - flash_flood
    """
    def __init__(self, nc_path="data/chamoli_nowcast_input.nc", num_samples=32, augment=False, seed=42):
        super().__init__()
        self.nc_path = nc_path
        self.num_samples = num_samples
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        if os.path.exists(nc_path):
            with nc.Dataset(nc_path, "r") as ds:
                self.base_dem = ds.variables["dem"][:].astype(np.float32)
                self.base_iwv = ds.variables["iwv"][:].astype(np.float32)
                self.base_cape = ds.variables["cape"][:].astype(np.float32)
                self.base_cin = ds.variables["cin"][:].astype(np.float32)
                self.base_ctt = ds.variables["ctt"][:].astype(np.float32)
        else:
            # Fallback inline generation if file not present yet
            from mock_data_generator import generate_chamoli_topography, generate_atmospheric_fields
            self.base_dem = generate_chamoli_topography()
            atm_fields = generate_atmospheric_fields(self.base_dem)
            if len(atm_fields) == 5:
                self.base_iwv, self.base_cape, self.base_cin, _, self.base_ctt = atm_fields
            else:
                self.base_iwv, self.base_cape, self.base_cin, self.base_ctt = atm_fields

    def __len__(self):
        return self.num_samples

    def _generate_synthetic_targets(self, tensor):
        """
        Generates physically grounded ground-truth hazard labels [1, 64, 64]:
        - Thunderstorm: Driven by high CAPE (>0.5) and CIN release (>0.6) and CTT drop (>0.4)
        - Cloudburst: Driven by extreme IWV (>0.65) + extreme CTT drop (>0.5)
        - Flash Flood: Driven by Cloudburst precipitation + steep mountain slopes / valley accumulation
        """
        # Tensor shape: [5, 4, 64, 64]
        # At final time step T=3 (T0)
        iwv_t0 = tensor[CH_IWV, -1].numpy()
        cape_t0 = tensor[CH_CAPE, -1].numpy()
        cin_t0 = tensor[CH_CIN, -1].numpy()
        drop_t0 = tensor[CH_CTT_DROP, -1].numpy()
        dem = tensor[CH_DEM, -1].numpy()

        # 1. Thunderstorm probability surface
        ts_prob = 0.45 * cape_t0 + 0.25 * cin_t0 + 0.30 * drop_t0
        ts_prob = np.clip(ts_prob, 0.0, 1.0)
        ts_target = (ts_prob > 0.48).astype(np.float32)

        # 2. Cloudburst probability surface: requires BOTH high moisture & violent cooling
        cb_prob = (iwv_t0 ** 1.3) * (drop_t0 ** 1.1) * (cape_t0 ** 0.6) * 1.5
        cb_prob = np.clip(cb_prob, 0.0, 1.0)
        cb_target = (cb_prob > 0.42).astype(np.float32)

        # 3. Flash Flood probability: Cloudburst water funneled into valley bottoms (low DEM, high slope)
        # Topographic convergence: valleys (lower DEM) collect runoff from surrounding high ridges
        valley_factor = 1.0 - dem  # Low elevation = higher inundation vulnerability
        ff_prob = 0.55 * cb_prob + 0.45 * (valley_factor * cb_prob * 1.8)
        ff_prob = np.clip(ff_prob, 0.0, 1.0)
        ff_target = (ff_prob > 0.40).astype(np.float32)

        return {
            "thunderstorm": torch.from_numpy(ts_target).unsqueeze(0).float(),
            "cloudburst": torch.from_numpy(cb_target).unsqueeze(0).float(),
            "flash_flood": torch.from_numpy(ff_target).unsqueeze(0).float(),
        }

    def __getitem__(self, idx):
        # Add slight stochastic noise if augment or for diverse batch generation
        if self.augment or idx > 0:
            noise_scale = 0.05
            iwv = self.base_iwv + self.rng.normal(0, 1.0, self.base_iwv.shape).astype(np.float32) * noise_scale * 10
            cape = self.base_cape + self.rng.normal(0, 1.0, self.base_cape.shape).astype(np.float32) * noise_scale * 300
            cin = self.base_cin + self.rng.normal(0, 1.0, self.base_cin.shape).astype(np.float32) * noise_scale * 10
            ctt = self.base_ctt + self.rng.normal(0, 1.0, self.base_ctt.shape).astype(np.float32) * noise_scale * 5
            dem = self.base_dem
        else:
            iwv = self.base_iwv
            cape = self.base_cape
            cin = self.base_cin
            ctt = self.base_ctt
            dem = self.base_dem

        x_tensor = build_spatiotemporal_tensor(iwv, cape, cin, ctt, dem)
        targets = self._generate_synthetic_targets(x_tensor)
        return x_tensor, targets
