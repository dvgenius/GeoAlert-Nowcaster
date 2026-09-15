"""
FastAPI Server for AI-Driven Hyper-Local Early Warning Nowcasting
Supports Dynamic Pan-India Bounding Boxes & Regional Presets for Hackathon Testing
"""
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.dataset import WeatherTensorDataset, build_spatiotemporal_tensor
from engine.model import MultiTaskWeatherNowcaster
from engine.explainability import WeatherExplainer
from server.geojson_utils import create_nowcast_feature_collection

app = FastAPI(
    title="GeoAlert Nowcaster API",
    description="Hyper-Local AI Early Warning System for Severe Weather Nowcasting with Pan-India Bounding Boxes",
    version="1.1.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://geo-alert-nowcaster.vercel.app",
        "https://geo-alert-nowcaster-o2n0zc2dz-dvgenius-9874s-projects.vercel.app",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5175",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
import os
import gc

# 1. Prevent thread explosion & disable GUI backend before importing matplotlib/torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import torch
torch.set_num_threads(1)

# Inside your nowcast prediction endpoint (e.g., get_nowcast_prediction):
# Ensure model inference runs strictly with no gradient tracking:
with torch.no_grad():
    # ... your model forward pass ...
    pass

# Run memory cleanup at the end of the endpoint before returning response
gc.collect()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[MultiTaskWeatherNowcaster] = None
explainer: Optional[WeatherExplainer] = None


INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 37.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 98.0
GRID_SIZE = 64


REGION_PRESETS: Dict[str, Dict[str, Any]] = {
    "uttarakhand": {
        "preset_id": "uttarakhand",
        "label": "Uttarakhand (Chamoli / Alaknanda Basin)",
        "min_lat": 30.15,
        "max_lat": 30.75,
        "min_lon": 79.25,
        "max_lon": 79.85,
        "elevation_range": "1,250m - 4,900m",
        "topography_type": "High Himalayan Peaks, Gorges & Glacial Valleys",
        "default_action": "Immediate mandatory evacuation of lower Alaknanda & Rishi Ganga valleys. Mobilize NDRF, activate district sirens.",
        "dem_base": 2800.0,
        "dem_scale": 1500.0,
        "dem_min": 1250.0,
        "dem_max": 4900.0,
    },
    "mumbai": {
        "preset_id": "mumbai",
        "label": "Mumbai Metropolitan Region (MMR / Coastal Maharashtra)",
        "min_lat": 18.80,
        "max_lat": 19.35,
        "min_lon": 72.75,
        "max_lon": 73.15,
        "elevation_range": "2m - 420m",
        "topography_type": "Coastal Estuary, Lowland Mangroves & Mithi River Basin",
        "default_action": "Issue Red Inundation Warning for low-lying coastal and Mithi basin corridors. Halt suburban transit in flood-prone junctions (Kurla, Dadar, Hindmata). BMC & NDRF flood rescue teams deployed.",
        "dem_base": 80.0,
        "dem_scale": 120.0,
        "dem_min": 2.0,
        "dem_max": 420.0,
    },
    "western_ghats": {
        "preset_id": "western_ghats",
        "label": "Western Ghats (Wayanad / Nilgiri Hills)",
        "min_lat": 11.35,
        "max_lat": 11.95,
        "min_lon": 75.80,
        "max_lon": 76.40,
        "elevation_range": "650m - 2,550m",
        "topography_type": "Steep Orographic Escarpments, Tea Slope Catchments & River Chaliyar Basin",
        "default_action": "Critical Debris Flow & Landslide Alert for steep tea plantation slopes in Wayanad/Meppadi. Immediate evacuation of vulnerable hillside habitations to designated relief shelters.",
        "dem_base": 1400.0,
        "dem_scale": 650.0,
        "dem_min": 650.0,
        "dem_max": 2550.0,
    },
    "northeast": {
        "preset_id": "northeast",
        "label": "North-East (Sohra / Khasi Hills)",
        "min_lat": 25.10,
        "max_lat": 25.60,
        "min_lon": 91.55,
        "max_lon": 92.05,
        "elevation_range": "200m - 1,950m",
        "topography_type": "Meghalaya Tableland Escarpment & Extreme Orographic Funnel",
        "default_action": "Flash Flood Warning for southern slopes and gorge river valleys draining to Sylhet basin. Suspend mining activities and road transit along cliff roads.",
        "dem_base": 1100.0,
        "dem_scale": 550.0,
        "dem_min": 200.0,
        "dem_max": 1950.0,
    },
}


class BoundingBoxModel(BaseModel):
    min_lat: float = Field(..., ge=6.0, le=37.0, description="Southern latitude boundary (6.0°N to 37.0°N)")
    max_lat: float = Field(..., ge=6.0, le=37.0, description="Northern latitude boundary (6.0°N to 37.0°N)")
    min_lon: float = Field(..., ge=68.0, le=98.0, description="Western longitude boundary (68.0°E to 98.0°E)")
    max_lon: float = Field(..., ge=68.0, le=98.0, description="Eastern longitude boundary (68.0°E to 98.0°E)")


@app.on_event("startup")
def load_ai_engine():
    global model, explainer
    print(" Initializing PyTorch AI Engine for Pan-India Nowcasting...")
    model = MultiTaskWeatherNowcaster(in_channels=5, time_steps=4, hidden_dim=32).to(DEVICE)

    weights_path = "weights/nowcaster_baseline.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f" Loaded trained weights from {weights_path}")
    else:
        print(" Using initialized model weights (weights file not found).")

    model.eval()
    explainer = WeatherExplainer(model)
    print(" AI Engine ready for dynamic pan-India inference.")


@app.get("/")
def root():
    return {
        "service": "GeoAlert Nowcaster AI Engine",
        "status": "online",
        "coverage": "Pan-India Dynamic Bounding Boxes (6.0-37.0°N, 68.0-98.0°E)",
        "available_presets": list(REGION_PRESETS.keys()),
        "lead_time_window": "2-6 Hours",
        "docs_url": "/docs"
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/v1/regions/presets")
def get_region_presets():
    """Returns available hackathon region presets and their geographic extents."""
    return {
        "total_presets": len(REGION_PRESETS),
        "presets": REGION_PRESETS
    }


def synthesize_regional_topography(dem_base: float, dem_scale: float, dem_min: float, dem_max: float, grid_size: int = GRID_SIZE):
    """
    Synthesizes realistic regional terrain topography for the target bounding box.
    """
    x = np.linspace(0, 4 * np.pi, grid_size)
    y = np.linspace(0, 4 * np.pi, grid_size)
    xx, yy = np.meshgrid(x, y)

    ridge = np.sin(xx * 0.9 + yy * 0.6) * (dem_scale * 0.7)
    slope = np.cos(xx * 1.3 - yy * 0.7) * (dem_scale * 0.4)
    valley = -np.exp(-((xx - 6.0) ** 2 + (yy - 6.0) ** 2) / 4.0) * (dem_scale * 0.8)

    dem = dem_base + ridge + slope + valley
    dem = np.clip(dem, dem_min, dem_max).astype(np.float32)
    return dem


def resolve_bounding_box(
    region: Optional[str] = None,
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    """
    Validates and resolves region preset or custom bounding coordinates.
    Returns: (lat_min, lat_max, lon_min, lon_max), region_meta
    """
    
    if any(param is not None for param in [min_lat, max_lat, min_lon, max_lon]):
        if None in [min_lat, max_lat, min_lon, max_lon]:
            raise HTTPException(
                status_code=422,
                detail="All 4 coordinates (min_lat, max_lat, min_lon, max_lon) must be provided for custom bounding box."
            )
        
        
        if not (INDIA_LAT_MIN <= min_lat < max_lat <= INDIA_LAT_MAX):
            raise HTTPException(
                status_code=422,
                detail=f"Latitude range [{min_lat}, {max_lat}] invalid. Must satisfy 6.0 <= min_lat < max_lat <= 37.0°N."
            )
        if not (INDIA_LON_MIN <= min_lon < max_lon <= INDIA_LON_MAX):
            raise HTTPException(
                status_code=422,
                detail=f"Longitude range [{min_lon}, {max_lon}] invalid. Must satisfy 68.0 <= min_lon < max_lon <= 98.0°E."
            )

        bounds = (min_lat, max_lat, min_lon, max_lon)
        meta = {
            "preset_id": "custom",
            "label": f"Custom Pan-India Grid ({min_lat:.2f}–{max_lat:.2f}°N, {min_lon:.2f}–{max_lon:.2f}°E)",
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "elevation_range": "100m - 3,200m",
            "topography_type": "Custom Regional Basin Topography",
            "default_action": "High-risk convective signatures detected. Implement local emergency response and monitor radar telemetry.",
            "dem_base": 1200.0,
            "dem_scale": 800.0,
            "dem_min": 50.0,
            "dem_max": 3800.0,
        }
        return bounds, meta


    preset_key = (region or "uttarakhand").strip().lower()
    if preset_key not in REGION_PRESETS:
        available = ", ".join(list(REGION_PRESETS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown region preset '{preset_key}'. Available presets: {available}"
        )

    preset_data = REGION_PRESETS[preset_key]
    bounds = (preset_data["min_lat"], preset_data["max_lat"], preset_data["min_lon"], preset_data["max_lon"])
    return bounds, preset_data


@app.get("/api/v1/predict/nowcast")
def get_nowcast_prediction(
    lead_time_hours: float = Query(3.0, ge=1.0, le=12.0, description="Nowcast lead time in hours (2 to 6 hours)"),
    region: Optional[str] = Query(None, description="Region preset ('uttarakhand', 'mumbai', 'western_ghats', 'northeast')"),
    min_lat: Optional[float] = Query(None, ge=6.0, le=37.0, description="Custom south latitude boundary (6.0 to 37.0)"),
    max_lat: Optional[float] = Query(None, ge=6.0, le=37.0, description="Custom north latitude boundary (6.0 to 37.0)"),
    min_lon: Optional[float] = Query(None, ge=68.0, le=98.0, description="Custom west longitude boundary (68.0 to 98.0)"),
    max_lon: Optional[float] = Query(None, ge=68.0, le=98.0, description="Custom east longitude boundary (68.0 to 98.0)"),
):
    """
    Main Pan-India Nowcast Endpoint:
    1. Validates and dynamically scales to requested regional bounding box or preset.
    2. Ingests simulated satellite & atmospheric tensors for the target geography.
    3. Runs PyTorch ConvLSTM U-Net multi-task inference.
    4. Computes Captum Integrated Gradients XAI feature importance breakdown.
    5. Triages alert level (RED / ORANGE / GREEN).
    6. Returns dynamically scaled vector GeoJSON polygons aligned with the target bounding box coordinates.
    """
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="AI Model engine is not yet initialized.")

    
    bounds, region_meta = resolve_bounding_box(
        region=region,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon
    )
    lat_min, lat_max, lon_min, lon_max = bounds
    lat_step = (lat_max - lat_min) / GRID_SIZE
    lon_step = (lon_max - lon_min) / GRID_SIZE

    
    dem = synthesize_regional_topography(
        dem_base=region_meta["dem_base"],
        dem_scale=region_meta["dem_scale"],
        dem_min=region_meta["dem_min"],
        dem_max=region_meta["dem_max"],
        grid_size=GRID_SIZE
    )

    from mock_data_generator import generate_atmospheric_fields
    seed = int(abs(lat_min * 100 + lon_min * 10) + lead_time_hours * 5) % 10000
    atm_fields = generate_atmospheric_fields(dem, seed=seed)
    if len(atm_fields) == 5:
        iwv, cape, cin, _, ctt_series = atm_fields
    else:
        iwv, cape, cin, ctt_series = atm_fields

    
    if lead_time_hours <= 2.5:
        intensity = 0.90
    elif lead_time_hours <= 4.5:
        intensity = 1.18
    else:
        intensity = 1.02

    iwv = np.clip(iwv * (0.95 + intensity * 0.1), 15.0, 72.0)
    cape = np.clip(cape * intensity, 200.0, 3950.0)

    # Build [5, 4, 64, 64] PyTorch tensor
    x_tensor = build_spatiotemporal_tensor(iwv, cape, cin, ctt_series, dem)
    x_batch = x_tensor.unsqueeze(0).to(DEVICE)

    # 3. Model Inference
    with torch.no_grad():
        preds = model(x_batch)

    ts_map = preds["thunderstorm"][0, 0].cpu().numpy()
    cb_map = preds["cloudburst"][0, 0].cpu().numpy()
    ff_map = preds["flash_flood"][0, 0].cpu().numpy()

    # Apply lead-time spatial expansion
    if lead_time_hours >= 3.0:
        boost = min(0.20, (lead_time_hours - 2.0) * 0.08)
        cb_map = np.clip(cb_map + boost * (cb_map > 0.35), 0.0, 1.0)
        ff_map = np.clip(ff_map + (boost * 1.2) * (ff_map > 0.30), 0.0, 1.0)

    hazard_maps = {
        "thunderstorm": ts_map,
        "cloudburst": cb_map,
        "flash_flood": ff_map
    }

    # 4. Risk Metrics & Alert Categorization
    peak_ts = float(np.max(ts_map))
    peak_cb = float(np.max(cb_map))
    peak_ff = float(np.max(ff_map))
    overall_peak_risk = max(peak_ts, peak_cb, peak_ff)

    if overall_peak_risk >= 0.68:
        alert_category = "RED"
        alert_title = "CRITICAL EVACUATION WARNING"
        color_code = "#ef4444"
        action_recommendation = region_meta["default_action"]
    elif overall_peak_risk >= 0.45:
        alert_category = "ORANGE"
        alert_title = "STANDBY ALERT"
        color_code = "#f97316"
        action_recommendation = (
            f"Convective instability intensifying over {region_meta['label']}. "
            "Emergency personnel placed on standby. Restrict movement across vulnerable low-lying infrastructure."
        )
    else:
        alert_category = "GREEN"
        alert_title = "ROUTINE MONITORING"
        color_code = "#22c55e"
        action_recommendation = (
            f"Atmospheric parameters within nominal thresholds across {region_meta['label']}. Radar telemetry active."
        )

    
    xai_results = explainer.attribute(x_batch, target_hazard="cloudburst", n_steps=10)
    channel_breakdown = xai_results["channel_breakdown"]
    dominant_factor = max(channel_breakdown.items(), key=lambda item: item[1]["percentage"])

    xai_narrative = (
        f"Primary convective driver over {region_meta['label']} is {dominant_factor[1]['full_name']} "
        f"contributing {dominant_factor[1]['percentage']:.1f}% of total risk. "
        f"Topographic profile ({region_meta['topography_type']}) accelerates convective funneling."
    )


    geojson_data = create_nowcast_feature_collection(
        hazard_maps=hazard_maps,
        bounds=bounds,
        region_label=region_meta["label"],
        dem_array=dem,
        threshold=0.40,
        action_override=action_recommendation
    )

    response = {
        "region_info": {
            "preset": region_meta.get("preset_id", "custom"),
            "label": region_meta["label"],
            "bounding_box": {
                "min_lat": round(lat_min, 4),
                "max_lat": round(lat_max, 4),
                "min_lon": round(lon_min, 4),
                "max_lon": round(lon_max, 4)
            },
            "center": {
                "lat": round((lat_min + lat_max) / 2.0, 4),
                "lon": round((lon_min + lon_max) / 2.0, 4)
            },
            "grid_resolution": f"{GRID_SIZE}x{GRID_SIZE}",
            "lat_step_deg": round(lat_step, 6),
            "lon_step_deg": round(lon_step, 6),
            "topography_type": region_meta["topography_type"],
            "elevation_range": region_meta["elevation_range"]
        },
        "summary": {
            "region": region_meta["label"],
            "alert_category": alert_category,
            "alert_title": alert_title,
            "color_code": color_code,
            "lead_time_hours": lead_time_hours,
            "overall_peak_risk": round(overall_peak_risk, 3),
            "recommended_action": action_recommendation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "hazard_metrics": {
            "thunderstorm": {
                "peak_probability": round(peak_ts, 3),
                "risk_status": "CRITICAL" if peak_ts >= 0.70 else "HIGH" if peak_ts >= 0.50 else "MODERATE"
            },
            "cloudburst": {
                "peak_probability": round(peak_cb, 3),
                "risk_status": "CRITICAL" if peak_cb >= 0.70 else "HIGH" if peak_cb >= 0.50 else "MODERATE"
            },
            "flash_flood": {
                "peak_probability": round(peak_ff, 3),
                "risk_status": "CRITICAL" if peak_ff >= 0.70 else "HIGH" if peak_ff >= 0.50 else "MODERATE"
            }
        },
        "xai_attribution": {
            "target_hazard": "Cloudburst & Flash Flood",
            "dominant_driver": dominant_factor[0],
            "narrative": xai_narrative,
            "breakdown": channel_breakdown
        },
        "geojson": geojson_data
    }

    return response
