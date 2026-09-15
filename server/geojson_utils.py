"""
GeoJSON Utilities for Raster to Vector Conversion
Converts 64x64 risk probability matrices into compliant GeoJSON FeatureCollections
with geospatial coordinates, hazard metadata, and emergency protocols.
Supports dynamic pan-India regional bounding boxes.
"""

import numpy as np
from shapely.geometry import box, mapping
from typing import Tuple, Dict, Any, List, Optional

GRID_SIZE = 64


def get_cell_bounds(row: int, col: int, bounds: Tuple[float, float, float, float], h: int = GRID_SIZE, w: int = GRID_SIZE):
    """
    Returns [lon_min, lat_min, lon_max, lat_max] for grid cell (row, col)
    given bounding box bounds = (lat_min, lat_max, lon_min, lon_max).
    row 0 is southern-most (lat_min).
    """
    lat_min_bound, lat_max_bound, lon_min_bound, lon_max_bound = bounds
    d_lat = (lat_max_bound - lat_min_bound) / h
    d_lon = (lon_max_bound - lon_min_bound) / w

    lat_min = lat_min_bound + row * d_lat
    lat_max = lat_min + d_lat
    lon_min = lon_min_bound + col * d_lon
    lon_max = lon_min + d_lon

    return lon_min, lat_min, lon_max, lat_max


def raster_to_geojson_features(
    prob_map: np.ndarray,
    hazard_type: str,
    bounds: Tuple[float, float, float, float],
    threshold: float = 0.40,
    dem_array: Optional[np.ndarray] = None,
    action_override: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Scans risk probability map and returns a list of GeoJSON features for cells >= threshold.
    """
    features = []
    h, w = prob_map.shape

    # Find high-risk cells
    high_risk_coords = np.argwhere(prob_map >= threshold)

    for r, c in high_risk_coords:
        score = float(prob_map[r, c])
        lon_min, lat_min, lon_max, lat_max = get_cell_bounds(int(r), int(c), bounds, h, w)
        cell_geom = box(lon_min, lat_min, lon_max, lat_max)

        # Categorize cell severity
        if score >= 0.70:
            level = "CRITICAL"
            color = "#ef4444"  # Red
            fill_opacity = 0.65
            action = action_override or "IMMEDIATE EVACUATION: Move to high ground, avoid riverbanks and ravines."
        elif score >= 0.55:
            level = "HIGH"
            color = "#f97316"  # Orange
            fill_opacity = 0.50
            action = "STANDBY ALERT: Emergency rescue teams on notice. Prepare emergency supplies."
        else:
            level = "MODERATE"
            color = "#eab308"  # Yellow-Amber
            fill_opacity = 0.35
            action = "ADVISORY: Monitor live radar. Restrict movement across flood-prone bottlenecks."

        elevation = float(dem_array[r, c]) if dem_array is not None else 1500.0

        features.append({
            "type": "Feature",
            "geometry": mapping(cell_geom),
            "properties": {
                "hazard_type": hazard_type,
                "risk_score": round(score, 3),
                "risk_level": level,
                "color": color,
                "fill_opacity": fill_opacity,
                "center_lat": round((lat_min + lat_max) / 2.0, 5),
                "center_lon": round((lon_min + lon_max) / 2.0, 5),
                "elevation_m": round(elevation, 1),
                "recommended_action": action,
                "grid_row": int(r),
                "grid_col": int(c)
            }
        })

    return features


def create_nowcast_feature_collection(
    hazard_maps: Dict[str, np.ndarray],
    bounds: Tuple[float, float, float, float],
    region_label: str = "Chamoli, Uttarakhand",
    dem_array: Optional[np.ndarray] = None,
    threshold: float = 0.40,
    action_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Combines multi-task hazard maps into a single unified GeoJSON FeatureCollection
    dynamically scaled to any pan-India bounding box.
    bounds = (lat_min, lat_max, lon_min, lon_max)
    """
    lat_min, lat_max, lon_min, lon_max = bounds
    all_features = []

    # Priority ordering: Cloudburst & Flash Flood take front precedence
    for hazard_name in ["cloudburst", "flash_flood", "thunderstorm"]:
        if hazard_name in hazard_maps:
            feats = raster_to_geojson_features(
                hazard_maps[hazard_name],
                hazard_type=hazard_name,
                bounds=bounds,
                threshold=threshold,
                dem_array=dem_array,
                action_override=action_override
            )
            all_features.extend(feats)

    lat_step = (lat_max - lat_min) / GRID_SIZE
    lon_step = (lon_max - lon_min) / GRID_SIZE

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "region": region_label,
            "bounds": [round(lon_min, 4), round(lat_min, 4), round(lon_max, 4), round(lat_max, 4)],
            "lat_step_deg": round(lat_step, 6),
            "lon_step_deg": round(lon_step, 6),
            "total_alert_cells": len(all_features),
            "grid_resolution": f"{GRID_SIZE}x{GRID_SIZE}"
        },
        "features": all_features
    }

    return geojson
