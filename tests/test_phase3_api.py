"""
Phase 3 Verification Tests: FastAPI Endpoint & GeoJSON Streaming Test
Tests the nowcast prediction endpoint, XAI attribution payload, and GeoJSON validity.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from server.app import app, load_ai_engine


def test_fastapi_endpoints():
    print("--- Initializing AI Engine for TestClient ---")
    load_ai_engine()
    client = TestClient(app)

    # 1. Test Root
    res_root = client.get("/")
    assert res_root.status_code == 200, f"Root failed: {res_root.status_code}"
    print(" GET / PASSED")

    # 2. Test Health
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True
    print(" GET /api/v1/health PASSED")

    # 3. Test Nowcast Prediction (Lead Time = 3h)
    res_nowcast = client.get("/api/v1/predict/nowcast?lead_time_hours=3")
    assert res_nowcast.status_code == 200, f"Nowcast failed: {res_nowcast.text}"
    data = res_nowcast.json()

    # Check Top-Level Keys
    assert "summary" in data
    assert "hazard_metrics" in data
    assert "xai_attribution" in data
    assert "geojson" in data

    summary = data["summary"]
    assert summary["alert_category"] in ["RED", "ORANGE", "GREEN"]
    assert 0.0 <= summary["overall_peak_risk"] <= 1.0
    print(f" Summary: {summary['alert_category']} ({summary['alert_title']}) - Risk: {summary['overall_peak_risk']}")

    # Check Hazard Metrics
    metrics = data["hazard_metrics"]
    for hazard in ["thunderstorm", "cloudburst", "flash_flood"]:
        assert hazard in metrics
        assert 0.0 <= metrics[hazard]["peak_probability"] <= 1.0
    print(f" Hazard Metrics: TS={metrics['thunderstorm']['peak_probability']}, CB={metrics['cloudburst']['peak_probability']}, FF={metrics['flash_flood']['peak_probability']}")

    # Check XAI Attribution
    xai = data["xai_attribution"]
    assert "breakdown" in xai
    for channel in ["IWV", "CAPE", "CIN", "CTT_DROP", "DEM"]:
        assert channel in xai["breakdown"]
        assert "percentage" in xai["breakdown"][channel]
    print(f" XAI Attribution: Dominant driver = {xai['dominant_driver']}")

    # Check GeoJSON
    geojson = data["geojson"]
    assert geojson["type"] == "FeatureCollection"
    assert "features" in geojson
    assert len(geojson["features"]) > 0, "Expected at least 1 high-risk GeoJSON feature"
    
    first_feat = geojson["features"][0]
    assert first_feat["type"] == "Feature"
    assert first_feat["geometry"]["type"] == "Polygon"
    assert "properties" in first_feat
    props = first_feat["properties"]
    assert props["risk_score"] >= 0.40
    assert props["hazard_type"] in ["cloudburst", "flash_flood", "thunderstorm"]
    assert "recommended_action" in props
    print(f" GeoJSON streaming validated: {len(geojson['features'])} vector polygons generated.")


if __name__ == "__main__":
    test_fastapi_endpoints()
    print("\n ALL PHASE 3 API UNIT TESTS PASSED SUCCESSFULLY!")
