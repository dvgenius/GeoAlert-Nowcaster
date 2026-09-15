"""
Phase 1 Verification Tests: Geospatial Raster Data Verification
Checks that NetCDF4 and GeoTIFF files in data/ conform to expected grid, bands, coordinates, and value ranges.
"""

import os
import rasterio
import netCDF4 as nc
import numpy as np


def test_geotiff_multiband():
    path = "data/chamoli_weather_multiband.tif"
    assert os.path.exists(path), f"{path} does not exist"

    with rasterio.open(path) as src:
        assert src.count == 5, f"Expected 5 bands, got {src.count}"
        assert src.width == 64 and src.height == 64, f"Expected 64x64, got {src.width}x{src.height}"
        assert src.crs.to_string() == "EPSG:4326", f"Expected EPSG:4326, got {src.crs}"
        
        # Read all bands
        bands = src.read()
        assert not np.isnan(bands).any(), "Found NaNs in GeoTIFF bands"
        assert not np.isinf(bands).any(), "Found Infs in GeoTIFF bands"
        
        # Band descriptions
        expected_bands = ["IWV", "CAPE", "CIN", "CTT_T0", "DEM"]
        for idx, name in enumerate(expected_bands, start=1):
            assert src.descriptions[idx - 1] == name, f"Band {idx} name mismatch: {src.descriptions[idx-1]}"

    print(" test_geotiff_multiband PASSED")


def test_geotiff_dem():
    path = "data/chamoli_dem.tif"
    assert os.path.exists(path), f"{path} does not exist"

    with rasterio.open(path) as src:
        assert src.count == 1, f"Expected 1 band, got {src.count}"
        assert src.width == 64 and src.height == 64
        dem = src.read(1)
        assert dem.min() >= 1200.0, f"DEM min elevation {dem.min()} too low"
        assert dem.max() <= 5500.0, f"DEM max elevation {dem.max()} too high"

    print(" test_geotiff_dem PASSED")


def test_netcdf_structure():
    path = "data/chamoli_nowcast_input.nc"
    assert os.path.exists(path), f"{path} does not exist"

    with nc.Dataset(path, "r") as ds:
        # Check dimensions
        assert "time" in ds.dimensions and len(ds.dimensions["time"]) == 4
        assert "lat" in ds.dimensions and len(ds.dimensions["lat"]) == 64
        assert "lon" in ds.dimensions and len(ds.dimensions["lon"]) == 64

        # Check variables
        expected_vars = ["dem", "iwv", "cape", "cin", "ctt"]
        for var in expected_vars:
            assert var in ds.variables, f"Variable {var} missing in NetCDF"
            data = ds.variables[var][:]
            assert not np.isnan(data).any(), f"Variable {var} contains NaNs"

        # Check CTT time-series cooling
        ctt = ds.variables["ctt"][:]
        # Minimum temperature at T0 should be cooler than T-3 due to convective cloud development
        assert ctt[0].min() > ctt[-1].min(), "Expected convective cooling over time"

    print(" test_netcdf_structure PASSED")


if __name__ == "__main__":
    print("--- Running Phase 1 Data Verification Tests ---")
    test_geotiff_multiband()
    test_geotiff_dem()
    test_netcdf_structure()
    print(" ALL PHASE 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
