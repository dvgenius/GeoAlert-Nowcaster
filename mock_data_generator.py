"""
Mock Geospatial Data Generator for Chamoli, Uttarakhand
Generates 64x64 synthetic raster channels (IWV, CAPE, CIN, CTT drop, DEM)
covering the Uttarakhand Chamoli region in the data/ directory.
Produces both multiband & individual GeoTIFF rasters and structured NetCDF4 files.
"""

import os
from datetime import datetime, timezone
import numpy as np

# Chamoli Region Bounding Box (Alaknanda & Rishi Ganga Valleys, Nanda Devi)
# Latitude: 30.15° N to 30.75° N, Longitude: 79.25° E to 79.85° E
LAT_MIN, LAT_MAX = 30.15, 30.75
LON_MIN, LON_MAX = 79.25, 79.85
GRID_SIZE = 64
TIME_STEPS = 4  # T-45m, T-30m, T-15m, T0


def generate_chamoli_topography(grid_size=GRID_SIZE):
    """
    Generates realistic 64x64 Himalayan topography for Chamoli district:
    High peaks (Nanda Devi, Trishul ridgelines ~4500-5200m)
    and deep valleys (Alaknanda & Rishi Ganga gorges ~1300-1800m).
    """
    x = np.linspace(0, 4 * np.pi, grid_size)
    y = np.linspace(0, 4 * np.pi, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Base elevation with mountain ridges and river valleys
    ridge_1 = np.sin(xx * 0.8 + yy * 0.5) * 1200
    ridge_2 = np.cos(xx * 1.2 - yy * 0.8) * 800
    valley = -np.exp(-((xx - 6.0) ** 2 + (yy - 6.0) ** 2) / 4.0) * 1500

    # Chamoli average elevation ~2800m, ranging from ~1300m in valleys to ~4900m peaks
    dem = 2900 + ridge_1 + ridge_2 + valley
    dem = np.clip(dem, 1250.0, 5200.0).astype(np.float32)
    return dem


def generate_atmospheric_fields(dem, grid_size=GRID_SIZE, seed=42):
    """
    Simulates pre-convective atmospheric environment for Chamoli:
    - IWV: Integrated Water Vapor (kg/m^2)
    - CAPE: Convective Available Potential Energy (J/kg)
    - CIN: Convective Inhibition (J/kg)
    - CTT drop: Cloud Top Temperature drop rate / cooling rate (-dCTT/dt in K/hr)
    - CTT series: 4-step temporal sequence (Kelvin) for time-series modeling
    """
    np.random.seed(seed)
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Convective epicenter near valley-mountain convergence (e.g. Joshimath / Tapovan area)
    center_x, center_y = 0.6, 0.4
    dist_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    plume = np.exp(-dist_sq / 0.8)

    # 1. Integrated Water Vapor (IWV): High moisture pool in valley funnel (35 - 65 kg/m^2)
    # Orographic moisture pooling: lower elevations hold more column water vapor
    orographic_moisture = (5200.0 - dem) / 4000.0 * 20.0
    iwv = 28.0 + orographic_moisture + plume * 22.0 + np.random.normal(0, 1.2, (grid_size, grid_size))
    iwv = np.clip(iwv, 18.0, 68.0).astype(np.float32)

    # 2. CAPE: High atmospheric instability (1200 - 3400 J/kg in convective zone)
    cape = 1100.0 + plume * 2100.0 - (dem - 2500.0) * 0.15 + np.random.normal(0, 60.0, (grid_size, grid_size))
    cape = np.clip(cape, 300.0, 3800.0).astype(np.float32)

    # 3. CIN: Convective Inhibition cap breaking near initiation site (-120 to -5 J/kg)
    cin = -90.0 + plume * 85.0 + np.random.normal(0, 5.0, (grid_size, grid_size))
    cin = np.clip(cin, -160.0, 0.0).astype(np.float32)

    # 4. CTT: Cloud Top Temperature time series (4 steps, 15-min intervals)
    # Rapid cloud top cooling down to 205K (-68°C) indicating deep convective updraft
    ctt_series = np.zeros((TIME_STEPS, grid_size, grid_size), dtype=np.float32)
    base_ctt = 285.0 - (dem / 1000.0) * 6.5  # Standard environmental lapse rate

    intensities = [0.15, 0.45, 0.75, 1.0]  # Progressive cloud vertical growth
    for t_idx, intensity in enumerate(intensities):
        cooling = intensity * (plume * 78.0)
        cell_noise = np.random.normal(0, 1.0, (grid_size, grid_size))
        ctt = base_ctt - cooling + cell_noise
        ctt_series[t_idx] = np.clip(ctt, 203.0, 298.0).astype(np.float32)

    # 5. CTT Drop Rate (-dCTT/dt in K/hr) over the final 15-minute interval (dt = 0.25 hr)
    # Rapid drop indicates severe convective overshoot and cloudburst risk
    delta_t_hours = 0.25
    ctt_drop = (ctt_series[-2] - ctt_series[-1]) / delta_t_hours
    ctt_drop = np.clip(ctt_drop, -5.0, 85.0).astype(np.float32)

    return iwv, cape, cin, ctt_drop, ctt_series


def save_geotiff(output_path, data_arrays, band_names, lat_bounds=(LAT_MIN, LAT_MAX), lon_bounds=(LON_MIN, LON_MAX)):
    """
    Saves multiband or single-band raster using rasterio with EPSG:4326 transform.
    """
    import rasterio
    from rasterio.transform import from_bounds

    num_bands = len(data_arrays)
    height, width = data_arrays[0].shape
    transform = from_bounds(lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1], width, height)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": num_bands,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw"
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, (arr, name) in enumerate(zip(data_arrays, band_names), start=1):
            dst.write(arr.astype(np.float32), idx)
            dst.set_band_description(idx, name)

    print(f" Saved GeoTIFF: {output_path} ({num_bands} band{'s' if num_bands > 1 else ''}: {', '.join(band_names)})")


def save_netcdf(output_path, dem, iwv, cape, cin, ctt_drop, ctt_series, lat_bounds=(LAT_MIN, LAT_MAX), lon_bounds=(LON_MIN, LON_MAX)):
    """
    Saves structured NetCDF4 dataset with spatial and temporal dimensions.
    """
    import netCDF4 as nc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lats = np.linspace(lat_bounds[0], lat_bounds[1], GRID_SIZE)
    lons = np.linspace(lon_bounds[0], lon_bounds[1], GRID_SIZE)
    times = np.array([0, 15, 30, 45], dtype=np.int32)  # minutes

    with nc.Dataset(output_path, "w", format="NETCDF4") as ds:
        # Create Dimensions
        ds.createDimension("time", TIME_STEPS)
        ds.createDimension("lat", GRID_SIZE)
        ds.createDimension("lon", GRID_SIZE)

        # Coordinate Variables
        var_time = ds.createVariable("time", "i4", ("time",))
        var_time.units = "minutes since initiation"
        var_time[:] = times

        var_lat = ds.createVariable("lat", "f4", ("lat",))
        var_lat.units = "degrees_north"
        var_lat.standard_name = "latitude"
        var_lat[:] = lats

        var_lon = ds.createVariable("lon", "f4", ("lon",))
        var_lon.units = "degrees_east"
        var_lon.standard_name = "longitude"
        var_lon[:] = lons

        # Data Variables (5 requested channels + CTT series)
        var_dem = ds.createVariable("dem", "f4", ("lat", "lon"), zlib=True)
        var_dem.units = "meters"
        var_dem.long_name = "Digital Elevation Model (Himalayan Rugged Topography)"
        var_dem[:, :] = dem

        var_iwv = ds.createVariable("iwv", "f4", ("lat", "lon"), zlib=True)
        var_iwv.units = "kg m-2"
        var_iwv.long_name = "Integrated Water Vapor"
        var_iwv[:, :] = iwv

        var_cape = ds.createVariable("cape", "f4", ("lat", "lon"), zlib=True)
        var_cape.units = "J kg-1"
        var_cape.long_name = "Convective Available Potential Energy"
        var_cape[:, :] = cape

        var_cin = ds.createVariable("cin", "f4", ("lat", "lon"), zlib=True)
        var_cin.units = "J kg-1"
        var_cin.long_name = "Convective Inhibition"
        var_cin[:, :] = cin

        var_ctt_drop = ds.createVariable("ctt_drop", "f4", ("lat", "lon"), zlib=True)
        var_ctt_drop.units = "K hr-1"
        var_ctt_drop.long_name = "Cloud Top Temperature Drop Rate (-dCTT/dt)"
        var_ctt_drop[:, :] = ctt_drop

        var_ctt = ds.createVariable("ctt", "f4", ("time", "lat", "lon"), zlib=True)
        var_ctt.units = "Kelvin"
        var_ctt.long_name = "Cloud Top Temperature Time Series (INSAT-3D/3DR Proxy)"
        var_ctt[:, :, :] = ctt_series

        # Global Attributes
        ds.title = "Synthetic Meteorological & Terrain Raster for Chamoli Early Warning Nowcasting"
        ds.region = "Chamoli District, Uttarakhand, India"
        ds.institution = "IIT BHU Hackathon AI Early Warning System"
        ds.spatial_resolution = "~1 km (64x64 grid)"
        ds.bbox_lat = f"{lat_bounds[0]} to {lat_bounds[1]} N"
        ds.bbox_lon = f"{lon_bounds[0]} to {lon_bounds[1]} E"
        ds.created_at = datetime.now(timezone.utc).isoformat()

    print(f" Saved NetCDF4: {output_path}")


def generate_all_mock_data(data_dir="data"):
    """
    Orchestrates generation of 64x64 synthetic raster channels:
    (IWV, CAPE, CIN, CTT drop, DEM) covering the Chamoli, Uttarakhand region.
    """
    os.makedirs(data_dir, exist_ok=True)
    print("=" * 70)
    print("Generating 64x64 synthetic raster channels for Chamoli, Uttarakhand...")
    print(f"Spatial Coverage: Lat [{LAT_MIN}°N - {LAT_MAX}°N], Lon [{LON_MIN}°E - {LON_MAX}°E]")
    print("=" * 70)

    dem = generate_chamoli_topography(grid_size=GRID_SIZE)
    iwv, cape, cin, ctt_drop, ctt_series = generate_atmospheric_fields(dem, grid_size=GRID_SIZE)

    # 1. Save Multiband GeoTIFF containing all 5 channels (IWV, CAPE, CIN, CTT drop, DEM)
    multiband_path = os.path.join(data_dir, "chamoli_weather_multiband.tif")
    bands = [iwv, cape, cin, ctt_drop, dem]
    band_names = ["IWV", "CAPE", "CIN", "CTT_DROP", "DEM"]
    save_geotiff(multiband_path, bands, band_names)

    # 2. Save Individual Single-Band GeoTIFFs for each channel
    individual_rasters = {
        "chamoli_iwv.tif": (iwv, "IWV_kg_m2"),
        "chamoli_cape.tif": (cape, "CAPE_J_kg"),
        "chamoli_cin.tif": (cin, "CIN_J_kg"),
        "chamoli_ctt_drop.tif": (ctt_drop, "CTT_Drop_K_hr"),
        "chamoli_dem.tif": (dem, "DEM_Elevation_m")
    }
    for filename, (arr, name) in individual_rasters.items():
        save_geotiff(os.path.join(data_dir, filename), [arr], [name])

    # 3. Save Comprehensive NetCDF4 dataset (coordinates + all 5 channels + CTT series)
    nc_path = os.path.join(data_dir, "chamoli_nowcast_input.nc")
    save_netcdf(nc_path, dem, iwv, cape, cin, ctt_drop, ctt_series)

    return {
        "dem": dem,
        "iwv": iwv,
        "cape": cape,
        "cin": cin,
        "ctt_drop": ctt_drop,
        "ctt_series": ctt_series,
        "multiband_path": multiband_path,
        "nc_path": nc_path,
    }


if __name__ == "__main__":
    result = generate_all_mock_data()
    print("\n" + "=" * 70)
    print(" Synthetic Raster Channels Successfully Verified:")
    print(f"  [Channel 1] IWV:      min = {result['iwv'].min():.2f} kg/m², max = {result['iwv'].max():.2f} kg/m²")
    print(f"  [Channel 2] CAPE:     min = {result['cape'].min():.2f} J/kg, max = {result['cape'].max():.2f} J/kg")
    print(f"  [Channel 3] CIN:      min = {result['cin'].min():.2f} J/kg, max = {result['cin'].max():.2f} J/kg")
    print(f"  [Channel 4] CTT Drop: min = {result['ctt_drop'].min():.2f} K/hr, max = {result['ctt_drop'].max():.2f} K/hr")
    print(f"  [Channel 5] DEM:      min = {result['dem'].min():.1f} m, max = {result['dem'].max():.1f} m")
    print(f"  [Raster Grid]         Shape = {GRID_SIZE}x{GRID_SIZE}, CRS = EPSG:4326")
    print("=" * 70)
