"""
Fire Data Loader
================

A tool for downloading and processing wildfire-related geospatial data from multiple sources:
- FEDS25MTBS: Fire perimeter data
- Fire Radiative Power (FRP): From NASA FIRMS (2024+) or FEDS25MTBS firepix (pre-2024)
- USGS 3DEP: Elevation data
- LANDFIRE: Canopy bulk density and canopy cover
- HRRR: Weather data (humidity, wind)
- Global Building Atlas: Building heights
- ESA WorldCover: Land cover classification
- Tree Canopy: Leaf Area Index (LAI)
- NAIP/Sentinel-2: Satellite imagery (RGB)
- Hillshade: Terrain visualization derived from elevation

Prerequisites:
    Set your Google Earth Engine project before running:
    $ earthengine set_project your-google-cloud-project-id

Usage:
    Single event:
        python main.py <event_id> [options]
    
    Batch processing (from file):
        python main.py --batch events.txt [options]
    
    Batch processing (comma-separated):
        python main.py --batch CA123,CA456,CA789 [options]

Examples:
    python main.py CA3859812261820171009 -v
    python main.py --batch event_ids.txt --workers 4
    python main.py --batch CA123,CA456 -r 10 -o results/
"""

# Standard library imports
import argparse
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from typing import Literal, Callable, Optional, Any

# Third-party imports
import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from affine import Affine
from herbie import Herbie
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box, MultiPolygon
from tqdm import tqdm

# Local imports
from schemas import FireInfo, TaskInfo, DataWithMetadata, ProcessingArgs

# Configure xarray options
xr.set_options(use_new_combine_kwarg_defaults=True)

# Module logger
log = logging.getLogger(__name__)

# Default Herbie cache directory
DEFAULT_HERBIE_CACHE_DIR = "./datasets/herbie"

# Default paths for FRP data
DEFAULT_FIRMS_DIR = "datasets/FIRMS"
DEFAULT_FEDS25MTBS_DIR = "datasets/FEDS25MTBS"
DEFAULT_FIREPIX_DIR = f"{DEFAULT_FEDS25MTBS_DIR}/firepix"
DEFAULT_FIRELIST_PATH = f"{DEFAULT_FEDS25MTBS_DIR}/fireslist2012-2023.csv"

# FIRMS data files (VIIRS)
FIRMS_FILES = [
    "fire_archive_SV-C2_708942.csv",  # VIIRS archive 2025
    "fire_nrt_SV-C2_708942.csv",       # VIIRS near-real-time 2025
    "fire_archive_SV-C2_713904.csv" ,  # VIIRS archive 2024
]

# Columns to load from FIRMS CSV (reduces memory and speeds up loading)
FIRMS_USECOLS = ['latitude', 'longitude', 'acq_date', 'acq_time', 'frp', 'confidence', 'daynight']

# Module-level caches for FRP data (avoid reloading large CSV files)
_firms_cache: dict[str, pd.DataFrame] = {}
_firepix_cache: dict[str, pd.DataFrame] = {}


def clear_frp_cache() -> None:
    """Clear cached FIRMS and Firepix data to free memory.
    
    Call this when done processing fires or when memory is needed.
    """
    _firms_cache.clear()
    _firepix_cache.clear()
    log.info("Cleared FIRMS and Firepix cache")


# =============================================================================
# Fire Information Functions
# =============================================================================

def get_fire_info(
    event_id: str,
    firelist_path: str = DEFAULT_FIRELIST_PATH
) -> FireInfo:
    """Load fire event information from the FEDS25MTBS dataset.
    
    Args:
        event_id: Unique identifier for the fire event.
        firelist_path: Path to the CSV file containing fire metadata.
    
    Returns:
        FireInfo object containing event details.
    
    Raises:
        AssertionError: If the file doesn't exist or event_id not found.
    """
    assert os.path.exists(
        firelist_path), f"Error: File not found at {firelist_path}"

    df = pd.read_csv(firelist_path)

    df.columns = df.columns.str.strip()

    df['tst'] = pd.to_datetime(df['tst'], errors='coerce')
    df['ted'] = pd.to_datetime(df['ted'], errors='coerce')

    fire_row = df[df['Event_ID'] == event_id]

    assert not fire_row.empty, f"Error: Event_ID {event_id} not found in the dataset"

    row = fire_row.iloc[0]

    return FireInfo(
        event_id=row['Event_ID'],
        name=row['Incid_Name'],
        year=int(row['Year']),
        acres_burned=int(row['BurnBndAc']),
        t_start=row['tst'].to_pydatetime(),
        t_end=row['ted'].to_pydatetime(),
        bounds=(row['lon0'], row['lat0'], row['lon1'], row['lat1']),
    )


def geometries_are_equal(geom1, geom2, threshold: float = 1e-4) -> bool:
    """Check if two geometries are equal within a tolerance.
    
    Uses symmetric difference to handle floating-point precision issues where
    geometries are identical but .equals() returns False due to tiny coordinate
    differences (machine epsilon).
    
    Args:
        geom1: First geometry.
        geom2: Second geometry.
        threshold: Maximum symmetric difference ratio to consider equal.
                   Default 1e-8 filters floating-point noise while keeping real changes.
    
    Returns:
        True if geometries are equal within the threshold.
    """
    if geom1 is None or geom2 is None:
        return geom1 is None and geom2 is None
    
    # Fast path: exact equality
    if geom1.equals(geom2):
        return True
    
    # Check using symmetric difference ratio
    try:
        sym_diff = geom1.symmetric_difference(geom2)
        total_area = max(geom1.area, geom2.area, 1e-10)
        return sym_diff.area / total_area < threshold
    except Exception:
        return False


def get_fire_progression_dates(
    event_id: str,
    year: int,
    base_dir: str = DEFAULT_FEDS25MTBS_DIR
) -> tuple[datetime, datetime]:
    """Find the actual fire progression dates from FEDS25MTBS perimeter data.
    
    Analyzes consecutive perimeters to find when the fire actually starts
    progressing (first change) and when it stops (last change). Uses a
    tolerance-based comparison to filter out floating-point noise.
    
    Args:
        event_id: Event ID to look up.
        year: Year of the fire event.
        base_dir: Base directory containing FEDS25MTBS data.
    
    Returns:
        Tuple of (t_start, t_end) representing the actual fire progression period.
    
    Raises:
        AssertionError: If the GeoPackage file doesn't exist.
    """
    data_dir = os.path.join(base_dir, str(year), event_id + '.gpkg')
    assert os.path.exists(data_dir), f"Error: FEDS25MTBS data not found at {data_dir}"
    
    gdf = gpd.read_file(data_dir, layer='perimeter')
    
    # Sort by timestamp to ensure correct consecutive comparisons
    gdf = gdf.sort_values('t').reset_index(drop=True)
    
    # Get timestamps and geometries (skip None geometries)
    timestamps: list[datetime] = []
    geometries = []
    
    for _, row in gdf.iterrows():
        if row.geometry is not None:
            timestamps.append(pd.to_datetime(row['t']).to_pydatetime())
            geometries.append(row.geometry)
    
    if not geometries:
        raise ValueError(f"No valid geometries found for event {event_id}")
    
    if len(geometries) < 2:
        # If there's only one frame, return it as both start and end
        return timestamps[0], timestamps[0]
    
    # Find where consecutive frames differ (using threshold to filter floating-point noise)
    # changes[i] is True if geometries[i] differs from geometries[i+1]
    changes: list[bool] = []
    for i in range(len(geometries) - 1):
        changes.append(not geometries_are_equal(geometries[i], geometries[i + 1]))
    
    # Find first change: t_start is the timestamp of the first frame that differs from the next
    first_change_idx = 0
    for i, changed in enumerate(changes):
        if changed:
            first_change_idx = i
            break
    
    # Find last change: t_end is the timestamp of the last frame that differs from the previous
    # This is the frame after the last True in changes
    last_change_idx = len(geometries) - 1
    for i in range(len(changes) - 1, -1, -1):
        if changes[i]:
            last_change_idx = i + 1  # The frame after the change
            break
    
    log.info(
        f"Fire progression dates: {timestamps[first_change_idx]} to {timestamps[last_change_idx]} "
        f"(frames {first_change_idx} to {last_change_idx} of {len(geometries)})"
    )
    
    return timestamps[first_change_idx], timestamps[last_change_idx]


def get_task_info(
    fire_info: FireInfo,
    resolution: int = 30,
    buffer: int = 100,
    crs: str = "EPSG:5070",
    feds25mtbs_base_dir: str = DEFAULT_FEDS25MTBS_DIR,
) -> TaskInfo:
    """Create a processing task configuration from fire event information.
    
    Transforms the fire bounds to the target CRS, applies a buffer, and
    calculates the output grid dimensions. The fire progression dates (t_start
    and t_end) are determined from the FEDS25MTBS perimeter data by finding
    when the fire perimeter actually changes.
    
    Args:
        fire_info: Fire event information.
        resolution: Target spatial resolution in meters.
        buffer: Buffer distance to add around the fire bounds in meters.
        crs: Target coordinate reference system.
        feds25mtbs_base_dir: Base directory containing FEDS25MTBS data.
    
    Returns:
        TaskInfo object defining the processing parameters.
    """
    # Get actual fire progression dates from perimeter data
    t_start, t_end = get_fire_progression_dates(
        fire_info.event_id,
        fire_info.year,
        feds25mtbs_base_dir
    )

    minx, miny, maxx, maxy = fire_info.bounds
    bbox_poly = box(minx, miny, maxx, maxy)
    bounds_gs = gpd.GeoSeries([bbox_poly], crs="EPSG:4326")
    bounds_proj = bounds_gs.to_crs(crs)

    bounds_proj = bounds_proj.buffer(buffer)

    t_minx, t_miny, t_maxx, t_maxy = bounds_proj.total_bounds

    target_bounds = (
        math.floor(t_minx / resolution) * resolution,
        math.floor(t_miny / resolution) * resolution,
        math.ceil(t_maxx / resolution) * resolution,
        math.ceil(t_maxy / resolution) * resolution
    )

    width = int((target_bounds[2] - target_bounds[0]) / resolution)
    height = int((target_bounds[3] - target_bounds[1]) / resolution)

    return TaskInfo(
        event_id=fire_info.event_id,
        name=fire_info.name,
        year=fire_info.year,
        t_start=t_start,
        t_end=t_end,
        resolution=resolution,
        bounds=target_bounds,
        shape=(height, width),
        crs=crs,
    )


# =============================================================================
# I/O Functions
# =============================================================================

def save_numpy(
    task_info: TaskInfo,
    data: DataWithMetadata,
    output_dir: str = 'output'
) -> None:
    """Save processed data to a numpy file.
    
    Creates a directory structure: output_dir/event_id/data_name.npy
    
    Args:
        task_info: Task configuration containing event_id.
        data: Data to save with metadata.
        output_dir: Base output directory.
    """
    event_id = task_info.event_id
    output_path = os.path.join(output_dir, event_id)
    os.makedirs(output_path, exist_ok=True)

    output_path = os.path.join(output_path, f"{data.name}.npy")
    np.save(output_path, asdict(data))

    log.info(f"Saved {data.name} data to {output_path}")


def load_numpy(filepath: str) -> DataWithMetadata:
    """Load processed data from a numpy file.
    
    Args:
        filepath: Path to the .npy file.
    
    Returns:
        DataWithMetadata object with loaded data.
    """
    loaded_dict = np.load(filepath, allow_pickle=True).item()
    obj = DataWithMetadata(**loaded_dict)
    return obj


# =============================================================================
# FEDS25MTBS Processing
# =============================================================================

def process_feds25mtbs(
    task_info: TaskInfo,
    base_dir: str = DEFAULT_FEDS25MTBS_DIR
) -> DataWithMetadata:
    """Process FEDS25MTBS fire perimeter data into rasterized time series.
    
    Reads the GeoPackage file for the fire event and rasterizes each timestep's
    perimeter polygon to match the task grid specification. Only includes frames
    within the task_info time range (t_start to t_end) where the perimeter
    actually changed from the previous frame.
    
    Args:
        task_info: Task configuration with event details and grid parameters.
        base_dir: Base directory containing the FEDS25MTBS data.
    
    Returns:
        DataWithMetadata containing boolean rasters for each timestep.
    
    Raises:
        AssertionError: If the GeoPackage file doesn't exist.
    """
    log.info(f"Processing FEDS25MTBS for event_id: {task_info.event_id}")

    data_dir = os.path.join(base_dir, str(
        task_info.year), task_info.event_id + '.gpkg')

    assert os.path.exists(
        data_dir), f"Error: FEDS25MTBS data not found at {data_dir}"

    gdf = gpd.read_file(data_dir, layer='perimeter')
    
    # Sort by timestamp to ensure correct consecutive comparisons
    gdf = gdf.sort_values('t').reset_index(drop=True)

    all_data = []
    all_timestamps = []

    for _, row in gdf.iterrows():
        timestamp = pd.to_datetime(row['t'])
        timestamp = timestamp.to_pydatetime()
        geom = row.geometry

        if geom is None:
            continue

        if geom.geom_type == 'MultiPolygon':
            all_data.append(geom)
        elif geom.geom_type == 'Polygon':
            all_data.append(MultiPolygon([geom]))
        else:
            log.warning(
                f"Unexpected geometry type: {geom.geom_type} for event_id: {task_info.event_id}")
            continue

        all_timestamps.append(timestamp)

    # Filter to only include frames within t_start and t_end
    filtered_data = []
    filtered_timestamps = []
    for ts, geom in zip(all_timestamps, all_data):
        if task_info.t_start <= ts <= task_info.t_end:
            filtered_timestamps.append(ts)
            filtered_data.append(geom)
    
    log.info(
        f"Filtered frames: {len(filtered_data)} of {len(all_data)} "
        f"(t_start={task_info.t_start}, t_end={task_info.t_end})"
    )

    # Filter to only include frames where perimeter is different from last imported
    # Uses threshold-based comparison to filter floating-point noise
    data_list = []
    timestamps = []
    last_imported_geom = None
    for ts, geom in zip(filtered_timestamps, filtered_data):
        if last_imported_geom is None or not geometries_are_equal(geom, last_imported_geom):
            data_list.append(geom)
            timestamps.append(ts)
            last_imported_geom = geom  # Only update when we actually import
    
    log.info(f"Unique frames imported: {len(data_list)} of {len(filtered_data)}")

    # Calculate grid dimensions from task bounds
    t_minx, t_miny, t_maxx, t_maxy = task_info.bounds

    res = task_info.resolution
    transform = from_origin(t_minx, t_maxy, res, res)

    log.info(f"Target Grid: {task_info.shape} pixels @ {res}m resolution")

    # Prepare GeoDataFrame with geometries
    gdf = gpd.GeoDataFrame({
        'geometry': data_list,
        'timestamp': timestamps
    }, crs="EPSG:4326")

    log.info(
        f"Reprojecting geometries from EPSG:4326 to target CRS {task_info.crs}"
    )
    gdf = gdf.to_crs(task_info.crs)

    # Rasterize each timestep
    processed_rasters = []
    for _, row in gdf.iterrows():
        # Burn value of 1 where polygon exists, 0 elsewhere
        shapes = [(row.geometry, 1)]

        raster = features.rasterize(
            shapes=shapes,
            out_shape=task_info.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )
        assert raster.shape == task_info.shape, (
            "Rasterized shape does not match target shape"
        )
        raster = raster.astype(np.bool)
        processed_rasters.append(raster)

    return DataWithMetadata(
        name="burn_perimeters",
        data=processed_rasters,
        timestamps=timestamps,
        source="'FEDS25MTBS; https://doi.org/10.1038/s41597-022-01343-0; requested via SharePoint by Huilin'",
        resolution=375,
    )


# =============================================================================
# FRP (Fire Radiative Power) Processing Functions
# =============================================================================

def _get_perimeter_masks_from_data(
    perimeter_data: DataWithMetadata,
) -> tuple[list[np.ndarray], list[datetime]]:
    """Extract perimeter masks and timestamps from DataWithMetadata.
    
    Converts the output of process_feds25mtbs into the format needed
    for FRP masking.
    
    Args:
        perimeter_data: DataWithMetadata from process_feds25mtbs.
    
    Returns:
        Tuple of (list of perimeter masks, list of timestamps).
    """
    masks = [mask.astype(bool) for mask in perimeter_data.data]
    timestamps = perimeter_data.timestamps or []
    return masks, timestamps


def _get_perimeter_mask_for_time(
    target_time: datetime,
    perimeter_masks: list[np.ndarray],
    perimeter_timestamps: list[datetime],
) -> Optional[np.ndarray]:
    """Get the appropriate perimeter mask for a given time.
    
    Returns the most recent perimeter that is <= target_time.
    
    Args:
        target_time: Target timestamp.
        perimeter_masks: List of perimeter masks.
        perimeter_timestamps: List of perimeter timestamps.
    
    Returns:
        Perimeter mask or None if no suitable mask found.
    """
    if not perimeter_masks:
        return None
    
    best_mask = None
    best_time = None
    
    for mask, ts in zip(perimeter_masks, perimeter_timestamps):
        if ts <= target_time:
            if best_time is None or ts > best_time:
                best_mask = mask
                best_time = ts
    
    return best_mask


def _load_firms_data(
    task_info: TaskInfo,
    firms_dir: str = DEFAULT_FIRMS_DIR,
    buffer_deg: float = 0.01,
) -> pd.DataFrame:
    """Load FIRMS data for fires from 2024 onwards.
    
    Filters FIRMS data by spatial and temporal bounds defined in TaskInfo.
    
    Args:
        task_info: Task configuration with bounds and time range.
        firms_dir: Directory containing FIRMS CSV files.
        buffer_deg: Buffer in degrees to add around bounds for spatial filtering.
    
    Returns:
        DataFrame with columns [Lat, Lon, FRP, Confidence, DNFlag, t, Event_ID].
    """
    log.info(f"Loading FIRMS data for event {task_info.event_id} (year {task_info.year})")
    
    # Transform bounds from target CRS back to lat/lon for filtering
    transformer = Transformer.from_crs(task_info.crs, "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = task_info.bounds
    
    # Transform all four corners
    corners_x = [minx, maxx, minx, maxx]
    corners_y = [miny, miny, maxy, maxy]
    lons, lats = transformer.transform(corners_x, corners_y)
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Add buffer
    min_lon -= buffer_deg
    max_lon += buffer_deg
    min_lat -= buffer_deg
    max_lat += buffer_deg
    
    # Time bounds (use exact task time range, no buffer to avoid empty first frame)
    t_start = task_info.t_start
    t_end = task_info.t_end
    
    all_data = []
    
    for firms_file in FIRMS_FILES:
        filepath = os.path.join(firms_dir, firms_file)
        if not os.path.exists(filepath):
            log.warning(f"FIRMS file not found: {filepath}")
            continue
        
        # Check cache first
        if filepath in _firms_cache:
            log.info(f"Using cached {firms_file} ({len(_firms_cache[filepath]):,} points)")
            df = _firms_cache[filepath]
        else:
            log.info(f"Loading {firms_file} (this may take a moment for large files)...")
            df = pd.read_csv(filepath, usecols=FIRMS_USECOLS)
            
            # Create datetime column
            df['acq_time_str'] = df['acq_time'].astype(str).str.zfill(4)
            df['t'] = pd.to_datetime(
                df['acq_date'] + ' ' + 
                df['acq_time_str'].str[:2] + ':' + 
                df['acq_time_str'].str[2:]
            )
            df = df.drop(columns=['acq_time_str', 'acq_date', 'acq_time'])
            
            # Cache for future use
            _firms_cache[filepath] = df
            log.info(f"  Cached {len(df):,} points from {firms_file}")
        
        # Spatial filtering
        spatial_mask = (
            (df['latitude'] >= min_lat) &
            (df['latitude'] <= max_lat) &
            (df['longitude'] >= min_lon) &
            (df['longitude'] <= max_lon)
        )

        # Temporal filtering
        temporal_mask = (df['t'] >= t_start) & (df['t'] <= t_end)
        
        filtered = df[spatial_mask & temporal_mask].copy()
        
        if len(filtered) > 0:
            log.info(f"  Found {len(filtered)} fire points in {firms_file}")
            all_data.append(filtered)
    
    if not all_data:
        log.warning("No FIRMS data found for the specified bounds and time range")
        return pd.DataFrame(columns=['Lat', 'Lon', 'FRP', 'Confidence', 'DNFlag', 't', 'Event_ID'])
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates (same location and time from different files)
    combined = combined.drop_duplicates(subset=['latitude', 'longitude', 't'])
    
    # Format output to match Firepix format
    output = pd.DataFrame({
        'Lat': combined['latitude'],
        'Lon': combined['longitude'],
        'FRP': combined['frp'],
        'Confidence': combined['confidence'],
        'DNFlag': combined['daynight'],
        't': combined['t'],
        'Event_ID': task_info.event_id
    })
    
    log.info(f"Total FIRMS points loaded: {len(output)}")
    return output


def _load_firepix_data(
    task_info: TaskInfo,
    firepix_dir: str = DEFAULT_FIREPIX_DIR,
) -> pd.DataFrame:
    """Load pre-processed firepix data for fires before 2024.
    
    Args:
        task_info: Task configuration with event ID and year.
        firepix_dir: Directory containing Firepix CSV files.
    
    Returns:
        DataFrame with columns [Lat, Lon, FRP, Confidence, DNFlag, t, Event_ID].
    """
    filepath = os.path.join(firepix_dir, f"Firepix_{task_info.year}.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Firepix file not found: {filepath}")
    
    # Check cache first
    if filepath in _firepix_cache:
        log.info(f"Using cached firepix data for year {task_info.year}")
        df = _firepix_cache[filepath]
    else:
        log.info(f"Loading firepix data from {filepath}")
        df = pd.read_csv(filepath)
        df["t"] = pd.to_datetime(df["t"])
        _firepix_cache[filepath] = df
        log.info(f"  Cached {len(df):,} firepix points for year {task_info.year}")
    
    # Filter by event_id
    df = df[df["Event_ID"] == task_info.event_id].copy()
    
    if df.empty:
        log.warning(f"No firepix data found for event_id: {task_info.event_id}")
        return pd.DataFrame(columns=['Lat', 'Lon', 'FRP', 'Confidence', 'DNFlag', 't', 'Event_ID'])
    
    # Select relevant columns
    output = df[['Lat', 'Lon', 'FRP', 'Confidence', 'DNFlag', 't', 'Event_ID']].copy()
    
    log.info(f"Loaded {len(output)} firepix points for event {task_info.event_id}")
    return output


def _gaussian_splat_rasterize(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    resolution: int,
    source_resolution: float = 375.0,
) -> np.ndarray:
    """Rasterize points using Gaussian splatting (average-preserving interpolation).
    
    Each point's FRP value is interpolated using a Gaussian kernel sized to match
    the source sensor's pixel footprint. The central pixel retains the original
    observed value, with surrounding pixels receiving gradually decreasing values
    based on Gaussian falloff. The average FRP across the source resolution area
    (e.g., 375m for VIIRS) approximates the original observed value.
    
    Args:
        x: X coordinates in target CRS.
        y: Y coordinates in target CRS.
        values: Values to rasterize (e.g., FRP).
        bounds: Bounding box (minx, miny, maxx, maxy).
        shape: Output shape (height, width).
        resolution: Target pixel resolution in meters.
        source_resolution: Source sensor pixel resolution in meters (default 375m for VIIRS).
    
    Returns:
        Rasterized array with average-preserved values within source pixel regions.
    """
    minx, miny, maxx, maxy = bounds
    height, width = shape
    
    # Initialize accumulator
    raster = np.zeros(shape, dtype=np.float64)
    
    # Calculate Gaussian sigma in pixels
    # Use half the source resolution as sigma so that ~95% of energy is within the footprint
    sigma_pixels = (source_resolution / resolution) / 2.0
    
    # Kernel radius: cover 3 sigma in each direction
    kernel_radius = int(np.ceil(3 * sigma_pixels))
    
    # Convert coordinates to fractional pixel indices
    px = (x - minx) / resolution
    py = (maxy - y) / resolution
    
    for i in range(len(values)):
        # Center pixel
        px_center = int(np.round(px[i]))
        py_center = int(np.round(py[i]))
        
        val = values[i]
        
        # Apply Gaussian kernel directly (average-preserving)
        for dy in range(-kernel_radius, kernel_radius + 1):
            for dx in range(-kernel_radius, kernel_radius + 1):
                row = py_center + dy
                col = px_center + dx
                
                if 0 <= row < height and 0 <= col < width:
                    # Distance from point center to pixel center
                    dist_x = (col + 0.5) - px[i]
                    dist_y = (row + 0.5) - py[i]
                    dist_sq = dist_x**2 + dist_y**2
                    
                    # Gaussian weight (peaks at 1 at center, decreases outward)
                    w = np.exp(-dist_sq / (2 * sigma_pixels**2))
                    
                    # Average-preserving: pixel value = weight * original value
                    # At center, w ≈ 1 so pixel ≈ val
                    # Away from center, w < 1 so pixel < val
                    raster[row, col] += w * val
    
    return raster.astype(np.float32)


def _rasterize_fire_points(
    df: pd.DataFrame,
    task_info: TaskInfo,
    perimeter_masks: list[np.ndarray],
    perimeter_timestamps: list[datetime],
    time_interval_hours: int = 12,
) -> tuple[list[np.ndarray], list[datetime]]:
    """Rasterize fire points and apply perimeter masking.
    
    Groups fire points by time intervals and creates FRP rasters
    using Gaussian splatting (average-preserving interpolation) method.
    
    Args:
        df: DataFrame with fire point data.
        task_info: Task configuration.
        perimeter_masks: List of fire perimeter masks.
        perimeter_timestamps: List of perimeter timestamps.
        time_interval_hours: Time interval for grouping points.
    
    Returns:
        Tuple of (list of FRP rasters, list of timestamps).
    """
    if df.empty:
        return [], []
    
    bounds = task_info.bounds
    shape = task_info.shape
    resolution = task_info.resolution
    
    # Transform point coordinates from lat/lon to target CRS
    transformer = Transformer.from_crs("EPSG:4326", task_info.crs, always_xy=True)
    x, y = transformer.transform(df["Lon"].values, df["Lat"].values)
    df = df.copy()
    df["x"] = np.array(x)
    df["y"] = np.array(y)
    
    # Group by time intervals
    df["time_group"] = df["t"].dt.floor(f"{time_interval_hours}h")
    time_groups = sorted(df["time_group"].unique())
    
    rasters = []
    timestamps = []
    
    for tg in tqdm(time_groups, desc="Rasterizing FRP"):
        group = df[df["time_group"] == tg]
        
        # Rasterize using Gaussian splatting
        raster = _gaussian_splat_rasterize(
            group["x"].values,
            group["y"].values,
            group["FRP"].values,
            bounds,
            shape,
            resolution,
        )
        
        # Use the first observation time in the group to preserve original timestep
        target_time = group["t"].iloc[0].to_pydatetime()
        
        # Apply perimeter mask
        mask = _get_perimeter_mask_for_time(target_time, perimeter_masks, perimeter_timestamps)
        if mask is not None:
            raster = raster * mask.astype(np.float32)
        
        rasters.append(raster)
        timestamps.append(target_time)
    
    return rasters, timestamps


def process_frp(
    task_info: TaskInfo,
    perimeter_data: DataWithMetadata,
    time_interval_hours: int = 12,
    time_of_day: Literal['all', 'day', 'night'] = 'all',
    firms_dir: str = DEFAULT_FIRMS_DIR,
    firepix_dir: str = DEFAULT_FIREPIX_DIR,
) -> DataWithMetadata:
    """Process fire pixels and FRP for a fire event.
    
    Automatically selects the data source based on the year:
    - For fires from 2024 onwards: Uses NASA FIRMS data
    - For fires before 2024: Uses pre-processed firepix data from FEDS25MTBS
    
    Uses Gaussian splatting for rasterization.
    Results are masked by the known fire perimeter at each time step.
    
    Args:
        task_info: Task configuration containing event details and processing parameters.
        perimeter_data: DataWithMetadata from process_feds25mtbs for masking.
        time_interval_hours: Time interval for grouping fire points (default 12 hours).
        time_of_day: Filter by time of day - 'all' (default), 'day' (06:00-18:00 UTC),
                     or 'night' (18:00-06:00 UTC).
        firms_dir: Directory containing FIRMS CSV files.
        firepix_dir: Directory containing pre-processed Firepix CSV files.
    
    Returns:
        DataWithMetadata containing:
        - name: "frp", "frp_day", or "frp_night" depending on time_of_day
        - data: List of FRP rasters (numpy arrays) for each time step
        - timestamps: List of datetime objects corresponding to each raster
        - source: Data source used (FIRMS or FEDS25MTBS firepix)
        - resolution: Spatial resolution in meters
        - unit: "MW" (megawatts)
    """
    # Determine output name and source suffix based on time_of_day
    if time_of_day == 'day':
        output_name = "frp_day"
        source_suffix = " - Day"
        observation_time = "day (06:00-18:00 UTC)"
        log.info(f"Processing DAYTIME FRP for event: {task_info.event_id}")
    elif time_of_day == 'night':
        output_name = "frp_night"
        source_suffix = " - Night"
        observation_time = "night (18:00-06:00 UTC)"
        log.info(f"Processing NIGHTTIME FRP for event: {task_info.event_id}")
    else:
        output_name = "frp"
        source_suffix = ""
        observation_time = None
        log.info(f"Processing FRP for event: {task_info.event_id}")
    
    log.info(f"Year: {task_info.year}, Time range: {task_info.t_start} to {task_info.t_end}")
    
    # Get perimeter masks from existing processed data
    perimeter_masks, perimeter_timestamps = _get_perimeter_masks_from_data(perimeter_data)
    log.info(f"Using {len(perimeter_masks)} perimeter frames for masking")
    
    # Select data source based on year
    if task_info.year >= 2024:
        log.info("Using FIRMS data source (year >= 2024)")
        source = f"NASA FIRMS (VIIRS Active Fire){source_suffix}"
        df = _load_firms_data(task_info, firms_dir)
    else:
        log.info("Using FEDS25MTBS firepix data source (year < 2024)")
        source = f"FEDS25MTBS Firepix (VIIRS Active Fire){source_suffix}"
        df = _load_firepix_data(task_info, firepix_dir)
    
    # Apply time-of-day filter if specified
    if time_of_day == 'day':
        df = df.copy()
        hour = df["t"].dt.hour
        df = df[(hour >= 6) & (hour < 18)].copy()
        log.info(f"Daytime fire points (06:00-18:00 UTC): {len(df)}")
    elif time_of_day == 'night':
        df = df.copy()
        hour = df["t"].dt.hour
        df = df[(hour >= 18) | (hour < 6)].copy()
        log.info(f"Nighttime fire points (18:00-06:00 UTC): {len(df)}")
    
    n_points = len(df)
    if time_of_day == 'all':
        log.info(f"Total fire points: {n_points}")
    
    # Build note dictionary
    note: dict = {
        "event_id": task_info.event_id,
        "year": task_info.year,
        "n_points": n_points,
        "time_interval_hours": time_interval_hours,
        "rasterization_method": "gaussian_splatting_average",
        "perimeter_masked": True,
    }
    if observation_time:
        note["observation_time"] = observation_time
    
    if df.empty:
        log.warning(f"No fire points found for {output_name}")
        return DataWithMetadata(
            name=output_name,
            data=[],
            timestamps=[],
            source=source,
            resolution=task_info.resolution,
            unit="MW",
            note=note,
        )
    
    # Rasterize fire points to grid with perimeter masking
    rasters, timestamps = _rasterize_fire_points(
        df, task_info, perimeter_masks, perimeter_timestamps, time_interval_hours
    )
    
    log.info(f"Created {len(rasters)} time step rasters")
    
    note["n_perimeter_frames"] = len(perimeter_masks)
    
    return DataWithMetadata(
        name=output_name,
        data=rasters,
        timestamps=timestamps,
        source=source,
        resolution=task_info.resolution,
        unit="MW",
        note=note,
    )


def process_frp_day(
    task_info: TaskInfo,
    perimeter_data: DataWithMetadata,
    firms_dir: str = DEFAULT_FIRMS_DIR,
    firepix_dir: str = DEFAULT_FIREPIX_DIR,
) -> DataWithMetadata:
    """Process daytime FRP only (observations from 06:00-18:00 UTC).
    
    Wrapper around process_frp with time_of_day='day' and 24-hour intervals.
    
    Args:
        task_info: Task configuration containing event details.
        perimeter_data: DataWithMetadata from process_feds25mtbs for masking.
        firms_dir: Directory containing FIRMS CSV files.
        firepix_dir: Directory containing pre-processed Firepix CSV files.
    
    Returns:
        DataWithMetadata with daytime FRP rasters at 24-hour intervals.
    """
    return process_frp(
        task_info=task_info,
        perimeter_data=perimeter_data,
        time_interval_hours=24,
        time_of_day='day',
        firms_dir=firms_dir,
        firepix_dir=firepix_dir,
    )


def process_frp_night(
    task_info: TaskInfo,
    perimeter_data: DataWithMetadata,
    firms_dir: str = DEFAULT_FIRMS_DIR,
    firepix_dir: str = DEFAULT_FIREPIX_DIR,
) -> DataWithMetadata:
    """Process nighttime FRP only (observations from 18:00-06:00 UTC).
    
    Wrapper around process_frp with time_of_day='night' and 24-hour intervals.
    
    Args:
        task_info: Task configuration containing event details.
        perimeter_data: DataWithMetadata from process_feds25mtbs for masking.
        firms_dir: Directory containing FIRMS CSV files.
        firepix_dir: Directory containing pre-processed Firepix CSV files.
    
    Returns:
        DataWithMetadata with nighttime FRP rasters at 24-hour intervals.
    """
    return process_frp(
        task_info=task_info,
        perimeter_data=perimeter_data,
        time_interval_hours=24,
        time_of_day='night',
        firms_dir=firms_dir,
        firepix_dir=firepix_dir,
    )


# =============================================================================
# Interpolation Functions
# =============================================================================

def signed_bwdist(im: np.ndarray) -> np.ndarray:
    """Compute the Signed Distance Field (SDF) for a binary image.
    
    Pixels inside the shape have positive values (distance to boundary).
    Pixels outside the shape have negative values (distance to boundary).
    
    Args:
        im: Binary input image.
    
    Returns:
        Signed distance field array.
    """
    # Ensure boolean
    im = im.astype(bool)

    # distance_transform_edt calculates distance to the nearest zero pixel
    inner_dist = distance_transform_edt(im)
    outer_dist = distance_transform_edt(~im)

    # Inside is positive, outside is negative
    return inner_dist - outer_dist


def interp_shape(
    array_a: np.ndarray,
    array_b: np.ndarray,
    precision: float = 0.5
) -> np.ndarray:
    """Interpolate between two contours (boolean masks) using SDF interpolation.
    
    Uses signed distance field interpolation to create smooth transitions
    between two binary shapes.
    
    Reference: https://stackoverflow.com/questions/48818373/interpolate-between-two-images
    
    Args:
        array_a: First binary mask (precision=0.0 returns this).
        array_b: Second binary mask (precision=1.0 returns this).
        precision: Interpolation factor between 0.0 and 1.0.
    
    Returns:
        Interpolated binary mask.
    
    Raises:
        ValueError: If shapes don't match or precision is out of range.
    """
    if array_a.shape != array_b.shape:
        raise ValueError(f"Shape mismatch: {array_a.shape} vs {array_b.shape}")

    if not (0 <= precision <= 1):
        raise ValueError("Precision must be between 0 and 1")

    # Get Signed Distance Functions
    sdf_a = signed_bwdist(array_a)
    sdf_b = signed_bwdist(array_b)

    # Linear Interpolation of the SDFs
    # Formula: (1 - t) * A + t * B
    interpolated_sdf = (1 - precision) * sdf_a + precision * sdf_b

    # Threshold back to boolean
    # Any value > 0 represents the inside of the new shape
    out = interpolated_sdf > 0

    return out


def interpolate_burn_perimeters(
    data: DataWithMetadata,
    multiplier: int
) -> DataWithMetadata:
    """Interpolate additional frames between existing burn perimeter timesteps.
    
    Uses SDF-based shape interpolation to create smooth temporal transitions
    between fire perimeter snapshots.
    
    Args:
        data: Burn perimeter data with timestamps.
        multiplier: Number of intermediate frames to insert between each pair.
    
    Returns:
        DataWithMetadata with interpolated frames and timestamps.
    """
    # Cannot interpolate if fewer than 2 frames
    if not data.data or len(data.data) < 2:
        return data

    n_original = len(data.data)
    new_data_list = []

    assert (data.timestamps is not None) and (
        len(data.timestamps) == n_original
    )
    new_timestamps = []

    # Iterate through consecutive pairs
    for i in range(n_original - 1):
        curr_frame = data.data[i]
        next_frame = data.data[i + 1]
        curr_time = data.timestamps[i]
        next_time = data.timestamps[i + 1]
        time_diff = next_time - curr_time

        # Add original frame
        new_data_list.append(curr_frame)
        new_timestamps.append(curr_time)

        # Generate intermediate frames
        steps = multiplier + 1
        for step in range(1, steps):
            t = step / steps
            interp_frame = interp_shape(curr_frame, next_frame, precision=t)
            interp_time = curr_time + (time_diff * t)
            new_data_list.append(interp_frame)
            new_timestamps.append(interp_time)

    # Add final frame
    new_data_list.append(data.data[-1])
    new_timestamps.append(data.timestamps[-1])

    # Return new object with updated fields
    return replace(
        data,
        data=new_data_list,
        timestamps=new_timestamps,
        note=data.note | {'interpolate': multiplier}
    )


# =============================================================================
# Google Earth Engine Functions
# =============================================================================

class GEEProjectNotConfiguredError(Exception):
    """Raised when Google Earth Engine project is not configured."""
    pass


def _ensure_ee_initialized() -> None:
    """Ensure Google Earth Engine is initialized.
    
    Uses the project configured via `earthengine set_project` command.
    Prompts for authentication if needed.
    
    Raises:
        GEEProjectNotConfiguredError: If no GEE project is configured.
    """
    try:
        # Check if already initialized
        ee.Number(1).getInfo()
    except Exception:
        log.info("Earth Engine not initialized. Attempting to initialize...")
        try:
            ee.Initialize()
        except Exception as e:
            error_msg = str(e).lower()
            if "project" in error_msg or "quota" in error_msg or "credentials" in error_msg:
                log.info("Authentication required. Opening browser...")
                ee.Authenticate()
                try:
                    ee.Initialize()
                except Exception as init_error:
                    raise GEEProjectNotConfiguredError(
                        "\n\n" + "="*60 + "\n"
                        "Google Earth Engine project not configured!\n"
                        + "="*60 + "\n\n"
                        "Please set your GEE project by running:\n\n"
                        "    earthengine set_project YOUR-PROJECT-ID\n\n"
                        "To find your project ID:\n"
                        "  1. Go to https://console.cloud.google.com/\n"
                        "  2. Select or create a project with Earth Engine enabled\n"
                        "  3. Copy the project ID from the project selector\n\n"
                        f"Original error: {init_error}\n"
                    ) from init_error
            else:
                raise


def _download_processed_image(
    image: ee.Image,
    task_info: TaskInfo,
    band_name: str
) -> np.ndarray:
    """Download a processed Earth Engine image as a numpy array.
    
    Args:
        image: Earth Engine image to download.
        task_info: Task configuration with bounds and shape.
        band_name: Name of the band to select.
    
    Returns:
        2D numpy array with the downloaded data.
    
    Raises:
        ValueError: If download fails.
    """
    _ensure_ee_initialized()

    roi = ee.Geometry.Rectangle(task_info.bounds, task_info.crs, False)

    log.info(f"Downloading band '{band_name}' via geemap...")

    try:
        image = image.select(band_name).reproject(
            crs=task_info.crs,
            scale=task_info.resolution
        )
        data = geemap.ee_to_numpy(
            image,
            region=roi,
        )
    except Exception as e:
        log.error(f"Error downloading with geemap: {e}")
        raise e

    if data is None:
        raise ValueError("Download failed: geemap returned None.")

    # geemap.ee_to_numpy typically returns shape (Height, Width, Bands).
    # If the result is 3D with a single band channel, squeeze it to 2D (Height, Width)
    # to match the behavior of the original NPY extraction.
    assert data.ndim == 3 and data.shape[2] == 1
    data = np.squeeze(data, axis=2)

    log.info(f"Downloaded data shape: {data.shape}")

    assert data.shape == task_info.shape, \
        f"Error: Downloaded data shape {data.shape} does not match expected shape {task_info.shape}"

    return data


def download_gee_task(
    task_info: TaskInfo,
    dataset_name: str,
    imagecollection: str | list[ee.Image],
    band: str,
    resample: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear'
) -> DataWithMetadata:
    """Download data from a Google Earth Engine ImageCollection.
    
    Args:
        task_info: Task configuration with bounds and resolution.
        dataset_name: Name for the output data layer.
        imagecollection: GEE ImageCollection path or list of ee.Image objects.
        band: Band name to extract.
        resample: Resampling method ('nearest', 'bilinear', or 'bicubic').
    
    Returns:
        DataWithMetadata containing the downloaded array.
    
    Raises:
        ValueError: If the ROI is outside the collection coverage.
    """

    log.info(
        f"Downloading {dataset_name} data for event_id: {task_info.event_id} "
        f"from Google Earth Engine"
    )

    _ensure_ee_initialized()

    roi = ee.Geometry.Rectangle(task_info.bounds, task_info.crs, False)
    collection = ee.ImageCollection(imagecollection).filterBounds(roi)

    # Handle case where imagecollection is already a list of images
    if isinstance(imagecollection, list):
        collection = ee.ImageCollection(imagecollection)

    if collection.size().getInfo() == 0:
        error_msg = (f"The requested ROI is outside the coverage of {imagecollection}. "
                     f"Images found: 0. Bounds: {task_info.bounds}")
        raise ValueError(error_msg)

    native_proj = collection.first().select(band).projection()
    image = collection.mosaic().select(band)
    image = image.setDefaultProjection(native_proj)

    if resample != 'nearest':
        image = image.resample(resample)

    data_array = _download_processed_image(image, task_info, band)

    return DataWithMetadata(
        name=dataset_name,
        data=[data_array],
        timestamps=[task_info.t_start],
        source=f"Google Earth Engine: {imagecollection}",
    )


# =============================================================================
# Data Download Functions
# =============================================================================

def download_usgs(task_info: TaskInfo) -> DataWithMetadata:
    """Download USGS 3DEP 1m elevation data.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataWithMetadata containing elevation in meters (int16).
    """
    log.info(f"Downloading USGS data for event_id: {task_info.event_id}")
    data = download_gee_task(
        task_info,
        dataset_name="elevation",
        band="elevation",
        imagecollection="USGS/3DEP/1m",
        resample='bilinear',
    )
    return replace(
        data,
        data=[np.around(data.data[0]).astype(np.int16)],
        resolution=1,
        unit="m",
    )


def download_landfire(task_info: TaskInfo) -> list[DataWithMetadata]:
    """Download LANDFIRE fuel data (Canopy Bulk Density and Canopy Cover).
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        List of DataWithMetadata for CBD and CC layers.
    """
    log.info(f"Downloading LANDFIRE data for event_id: {task_info.event_id}")
    payload = []

    data = download_gee_task(
        task_info,
        dataset_name="cbd",
        band="CBD",
        imagecollection="projects/sat-io/open-datasets/landfire/FUEL/CBD",
        resample='bilinear',
    )
    payload.append(replace(
        data,
        data=[np.around(data.data[0]).astype(np.int16)],
        resolution=30,
        unit="100kg/m^3",
    ))

    data = download_gee_task(
        task_info,
        dataset_name="cc",
        band="CC",
        imagecollection="projects/sat-io/open-datasets/landfire/FUEL/CC",
        resample='bilinear',
    )
    payload.append(replace(
        data,
        data=[np.around(data.data[0]).astype(np.int16)],
        resolution=30,
        unit="%",
    ))

    return payload


# =============================================================================
# HRRR Weather Data Functions
# =============================================================================

def clip_hrrr_to_task(
    hrrr_data: xr.Dataset,
    task_info: TaskInfo,
    target_resolution: int = 500
) -> xr.Dataset:
    """Clip and reproject HRRR data to match task bounds.
    
    Transforms HRRR data from its native Lambert Conformal Conic projection
    to the task CRS and clips to the task bounds.
    
    Args:
        hrrr_data: HRRR xarray Dataset from Herbie.
        task_info: Task configuration with bounds and CRS.
        target_resolution: Output resolution in meters.
    
    Returns:
        Reprojected and clipped xarray Dataset.
    """

    assert "x" in hrrr_data.dims and "y" in hrrr_data.dims, (
        "HRRR data must have 'x' and 'y' dimensions"
    )
    assert hrrr_data.rio.crs is not None, (
        "HRRR data must have a valid CRS for reprojection"
    )

    # Get CRS from Herbie accessor (Lambert Conformal Conic)
    crs = hrrr_data.herbie.crs

    # Transform coordinates from lat/lon to HRRR projection
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xx, yy = transformer.transform(
        hrrr_data.longitude.values, hrrr_data.latitude.values
    )

    # Extract 1D axes from the rectilinear grid
    x_coords = xx[0, :]
    y_coords = yy[:, 0]

    # Assign projected coordinates
    hrrr_data = hrrr_data.assign_coords(x=x_coords, y=y_coords)
    hrrr_data = hrrr_data.rio.write_crs(crs)

    # Define output grid
    minx, miny, maxx, maxy = task_info.bounds
    width = int((maxx - minx) / target_resolution)
    height = int((maxy - miny) / target_resolution)

    # Create affine transform for target grid
    target_transform = Affine.translation(
        minx, maxy
    ) * Affine.scale(target_resolution, -target_resolution)

    # Reproject and clip
    hrrr_data = hrrr_data.rio.reproject(
        dst_crs=task_info.crs,
        shape=(height, width),
        transform=target_transform,
        resampling=Resampling.bilinear,
    )

    return hrrr_data


def _calculate_rh_from_t_td(t_celsius: np.ndarray, td_celsius: np.ndarray) -> np.ndarray:
    """Calculate relative humidity from temperature and dewpoint temperature.
    
    Uses the Magnus formula approximation for saturation vapor pressure.
    
    Args:
        t_celsius: Temperature in Celsius.
        td_celsius: Dewpoint temperature in Celsius.
    
    Returns:
        Relative humidity as percentage (0-100).
    """
    # Magnus formula constants (for temperatures -40°C to 50°C)
    a = 17.625
    b = 243.04  # °C
    
    # Saturation vapor pressure ratio
    # RH = 100 * exp((a * Td) / (b + Td)) / exp((a * T) / (b + T))
    rh = 100.0 * np.exp((a * td_celsius) / (b + td_celsius)) / np.exp((a * t_celsius) / (b + t_celsius))
    
    # Clamp to valid range
    rh = np.clip(rh, 0, 100)
    
    return rh


# HRRR data availability: AWS archive starts from 2014-09-30
HRRR_ARCHIVE_START_DATE = datetime(2014, 9, 30)


def download_hrrr(
    task_info: TaskInfo,
    herbie_cache_dir: str = DEFAULT_HERBIE_CACHE_DIR,
    delta_hour: int = 1
) -> list[DataWithMetadata]:
    """Download HRRR weather data (humidity and wind) for the fire duration.
    
    Downloads hourly data from NOAA's High-Resolution Rapid Refresh model
    including relative humidity (r2) and wind components (u10, v10).
    
    For older HRRR data where RH:2m is not available, relative humidity is
    calculated from 2m temperature (TMP) and 2m dewpoint temperature (DPT)
    using the Magnus formula.
    
    Args:
        task_info: Task configuration with time range and bounds.
        herbie_cache_dir: Directory for caching downloaded GRIB files.
        delta_hour: Time interval between downloads in hours.
        batch_size: Number of parallel downloads (default: 4, recommended: 4-8).
    
    Returns:
        List of DataWithMetadata for r2, u10, and v10 variables.
    
    Raises:
        ValueError: If no data could be downloaded.
    
    Note:
        Batch size recommendations:
        - 4: Conservative, lower memory usage, stable on slower connections
        - 8: Good balance of speed and reliability
        - 12+: May hit rate limits or cause memory issues
    """
    log.info(f"Downloading HRRR data for event_id: {task_info.event_id}")

    # Check if the fire event is within HRRR data availability
    if task_info.t_end < HRRR_ARCHIVE_START_DATE:
        log.warning(
            f"⚠️ Fire event {task_info.event_id} occurred before HRRR archive start date. "
            f"Event dates: {task_info.t_start.date()} to {task_info.t_end.date()}, "
            f"HRRR archive starts: {HRRR_ARCHIVE_START_DATE.date()}. "
            f"Skipping HRRR download - no data available for this period."
        )
        return []  # Return empty list instead of raising error
    
    if task_info.t_start < HRRR_ARCHIVE_START_DATE:
        log.warning(
            f"⚠️ Fire event {task_info.event_id} starts before HRRR archive availability. "
            f"Adjusting start time from {task_info.t_start} to {HRRR_ARCHIVE_START_DATE}. "
            f"Some early data will be missing."
        )
        effective_t_start = HRRR_ARCHIVE_START_DATE
    else:
        effective_t_start = task_info.t_start

    # Build list of timestamps to download
    timestamps_iter: list[datetime] = []
    current_time = effective_t_start
    while current_time <= task_info.t_end:
        timestamps_iter.append(current_time)
        current_time += timedelta(hours=delta_hour)

    if not timestamps_iter:
        raise ValueError("No time range defined.")

    # Track data gaps
    data_gaps: list[dict] = []

    def fetch_single_timestamp(ts: datetime) -> dict | None:
        """Fetch HRRR data for a single timestamp."""
        try:
            H = Herbie(
                ts,
                model='hrrr',
                product='sfc',
                fxx=0,
                save_dir=herbie_cache_dir,
                verbose=False,  # Disable verbose for parallel execution
            )
            result: dict = {
                'timestamp': ts, 
                'r2': None, 'u10': None, 'v10': None,
                'r2_error': None, 'wind_error': None,
                'r2_source': None,  # Track how RH was obtained
            }
            
            # Try to get RH directly first
            try:
                ds_rh = H.xarray(":RH:2 m", remove_grib=False)
                ds_rh = clip_hrrr_to_task(ds_rh, task_info)
                result['r2'] = ds_rh.r2.values
                result['r2_source'] = 'direct'
            except Exception as e_rh_direct:
                # Fallback: Calculate RH from temperature and dewpoint
                # This is needed for older HRRR data where RH:2m is not available
                try:
                    # Get 2m temperature (in Kelvin)
                    ds_t2m = H.xarray(":TMP:2 m", remove_grib=False)
                    ds_t2m = clip_hrrr_to_task(ds_t2m, task_info)
                    t2m_kelvin = ds_t2m.t2m.values
                    
                    # Get 2m dewpoint temperature (in Kelvin)
                    ds_dpt = H.xarray(":DPT:2 m", remove_grib=False)
                    ds_dpt = clip_hrrr_to_task(ds_dpt, task_info)
                    dpt_kelvin = ds_dpt.d2m.values
                    
                    # Convert Kelvin to Celsius
                    t2m_celsius = t2m_kelvin - 273.15
                    dpt_celsius = dpt_kelvin - 273.15
                    
                    # Calculate RH using Magnus formula
                    result['r2'] = _calculate_rh_from_t_td(t2m_celsius, dpt_celsius)
                    result['r2_source'] = 'calculated_from_t_dpt'
                    log.debug(f"Calculated RH from T/DPT for {ts}")
                    
                except Exception as e_rh_calc:
                    result['r2_error'] = f"Direct: {e_rh_direct}; Calculated: {e_rh_calc}"
            
            try:
                ds_wind = H.xarray(":(?:UGRD|VGRD):10 m", remove_grib=False)
                ds_wind = clip_hrrr_to_task(ds_wind, task_info)
                result['u10'] = ds_wind.u10.values
                result['v10'] = ds_wind.v10.values
            except Exception as e_wind:
                result['wind_error'] = str(e_wind)
            
            return result
            
        except Exception as e:
            return {'timestamp': ts, 'r2': None, 'u10': None, 'v10': None, 
                    'r2_error': str(e), 'wind_error': str(e), 'r2_source': None}

    # Parallel download with progress bar
    batch_size = 8  # Conservative default, 4-8 recommended
    results: list[dict] = []
    
    log.info(f"Downloading {len(timestamps_iter)} HRRR timestamps (batch size: {batch_size})...")
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(fetch_single_timestamp, ts): ts for ts in timestamps_iter}
        
        with tqdm(total=len(timestamps_iter), desc="HRRR Download") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    # Sort results by timestamp to maintain chronological order
    results.sort(key=lambda x: x['timestamp'])
    
    # Identify and log data gaps
    r2_sources = {'direct': 0, 'calculated_from_t_dpt': 0}
    for r in results:
        ts = r['timestamp']
        has_gap = False
        gap_info = {'timestamp': ts.isoformat(), 'missing': []}
        
        if r['r2'] is None:
            gap_info['missing'].append('r2')
            gap_info['r2_error'] = r.get('r2_error', 'Unknown error')
            has_gap = True
        else:
            # Track source of RH data
            source = r.get('r2_source', 'unknown')
            if source in r2_sources:
                r2_sources[source] += 1
                
        if r['u10'] is None:
            gap_info['missing'].append('u10')
            gap_info['missing'].append('v10')
            gap_info['wind_error'] = r.get('wind_error', 'Unknown error')
            has_gap = True
        
        if has_gap:
            data_gaps.append(gap_info)
            log.warning(f"📊 DATA GAP at {ts}: missing {', '.join(gap_info['missing'])}")
    
    # Log RH source statistics
    if r2_sources['calculated_from_t_dpt'] > 0:
        log.info(f"📊 RH Data Sources: {r2_sources['direct']} direct, "
                 f"{r2_sources['calculated_from_t_dpt']} calculated from T/DPT")
    
    # Filter results with at least some data
    valid_results = [r for r in results if r['r2'] is not None or r['u10'] is not None]
    
    if not valid_results:
        raise ValueError("No HRRR data downloaded.")

    # Build output data structures with per-variable timestamps
    # Each variable gets its own timestamps list to handle partial data gaps
    data_buffer: dict[str, dict] = {
        'r2': {
            'data': [r['r2'] for r in valid_results if r['r2'] is not None],
            'timestamps': [r['timestamp'] for r in valid_results if r['r2'] is not None],
        },
        'u10': {
            'data': [r['u10'] for r in valid_results if r['u10'] is not None],
            'timestamps': [r['timestamp'] for r in valid_results if r['u10'] is not None],
        },
        'v10': {
            'data': [r['v10'] for r in valid_results if r['v10'] is not None],
            'timestamps': [r['timestamp'] for r in valid_results if r['v10'] is not None],
        },
    }

    n_total = len(timestamps_iter)
    n_success = len(valid_results)
    n_gaps = len(data_gaps)
    
    log.info(f"HRRR Download Summary: {n_success}/{n_total} timestamps successful, {n_gaps} with data gaps")
    log.info(f"  r2: {len(data_buffer['r2']['data'])} samples, "
             f"u10: {len(data_buffer['u10']['data'])} samples, "
             f"v10: {len(data_buffer['v10']['data'])} samples")

    payload = []
    for var_name, var_info in data_buffer.items():
        if var_info['data']:  # Only add if there's data
            payload.append(DataWithMetadata(
                name=var_name,
                data=var_info['data'],
                timestamps=var_info['timestamps'],
                source="HRRR via Herbie",
                resolution=3000,
                unit="%" if var_name == 'r2' else "m/s",
                note={'data_gaps': data_gaps} if data_gaps else {},
            ))
        else:
            log.warning(f"No data collected for {var_name}")

    return payload


def write_data_gap_log(
    task_info: TaskInfo,
    data_gaps: list[dict],
    output_dir: str = 'output'
) -> None:
    """Write data gap information to a log file.
    
    Args:
        task_info: Task configuration containing event_id.
        data_gaps: List of data gap records.
        output_dir: Base output directory.
    """
    event_id = task_info.event_id
    output_path = os.path.join(output_dir, event_id)
    os.makedirs(output_path, exist_ok=True)
    
    log_path = os.path.join(output_path, "data_gaps.json")
    
    gap_log = {
        'event_id': event_id,
        'generated_at': datetime.now().isoformat(),
        'total_gaps': len(data_gaps),
        'gaps': data_gaps
    }
    
    with open(log_path, 'w') as f:
        json.dump(gap_log, f, indent=2)
    
    log.info(f"Data gap log written to {log_path}")


# =============================================================================
# Building and Land Cover Functions
# =============================================================================

def _format_lat_lon_string(val: int, is_lon: bool) -> str:
    """Format lat/lon integers for GBA tile filenames.
    
    Args:
        val: Latitude or longitude value.
        is_lon: True for longitude, False for latitude.
    
    Returns:
        Formatted string (e.g., -120 -> 'w120', 35 -> 'n35').
    """
    if is_lon:
        prefix = 'e' if val >= 0 else 'w'
        return f"{prefix}{abs(val):03d}"
    else:
        prefix = 'n' if val >= 0 else 's'
        return f"{prefix}{abs(val):02d}"


def _get_gba_tile_ids(bounds: tuple[float, float, float, float]) -> list[str]:
    """Calculate Global Building Atlas tile IDs needed to cover the bounds.
    
    The GBA uses 5x5 degree tiles with naming format: w120_n35_w115_n30
    representing {WestLon}_{NorthLat}_{EastLon}_{SouthLat}.
    
    Args:
        bounds: Bounding box as (minx, miny, maxx, maxy) in EPSG:4326.
    
    Returns:
        List of full GEE asset paths for the required tiles.
    """
    min_x, min_y, max_x, max_y = bounds

    # Align to 5-degree grid
    start_x = math.floor(min_x / 5.0) * 5
    start_y = math.floor(min_y / 5.0) * 5

    tile_paths = []
    base_path = "projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS"

    # Iterate through 5x5 degree grid cells
    curr_x = start_x
    while curr_x < max_x:
        curr_y = start_y
        while curr_y < max_y:
            # Tile corners
            tile_w = int(curr_x)
            tile_s = int(curr_y)
            tile_e = int(curr_x + 5)
            tile_n = int(curr_y + 5)

            # Format tile ID
            part1 = _format_lat_lon_string(tile_w, is_lon=True)
            part2 = _format_lat_lon_string(tile_n, is_lon=False)
            part3 = _format_lat_lon_string(tile_e, is_lon=True)
            part4 = _format_lat_lon_string(tile_s, is_lon=False)

            tile_id = f"{part1}_{part2}_{part3}_{part4}"
            tile_paths.append(f"{base_path}/{tile_id}")

            curr_y += 5
        curr_x += 5

    return tile_paths


def download_building_height(task_info: TaskInfo) -> DataWithMetadata:
    """Download building heights from the Global Building Atlas.
    
    Uses area-weighted averaging when multiple buildings fall within a single
    pixel, where larger footprint buildings contribute more to the average.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataWithMetadata containing building heights in meters.
    
    Raises:
        ValueError: If no tiles could be loaded for the region.
    """
    dataset_name = "building_height"
    log.info(
        f"Downloading {dataset_name} data for event_id: {task_info.event_id}"
    )
    _ensure_ee_initialized()

    roi = ee.Geometry.Rectangle(task_info.bounds, task_info.crs, False)

    transformer = Transformer.from_crs(
        task_info.crs, "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(
        task_info.bounds[0], task_info.bounds[1])
    maxx, maxy = transformer.transform(
        task_info.bounds[2], task_info.bounds[3])
    latlon_bounds = (minx, miny, maxx, maxy)

    # Determine which tiles we need
    tile_paths = _get_gba_tile_ids(latlon_bounds)
    log.info(f"Identified GBA tiles: {tile_paths}")

    # Load and merge tile collections
    collection = None
    for path in tile_paths:
        try:
            col = ee.FeatureCollection(path)
            if collection is None:
                collection = col
            else:
                collection = collection.merge(col)
        except Exception as e:
            log.warning(
                f"Could not load GBA tile: {path}. Error: {e}"
            )

    if collection is None:
        raise ValueError(
            "Could not load any building atlas tiles for the requested region."
        )

    # Filter to ROI and valid buildings
    clipped = collection.filterBounds(roi)
    clipped = clipped.filter(
        ee.Filter.And(
            ee.Filter.neq('height', None),
            ee.Filter.gt('height', 3.971)
        )
    )

    count = clipped.size().getInfo()
    log.info(f"Buildings in region: {count}")

    # Compute area-weighted average height
    def add_weighted_height(feature):
        """Add height*area and area properties for weighted averaging."""
        height = ee.Number(feature.get('height'))
        area = feature.geometry().area()
        return feature.set(
            'height_x_area', height.multiply(area)
        ).set('footprint_area', area)

    clipped = clipped.map(add_weighted_height)

    # Sum of (height * area) per pixel
    height_x_area_raster = clipped.reduceToImage(
        properties=["height_x_area"],
        reducer=ee.Reducer.sum()
    )

    # Sum of area per pixel
    area_raster = clipped.reduceToImage(
        properties=["footprint_area"],
        reducer=ee.Reducer.sum()
    )

    # Weighted average: sum(height * area) / sum(area)
    height_raster = height_x_area_raster.divide(
        area_raster
    ).unmask(0).rename(dataset_name)

    # Download
    data_array = _download_processed_image(
        height_raster, task_info, band_name=dataset_name
    )

    return DataWithMetadata(
        name=dataset_name,
        data=[data_array],
        timestamps=[task_info.t_start],
        source=f"Global Building Atlas (Tiles: {len(tile_paths)})",
        resolution=3,
        unit="m",
        note={'aggregation': 'area-weighted average'},
    )


def download_eca(task_info: TaskInfo) -> DataWithMetadata:
    """Download ESA WorldCover land cover classification.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataWithMetadata containing land cover classes (int16).
    """
    log.info(f"Downloading ESA WorldCover data for event_id: {task_info.event_id}")
    data = download_gee_task(
        task_info,
        dataset_name="landcover",
        band="Map",
        imagecollection="ESA/WorldCover/v200",
        resample='nearest',
    )
    return replace(
        data,
        data=[np.around(data.data[0]).astype(np.int16)],
        resolution=10,
        unit="class",
        note=data.note | {'mapping': {
            0: "No Data",
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare / sparse vegetation",
            70: "Snow and ice",
            80: "Permanent water bodies",
            90: "Herbaceous wetland",
            95: "Mangroves",
            100: "Moss and lichen",
        }}
    )


def download_tc(task_info: TaskInfo) -> DataWithMetadata:
    """Download Tree Canopy Leaf Area Index (LAI) data.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataWithMetadata containing LAI values in m²/m².
    """
    log.info(f"Downloading LAI data for event_id: {task_info.event_id}")

    # LAI tile asset paths
    ASSET_ROOT = 'projects/tc-global-urban/assets/'
    FILENAMES = [
        'LAI_Grid_30deg_101_2020-07-02', 'LAI_Grid_30deg_102_2020-07-02', 'LAI_Grid_30deg_103_2020-07-02',
        'LAI_Grid_30deg_104_2020-07-02', 'LAI_Grid_30deg_105_2020-07-02', 'LAI_Grid_30deg_107_2020-07-02',
        'LAI_Grid_30deg_108_2020-07-02', 'LAI_Grid_30deg_0_2020-07-02',   'LAI_Grid_30deg_1_2020-07-02',
        'LAI_Grid_30deg_2_2020-07-02',   'LAI_Grid_30deg_3_2020-07-02',   'LAI_Grid_30deg_4_2020-07-02',
        'LAI_Grid_30deg_5_2020-07-02',   'LAI_Grid_30deg_6_2020-07-02',   'LAI_Grid_30deg_7_2020-07-02',
        'LAI_Grid_30deg_8_2020-07-02',   'LAI_Grid_30deg_9_2020-07-02',   'LAI_Grid_30deg_10_2020-07-02',
        'LAI_Grid_30deg_11NE_2020-07-02', 'LAI_Grid_30deg_11NW_2020-07-02', 'LAI_Grid_30deg_11SE_2020-07-02',
        'LAI_Grid_30deg_11SW_2020-07-02', 'LAI_Grid_30deg_12_2020-07-02',  'LAI_Grid_30deg_13_2020-07-02',
        'LAI_Grid_30deg_14_2020-07-02',  'LAI_Grid_30deg_15_2020-07-02',  'LAI_Grid_30deg_16NE_2020-07-02',
        'LAI_Grid_30deg_16NW_2020-07-02', 'LAI_Grid_30deg_16SE_2020-07-02', 'LAI_Grid_30deg_16SW_2020-07-02',
        'LAI_Grid_30deg_17_2020-07-02',  'LAI_Grid_30deg_18_2020-07-02',  'LAI_Grid_30deg_19_2020-07-02',
        'LAI_Grid_30deg_20_2020-07-02',  'LAI_Grid_30deg_21NE_2020-07-02', 'LAI_Grid_30deg_21NW_2020-07-02',
        'LAI_Grid_30deg_21SE_2020-07-02', 'LAI_Grid_30deg_21SW_2020-07-02', 'LAI_Grid_30deg_22_2020-07-02',
        'LAI_Grid_30deg_23_2020-07-02',  'LAI_Grid_30deg_24_2020-07-02',  'LAI_Grid_30deg_25_2020-07-02',
        'LAI_Grid_30deg_26_2020-07-02',  'LAI_Grid_30deg_27_2020-07-02',  'LAI_Grid_30deg_28_2020-07-02',
        'LAI_Grid_30deg_29_2020-07-02',  'LAI_Grid_30deg_30_2020-07-02',  'LAI_Grid_30deg_31_2020-07-02',
        'LAI_Grid_30deg_32_2020-07-02',  'LAI_Grid_30deg_33_2020-07-02',  'LAI_Grid_30deg_34_2020-07-02',
        'LAI_Grid_30deg_35_2020-07-02',  'LAI_Grid_30deg_36_2020-07-02',  'LAI_Grid_30deg_37_2020-07-02',
        'LAI_Grid_30deg_38_2020-07-02',  'LAI_Grid_30deg_39_2020-07-02',  'LAI_Grid_30deg_40_2020-07-02',
        'LAI_Grid_30deg_41_2020-07-02',  'LAI_Grid_30deg_42_2020-07-02',  'LAI_Grid_30deg_43_2020-07-02',
        'LAI_Grid_30deg_44_2020-07-02',  'LAI_Grid_30deg_45_2020-07-02',  'LAI_Grid_30deg_46_2020-07-02',
        'LAI_Grid_30deg_47_2020-07-02',  'LAI_Grid_30deg_48_2020-07-02',  'LAI_Grid_30deg_49_2020-07-02',
        'LAI_Grid_30deg_50_2020-07-02',  'LAI_Grid_30deg_51_2020-07-02',  'LAI_Grid_30deg_52_2020-07-02',
        'LAI_Grid_30deg_53_2020-07-02',  'LAI_Grid_30deg_54_2020-07-02',  'LAI_Grid_30deg_55_2020-07-02',
        'LAI_Grid_30deg_56_2020-07-02',  'LAI_Grid_30deg_101_2020-07-02', 'LAI_Grid_30deg_102_2020-07-02',
        'LAI_Grid_30deg_103_2020-07-02', 'LAI_Grid_30deg_104_2020-07-02', 'LAI_Grid_30deg_105_2020-07-02',
        'LAI_Grid_30deg_107_2020-07-02', 'LAI_Grid_30deg_108_2020-07-02'
    ]

    def prepare_lai_image(filename: str) -> ee.Image:
        """Load LAI image and normalize band name."""
        return ee.Image(ASSET_ROOT + filename).select([0]).rename('lai')

    image_list = [prepare_lai_image(name) for name in FILENAMES]

    # Call Generic Downloader
    data = download_gee_task(
        task_info=task_info,
        dataset_name="lai",
        imagecollection=image_list,
        band="lai",
        resample='bilinear'
    )

    return replace(
        data,
        data=[data_array.astype(np.float32) for data_array in data.data],
        resolution=10,
        unit="m2/m2",
    )


# =============================================================================
# Satellite Imagery Functions
# =============================================================================

def download_satellite(task_info: TaskInfo) -> DataWithMetadata:
    """Download satellite imagery (RGB) from NAIP or Sentinel-2.
    
    Attempts to download high-resolution NAIP imagery (1m) for US locations.
    Falls back to Sentinel-2 (10m) if NAIP is not available.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataWithMetadata containing RGB satellite imagery as (H, W, 3) array.
    """
    log.info(f"Downloading satellite imagery for event_id: {task_info.event_id}")
    _ensure_ee_initialized()
    
    roi = ee.Geometry.Rectangle(task_info.bounds, task_info.crs, False)
    
    # Try NAIP first (US only, 1m resolution)
    try:
        # Get NAIP imagery around the fire start date
        year = task_info.year
        naip = ee.ImageCollection("USDA/NAIP/DOQQ") \
            .filterBounds(roi) \
            .filter(ee.Filter.calendarRange(year - 1, year + 1, 'year')) \
            .select(['R', 'G', 'B'])
        
        if naip.size().getInfo() > 0:
            log.info("Using NAIP imagery (1m resolution)")
            image = naip.mosaic()
            source = "USDA NAIP"
            native_res = 1
        else:
            raise ValueError("No NAIP imagery available")
            
    except Exception as e:
        log.info(f"NAIP not available ({e}), falling back to Sentinel-2")
        
        # Fall back to Sentinel-2 (global, 10m resolution)
        # Get cloud-free imagery around fire start date
        start_date = task_info.t_start.strftime('%Y-%m-%d')
        end_date = (task_info.t_start + timedelta(days=180)).strftime('%Y-%m-%d')
        
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .select(['B4', 'B3', 'B2'])  # RGB bands
        
        if s2.size().getInfo() == 0:
            # Try a wider date range
            start_date = f"{task_info.year}-01-01"
            end_date = f"{task_info.year}-12-31"
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterBounds(roi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                .select(['B4', 'B3', 'B2'])
        
        image = s2.median().rename(['R', 'G', 'B'])
        # Scale Sentinel-2 values to 0-255 range
        image = image.divide(10000).multiply(255).clamp(0, 255)
        source = "Copernicus Sentinel-2"
        native_res = 10
    
    # Download RGB bands
    rgb_data = []
    for band in ['R', 'G', 'B']:
        band_image = image.select(band).reproject(
            crs=task_info.crs,
            scale=task_info.resolution
        )
        data_array = _download_processed_image(band_image, task_info, band)
        rgb_data.append(data_array)
    
    # Stack into (H, W, 3) array
    rgb_array = np.stack(rgb_data, axis=-1).astype(np.uint8)
    
    log.info(f"✓ Downloaded satellite imagery: {rgb_array.shape}")
    
    return DataWithMetadata(
        name="satellite",
        data=[rgb_array],
        timestamps=[task_info.t_start],
        source=source,
        resolution=native_res,
        unit="RGB (0-255)",
        note={'description': 'True color satellite imagery'},
    )


# =============================================================================
# Global WUI (Wildland-Urban Interface) Functions
# =============================================================================

# Global WUI class definitions
GLOBAL_WUI_CLASSES = {
    1: "Forest/Shrub/Wetland Intermix WUI",
    2: "Forest/Shrub/Wetland Interface WUI",
    3: "Grassland Intermix WUI",
    4: "Grassland Interface WUI",
    5: "Non-WUI: Forest/Shrub/Wetland",
    6: "Non-WUI: Grassland",
    7: "Non-WUI: Urban",
    8: "Non-WUI: Other",
}


def _get_equi7_tile_params() -> dict:
    """Get EQUI7 grid parameters for North America tile system.
    
    The Global WUI data uses the EQUI7 Azimuthal Equidistant projection
    with 100km x 100km tiles.
    
    Returns:
        Dictionary with projection and tile parameters.
    """
    return {
        'crs': 'PROJCS["Azimuthal_Equidistant",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Azimuthal_Equidistant"],PARAMETER["false_easting",8264722.17686],PARAMETER["false_northing",4867518.35323],PARAMETER["longitude_of_center",-97.5],PARAMETER["latitude_of_center",52.0],UNIT["Meter",1.0]]',
        'tile_size': 100000,  # 100km tiles
        'origin_x': 0,
        'origin_y': 9900000,
    }


def _get_equi7_tiles_for_bounds(
    bounds: tuple[float, float, float, float],
    source_crs: str,
) -> list[str]:
    """Calculate EQUI7 tile IDs needed to cover the given bounds.
    
    Args:
        bounds: Bounding box (minx, miny, maxx, maxy) in source_crs.
        source_crs: CRS of the input bounds.
    
    Returns:
        List of tile IDs in format 'X0065_Y0040'.
    """
    equi7_params = _get_equi7_tile_params()
    
    # Transform bounds to EQUI7 projection
    transformer = Transformer.from_crs(source_crs, equi7_params['crs'], always_xy=True)
    
    # Transform all corners to handle projection distortion
    minx, miny, maxx, maxy = bounds
    corners = [
        (minx, miny), (minx, maxy),
        (maxx, miny), (maxx, maxy),
    ]
    
    transformed_x = []
    transformed_y = []
    for x, y in corners:
        tx, ty = transformer.transform(x, y)
        transformed_x.append(tx)
        transformed_y.append(ty)
    
    # Get bounding box in EQUI7 coordinates
    equi7_minx = min(transformed_x)
    equi7_miny = min(transformed_y)
    equi7_maxx = max(transformed_x)
    equi7_maxy = max(transformed_y)
    
    # Calculate tile indices
    tile_size = equi7_params['tile_size']
    origin_x = equi7_params['origin_x']
    origin_y = equi7_params['origin_y']
    
    # Tile index calculation: tile_x = floor(x / tile_size), tile_y = floor((origin_y - y) / tile_size)
    # Note: Y increases downward in the tile naming scheme
    min_tile_x = int(math.floor(equi7_minx / tile_size))
    max_tile_x = int(math.floor(equi7_maxx / tile_size))
    min_tile_y = int(math.floor((origin_y - equi7_maxy) / tile_size))
    max_tile_y = int(math.floor((origin_y - equi7_miny) / tile_size))
    
    tiles = []
    for tx in range(min_tile_x, max_tile_x + 1):
        for ty in range(min_tile_y, max_tile_y + 1):
            tile_id = f"X{tx:04d}_Y{ty:04d}"
            tiles.append(tile_id)
    
    return tiles


def download_globalwui(
    task_info: TaskInfo,
    base_dir: str = 'datasets/globalwui'
) -> DataWithMetadata:
    """Download Global Wildland-Urban Interface (WUI) data.
    
    The Global WUI dataset maps the interface between human settlements and
    wildland vegetation at 10m resolution. Data is organized in EQUI7 tiles.
    
    Reference:
        Schug, F. et al. (2023). The global wildland–urban interface.
        Nature. https://doi.org/10.1038/s41586-023-06320-0
    
    Args:
        task_info: Task configuration with bounds and resolution.
        base_dir: Base directory containing the Global WUI data tiles.
    
    Returns:
        DataWithMetadata containing WUI class values (uint8, 1-8).
    
    Raises:
        FileNotFoundError: If required tiles are not found.
    """
    log.info(f"Processing Global WUI data for event_id: {task_info.event_id}")
    
    # Get tiles that intersect with task bounds
    tiles_needed = _get_equi7_tiles_for_bounds(task_info.bounds, task_info.crs)
    log.info(f"Required EQUI7 tiles: {tiles_needed}")
    
    # Load and merge tiles using rioxarray
    tile_datasets = []
    missing_tiles = []
    
    for tile_id in tiles_needed:
        tile_path = os.path.join(base_dir, tile_id, 'WUI.tif')
        if os.path.exists(tile_path):
            try:
                ds = rioxarray.open_rasterio(tile_path)
                tile_datasets.append(ds)
                log.debug(f"Loaded tile: {tile_id}")
            except Exception as e:
                log.warning(f"Error loading tile {tile_id}: {e}")
                missing_tiles.append(tile_id)
        else:
            log.warning(f"Tile not found: {tile_path}")
            missing_tiles.append(tile_id)
    
    if not tile_datasets:
        raise FileNotFoundError(
            f"No Global WUI tiles found for region. "
            f"Expected tiles: {tiles_needed}. "
            f"Please download the required tiles from the Global WUI dataset."
        )
    
    if missing_tiles:
        log.warning(f"Missing tiles (may be outside WUI coverage): {missing_tiles}")
    
    # Merge tiles if multiple
    if len(tile_datasets) == 1:
        merged = tile_datasets[0]
    else:
        from rioxarray.merge import merge_arrays
        merged = merge_arrays(tile_datasets)
    
    # Reproject to task CRS and clip to bounds
    t_minx, t_miny, t_maxx, t_maxy = task_info.bounds
    
    # Create the target transform
    target_transform = from_origin(
        t_minx, t_maxy,
        task_info.resolution, task_info.resolution
    )
    
    # Reproject and resample
    reprojected = merged.rio.reproject(
        dst_crs=task_info.crs,
        shape=task_info.shape,
        transform=target_transform,
        resampling=Resampling.nearest,  # Use nearest neighbor for categorical data
    )
    
    # Extract data array
    data_array = reprojected.values
    
    # Handle band dimension if present
    if data_array.ndim == 3:
        data_array = data_array[0]  # Take first band
    
    assert data_array.shape == task_info.shape, (
        f"Shape mismatch: got {data_array.shape}, expected {task_info.shape}"
    )
    
    # Convert to uint8
    data_array = data_array.astype(np.uint8)
    
    log.info(f"✓ Processed Global WUI data: {data_array.shape}, "
             f"unique classes: {np.unique(data_array).tolist()}")
    
    return DataWithMetadata(
        name="wui",
        data=[data_array],
        timestamps=[task_info.t_start],
        source="Global WUI; Schug et al. 2023; https://doi.org/10.1038/s41586-023-06320-0",
        resolution=10,
        unit="class",
        note={
            'mapping': GLOBAL_WUI_CLASSES,
            'description': 'Wildland-Urban Interface classification',
            'temporal_coverage': 'ca. 2020',
            'missing_tiles': missing_tiles if missing_tiles else None,
        }
    )


# =============================================================================
# Hillshade (Terrain Visualization) Functions
# =============================================================================

def download_hillshade(task_info: TaskInfo) -> DataWithMetadata:
    """Download hillshade terrain visualization from Google Earth Engine.
    
    Computes hillshade from elevation data to create a terrain visualization
    similar to Google Maps terrain view. Hillshade simulates illumination
    from a light source to show terrain relief.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataWithMetadata containing hillshade values (0-255).
    """
    log.info(f"Downloading hillshade for event_id: {task_info.event_id}")
    _ensure_ee_initialized()
    
    roi = ee.Geometry.Rectangle(task_info.bounds, task_info.crs, False)
    
    # Get elevation data
    collection = ee.ImageCollection("USGS/3DEP/1m").filterBounds(roi)
    
    if collection.size().getInfo() == 0:
        # Fallback to SRTM if 3DEP not available
        log.info("3DEP not available, falling back to SRTM 30m")
        elevation = ee.Image("USGS/SRTMGL1_003").select('elevation')
        native_res = 30
    else:
        native_proj = collection.first().select('elevation').projection()
        elevation = collection.mosaic().select('elevation')
        elevation = elevation.setDefaultProjection(native_proj)
        native_res = 1
    
    # Compute hillshade using ee.Terrain.hillshade
    # Default azimuth=315 (NW), elevation=45 degrees - standard cartographic lighting
    hillshade = ee.Terrain.hillshade(elevation, azimuth=315, elevation=45)
    hillshade = hillshade.resample('bilinear').rename('hillshade')
    
    data_array = _download_processed_image(hillshade, task_info, 'hillshade')
    
    log.info("✓ Downloaded hillshade data")
    
    return DataWithMetadata(
        name="hillshade",
        data=[data_array.astype(np.uint8)],
        timestamps=[task_info.t_start],
        source="Computed from USGS 3DEP/SRTM via Google Earth Engine",
        resolution=native_res,
        unit="0-255",
        note={'description': 'Terrain hillshade visualization (azimuth=315°, elevation=45°)'},
    )


# =============================================================================
# Single Fire Processing
# =============================================================================

def process_single_fire(
    event_id: str,
    args: ProcessingArgs,
) -> dict[str, Any]:
    """Process a single fire event and download all associated data.
    
    This is the main processing function for a single fire event. It:
    1. Loads fire information and creates task configuration
    2. Processes FEDS25MTBS perimeter data
    3. Processes FRP (day and night)
    4. Downloads GEE-based datasets in parallel
    5. Downloads HRRR weather data
    
    Args:
        event_id: Unique identifier for the fire event.
        args: Processing configuration parameters.
    
    Returns:
        Dictionary with dataset names as keys and success status (bool) as values.
        On error, contains 'error' key with False and 'message' with error details.
    """
    results_status: dict[str, Any] = {}
    
    # Helper function to check if a feature should be processed
    def should_process(feature_name: str) -> bool:
        """Check if a feature should be processed based on --only filter."""
        if args.only is None:
            return True  # Process all features if no filter specified
        return feature_name in args.only
    
    try:
        # Load fire information
        fire_info = get_fire_info(event_id)
        log.info(f"Retrieved fire info for event_id: {event_id}")
        log.debug(f"Generated fire info: {fire_info}")

        task_info = get_task_info(
            fire_info, resolution=args.resolution, buffer=args.buffer, crs=args.crs)

        log.debug(f"Generated task info: {task_info}")
        save_numpy(task_info, DataWithMetadata(
            name="task_info", data=[task_info]), args.output_dir)
        
        log.info(f"Processing event: {task_info.event_id}")
        log.info(f"  Date range: {task_info.t_start} to {task_info.t_end}")
        log.info(f"  Resolution: {task_info.resolution}m, Shape: {task_info.shape}")
        log.info(f"  CRS: {task_info.crs}, Bounds: {task_info.bounds}")
        if args.only:
            log.info(f"  Only processing: {args.only}")
        
        # Process FEDS25MTBS (always needed for FRP, so process if any FRP is requested)
        needs_perimeters = should_process("burn_perimeters") or should_process("frp_day") or should_process("frp_night")
        feds25mtbs = None
        
        if needs_perimeters:
            feds25mtbs = process_feds25mtbs(task_info)
            log.debug(f"Processed FEDS25MTBS data: {feds25mtbs}")
            
            if args.interpolation > 0:
                log.info(f"Interpolating burn perimeters with multiplier: {args.interpolation}")
                feds25mtbs = interpolate_burn_perimeters(feds25mtbs, multiplier=args.interpolation)
                log.debug(f"Interpolated FEDS25MTBS data: {feds25mtbs}")
            
            if should_process("burn_perimeters"):
                save_numpy(task_info, feds25mtbs, args.output_dir)
                results_status["burn_perimeters"] = True
        else:
            log.info("⏭️ Skipping burn_perimeters (not in --only)")

        # Process FRP
        if should_process("frp_day"):
            log.info("Processing FRP (Fire Radiative Power) day data...")
            frp_day = process_frp_day(task_info, feds25mtbs)
            save_numpy(task_info, frp_day, args.output_dir)
            log.info(f"✓ Saved frp_day.npy ({len(frp_day.data)} time steps)")
            results_status["frp_day"] = True
        else:
            log.info("⏭️ Skipping frp_day (not in --only)")
        
        if should_process("frp_night"):
            log.info("Processing FRP (Fire Radiative Power) night data...")
            frp_night = process_frp_night(task_info, feds25mtbs)
            save_numpy(task_info, frp_night, args.output_dir)
            log.info(f"✓ Saved frp_night.npy ({len(frp_night.data)} time steps)")
            results_status["frp_night"] = True
        else:
            log.info("⏭️ Skipping frp_night (not in --only)")

        # Parallel Downloads (GEE-based datasets)
        all_gee_tasks: list[tuple[str, Callable, tuple]] = [
            ("elevation", download_usgs, (task_info,)),
            ("landfire", download_landfire, (task_info,)),
            ("building_height", download_building_height, (task_info,)),
            ("landcover", download_eca, (task_info,)),
            ("lai", download_tc, (task_info,)),
            ("satellite", download_satellite, (task_info,)),
            ("hillshade", download_hillshade, (task_info,)),
            ("wui", download_globalwui, (task_info,)),
        ]
        
        # Filter to only requested features
        download_tasks = [(name, func, func_args) 
                          for name, func, func_args in all_gee_tasks 
                          if should_process(name)]
        
        if download_tasks:
            log.info(f"Starting parallel downloads for GEE datasets: {[t[0] for t in download_tasks]}")
            
            results: dict[str, DataWithMetadata | list[DataWithMetadata] | Exception] = {}
            max_retries = 3
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_name = {
                    executor.submit(func, *func_args): name
                    for name, func, func_args in download_tasks
                }
                
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        results[name] = future.result()
                        log.info(f"✓ Completed: {name}")
                    except Exception as e:
                        log.error(f"✗ Failed: {name} - {e}")
                        results[name] = e
            
            # Retry failed tasks
            failed_tasks = [(name, func, func_args) 
                            for name, func, func_args in download_tasks 
                            if isinstance(results.get(name), Exception)]
            
            for retry in range(max_retries):
                if not failed_tasks:
                    break
                
                log.info(f"Retrying {len(failed_tasks)} failed task(s) (attempt {retry + 1}/{max_retries})...")
                still_failed = []
                
                for name, func, func_args in failed_tasks:
                    try:
                        results[name] = func(*func_args)
                        log.info(f"✓ Retry succeeded: {name}")
                    except Exception as e:
                        log.error(f"✗ Retry failed: {name} - {e}")
                        results[name] = e
                        still_failed.append((name, func, func_args))
                
                failed_tasks = still_failed
            
            # Save successful results
            for name in ["elevation", "building_height", "landcover", "lai", "satellite", "hillshade", "wui"]:
                if name in results and not isinstance(results[name], Exception):
                    data = results[name]
                    log.debug(f"Downloaded {name} data: {data}")
                    save_numpy(task_info, data, args.output_dir)
                    results_status[name] = True
                elif name in results:
                    results_status[name] = False
            
            # Handle landfire (returns list)
            if "landfire" in results and not isinstance(results["landfire"], Exception):
                for lf_data in results["landfire"]:
                    log.debug(f"Downloaded LANDFIRE data: {lf_data}")
                    save_numpy(task_info, lf_data, args.output_dir)
                results_status["landfire"] = True
            elif "landfire" in results:
                results_status["landfire"] = False
        else:
            log.info("⏭️ Skipping all GEE datasets (not in --only)")

        # Sequential Download (HRRR)
        if should_process("hrrr"):
            hrrr = download_hrrr(task_info, args.herbie_cache_dir)

            if hrrr:
                for hrrr_data in hrrr:
                    log.debug(f"Downloaded HRRR data: {hrrr_data}")
                    save_numpy(task_info, hrrr_data, args.output_dir)
                
                if hrrr[0].note and hrrr[0].note.get('data_gaps'):
                    write_data_gap_log(task_info, hrrr[0].note['data_gaps'], args.output_dir)
                results_status["hrrr"] = True
            else:
                log.warning(f"⚠️ No HRRR data available for event {task_info.event_id}")
                results_status["hrrr"] = False
        else:
            log.info("⏭️ Skipping hrrr (not in --only)")
        
        # Report status
        failed = [name for name, success in results_status.items() if not success]
        if failed:
            log.warning(f"Some downloads failed for {event_id}: {failed}")
        else:
            log.info(f"✓ Successfully processed all data for {event_id}")
        
        return results_status
        
    except Exception as e:
        log.error(f"Failed to process event {event_id}: {e}")
        return {"error": False, "message": str(e)}


# =============================================================================
# Batch Processing
# =============================================================================

def parse_batch_input(batch_input: str) -> list[str]:
    """Parse batch input which can be a file path or comma-separated event IDs.
    
    Args:
        batch_input: Either a path to a file containing event IDs (one per line)
                     or a comma-separated string of event IDs.
    
    Returns:
        List of event IDs to process.
    """
    # Check if it's a file
    if os.path.isfile(batch_input):
        log.info(f"Reading event IDs from file: {batch_input}")
        with open(batch_input, 'r') as f:
            event_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        log.info(f"Found {len(event_ids)} event IDs in file")
        return event_ids
    
    # Otherwise treat as comma-separated
    event_ids = [eid.strip() for eid in batch_input.split(',') if eid.strip()]
    log.info(f"Parsed {len(event_ids)} event IDs from input")
    return event_ids


def process_batch(
    event_ids: list[str],
    args: ProcessingArgs,
    max_workers: int = 2,
) -> dict[str, dict[str, Any]]:
    """Process multiple fire events in parallel.
    
    Processes multiple fire events concurrently while sharing cached data
    (FIRMS, firepix) across all workers. Each event is processed independently.
    
    Args:
        event_ids: List of event IDs to process.
        args: Processing configuration parameters.
        max_workers: Maximum number of parallel fire event processors.
                     Note: Each event also runs internal parallel downloads,
                     so keep this value moderate (2-4 recommended).
    
    Returns:
        Dictionary mapping event IDs to their processing results.
    """
    log.info(f"Starting batch processing of {len(event_ids)} fire events")
    log.info(f"Using {max_workers} parallel workers")
    
    # Initialize Earth Engine once before parallel processing
    log.info("Initializing Google Earth Engine...")
    _ensure_ee_initialized()
    
    all_results: dict[str, dict[str, Any]] = {}
    successful = 0
    failed = 0
    
    def process_with_logging(event_id: str) -> tuple[str, dict[str, Any]]:
        """Wrapper to process a single fire with proper logging context."""
        log.info(f"▶ Starting processing: {event_id}")
        try:
            result = process_single_fire(event_id, args)
            return event_id, result
        except Exception as e:
            log.error(f"✗ Fatal error processing {event_id}: {e}")
            return event_id, {"error": False, "message": str(e)}
    
    # Process events in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_with_logging, eid): eid for eid in event_ids}
        
        for future in tqdm(as_completed(futures), total=len(event_ids), desc="Batch Progress"):
            event_id = futures[future]
            try:
                eid, result = future.result()
                all_results[eid] = result
                
                if "error" not in result:
                    successful += 1
                    log.info(f"✓ Completed: {eid}")
                else:
                    failed += 1
                    log.error(f"✗ Failed: {eid} - {result.get('message', 'Unknown error')}")
            except Exception as e:
                failed += 1
                all_results[event_id] = {"error": False, "message": str(e)}
                log.error(f"✗ Exception for {event_id}: {e}")
    
    # Summary
    log.info("=" * 60)
    log.info("BATCH PROCESSING COMPLETE")
    log.info(f"  Total events: {len(event_ids)}")
    log.info(f"  Successful: {successful}")
    log.info(f"  Failed: {failed}")
    log.info("=" * 60)
    
    # Write batch summary
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    summary = {
        "total": len(event_ids),
        "successful": successful,
        "failed": failed,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Batch summary saved to {summary_path}")
    
    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for the Fire Data Loader.
    
    Parses command-line arguments and orchestrates the data download and
    processing pipeline for single or batch fire event processing.
    """
    parser = argparse.ArgumentParser(
        description="Fire Data Loader - Download and process wildfire geospatial data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mutually exclusive: single event_id or batch mode
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "event_id",
        type=str,
        nargs="?",
        help="Single event ID to process"
    )
    input_group.add_argument(
        "--batch",
        type=str,
        help="Batch mode: file path with event IDs (one per line) or comma-separated event IDs"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers for batch processing (recommended: 1-4)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=30,
        help="Spatial resolution in meters"
    )
    parser.add_argument(
        "--buffer", "-b",
        type=int,
        default=100,
        help="Buffer distance around fire bounds in meters"
    )
    parser.add_argument(
        "--crs", "-c",
        type=str,
        default="EPSG:5070",
        help="Target coordinate reference system"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="output",
        help="Output directory for saved data"
    )
    parser.add_argument(
        "--interpolation", "-t",
        type=int,
        default=0,
        help="Number of intermediate frames to interpolate between perimeter timesteps"
    )
    parser.add_argument(
        "--herbie_cache_dir",
        type=str,
        default=DEFAULT_HERBIE_CACHE_DIR,
        help="Directory for caching HRRR GRIB files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only process specific feature(s). Comma-separated list. "
             "Available: burn_perimeters, frp_day, frp_night, elevation, landfire, "
             "building_height, landcover, lai, satellite, hillshade, wui, hrrr"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create ProcessingArgs from command line arguments
    only_features = None
    if args.only:
        only_features = [f.strip() for f in args.only.split(',') if f.strip()]
        log.info(f"Processing only: {only_features}")
    
    processing_args = ProcessingArgs(
        resolution=args.resolution,
        buffer=args.buffer,
        crs=args.crs,
        output_dir=args.output_dir,
        interpolation=args.interpolation,
        herbie_cache_dir=args.herbie_cache_dir,
        verbose=args.verbose,
        only=only_features,
    )

    # Batch mode or single mode
    if args.batch:
        event_ids = parse_batch_input(args.batch)
        if not event_ids:
            log.error("No valid event IDs found in batch input")
            return
        
        # Initialize Earth Engine before batch processing
        log.info("Initializing Google Earth Engine...")
        _ensure_ee_initialized()
        
        process_batch(event_ids, processing_args, max_workers=args.workers)
    else:
        # Single event mode
        event_id = args.event_id
        if not event_id:
            parser.error("Either event_id or --batch is required")
        
        # Initialize Earth Engine
        log.info("Initializing Google Earth Engine...")
        _ensure_ee_initialized()
        
        process_single_fire(event_id, processing_args)


# =============================================================================
# Utility Functions
# =============================================================================

def plot(data: np.ndarray, *args, **kwargs) -> None:
    """Quick visualization of a 2D numpy array using matplotlib.
    
    Args:
        data: 2D array to visualize.
        *args: Additional positional arguments for imshow.
        **kwargs: Additional keyword arguments for imshow.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap='viridis', interpolation='nearest', *args, **kwargs)
    plt.colorbar(label='Value')
    plt.show()


if __name__ == "__main__":
    main()
