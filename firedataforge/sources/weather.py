"""HRRR near-surface weather via Herbie (AWS), clipped to the task grid."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor used by clip/reproject)
import xarray as xr
from affine import Affine
from herbie import Herbie
from pyproj import Transformer
from rasterio.enums import Resampling
from tqdm import tqdm

from firedataforge.constants import CACHE_DIR, HERBIE_CACHE_NAME
from firedataforge.schemas import DataLayer, ProcessingTask

xr.set_options(use_new_combine_kwarg_defaults=True)
log = logging.getLogger(__name__)

# HRRR is clipped/reprojected to approximately this resolution (m) rather than the
# 30 m task grid. HRRR's native grid is ~3 km, so resampling an hourly time series
# to 30 m would inflate file size ~250x without adding spatial information. The
# weather layers therefore sit on their own coarser grid, spanning exactly the same
# bounds as the task grid; Each weather layer
# records the nominal value in ``current_resolution`` and the exact per-axis cell
# size in ``note['grid']``.
HRRR_OUTPUT_RESOLUTION = 500


def clip_hrrr_to_task(
    hrrr_data: xr.Dataset,
    task_info: ProcessingTask,
    target_resolution: int = HRRR_OUTPUT_RESOLUTION
) -> xr.Dataset:
    """Clip and reproject HRRR data to match task bounds.

    Transforms HRRR data from its native Lambert Conformal Conic projection
    to the task CRS and clips to the task bounds.

    The output grid spans exactly ``task_info.bounds``: the cell count is
    ``target_resolution`` rounded to whole cells and the cell size is then the
    extent divided by that count, so ``target_resolution`` is nominal (typically
    within a few percent) and the grid tiles the bounds without remainder.

    Args:
        hrrr_data: HRRR xarray Dataset from Herbie.
        task_info: Task configuration with bounds and CRS.
        target_resolution: Nominal output resolution in meters.

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

    # Define output grid.
    minx, miny, maxx, maxy = task_info.bounds
    width = max(1, round((maxx - minx) / target_resolution))
    height = max(1, round((maxy - miny) / target_resolution))
    pixel_x = (maxx - minx) / width
    pixel_y = (maxy - miny) / height

    # Create affine transform for target grid
    target_transform = Affine.translation(
        minx, maxy
    ) * Affine.scale(pixel_x, -pixel_y)

    # Reproject and clip
    hrrr_data = hrrr_data.rio.reproject(
        dst_crs=task_info.crs,
        shape=(height, width),
        transform=target_transform,
        resampling=Resampling.bilinear,
    )

    return hrrr_data


def _weather_grid_note(task_info: ProcessingTask, shape: tuple[int, int]) -> dict:
    """Exact cell size of the weather grid, recorded alongside the nominal one.

    Derived the same way a consumer must -- from the shared task bounds and the
    array's own shape (see :class:`firedataforge.schemas.GeoReference`) -- so writing it out
    doubles as an in-code statement of that contract.
    """
    minx, miny, maxx, maxy = task_info.bounds
    height, width = shape
    return {
        'nominal_resolution_m': HRRR_OUTPUT_RESOLUTION,
        'pixel_size_x_m': (maxx - minx) / width,
        'pixel_size_y_m': (maxy - miny) / height,
        'shape': [int(height), int(width)],
    }


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


HRRR_ARCHIVE_START_DATE = datetime(2014, 9, 30)


def download_hrrr(
    task_info: ProcessingTask,
    cache_dir: str = CACHE_DIR,
    delta_hour: int = 1,
    batch_size: int = 4,
) -> list[DataLayer]:
    """Download HRRR weather data (humidity and wind) for the fire duration.

    Downloads hourly data from NOAA's High-Resolution Rapid Refresh model
    including relative humidity (r2) and wind components (u10, v10).

    For older HRRR data where RH:2m is not available, relative humidity is
    calculated from 2m temperature (TMP) and 2m dewpoint temperature (DPT)
    using the Magnus formula.

    Args:
        task_info: Task configuration with time range and bounds.
        cache_dir: Cache root for on-the-fly downloads; GRIB files are cached
            under its fixed ``herbie`` subfolder.
        delta_hour: Time interval between downloads in hours.
        batch_size: Number of parallel downloads (default: 4, recommended: 4-8).

    Returns:
        List of DataLayer for r2, u10, and v10 variables.

    Raises:
        ValueError: If no data could be downloaded.

    Note:
        Batch size recommendations:
        - 4: Conservative, lower memory usage, stable on slower connections
        - 8: Good balance of speed and reliability
        - 12+: May hit rate limits or cause memory issues
    """
    log.info(f"Downloading HRRR data for event_id: {task_info.event_id}")

    # GRIB files cache under the fixed ``herbie`` subfolder of the cache root.
    herbie_save_dir = os.path.join(cache_dir, HERBIE_CACHE_NAME)

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
                save_dir=herbie_save_dir,
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
            note = {'data_gaps': data_gaps} if data_gaps else {}
            note['grid'] = _weather_grid_note(task_info, var_info['data'][0].shape)
            payload.append(DataLayer(
                name=var_name,
                data=var_info['data'],
                timestamps=var_info['timestamps'],
                source="HRRR via Herbie",
                native_resolution=3000,
                # Weather sits on its own coarser grid, not the 30 m task grid.
                current_resolution=HRRR_OUTPUT_RESOLUTION,
                unit="%" if var_name == 'r2' else "m/s",
                note=note,
            ))
        else:
            log.warning(f"No data collected for {var_name}")

    return payload


def write_data_gap_log(
    task_info: ProcessingTask,
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
