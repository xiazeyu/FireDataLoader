"""VIIRS Fire Radiative Power: FIRMS/firepix loaders, mass-preserving
Gaussian splat, and optional perimeter masking."""

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from io import StringIO
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm

from firedataforge.config import _firms_map_key
from firedataforge.constants import (
    CACHE_DIR, DEFAULT_FIREPIX_DIR, DEFAULT_FIRMS_DIR,
    FIREPIX_CACHE_NAME, FIRMS_API_BASE, FIRMS_API_MAX_DAYS,
    FIRMS_API_SOURCES, FIRMS_CACHE_NAME, FIRMS_FILES, FIRMS_USECOLS,
)
from firedataforge.schemas import DataLayer, ProcessingTask

log = logging.getLogger(__name__)

# Module-level caches (avoid re-reading large CSVs across events).
_firms_cache: dict = {}
_firepix_cache: dict = {}


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write ``df`` to ``path`` atomically (tmp + rename) to avoid torn reads."""
    tmp = f"{path}.part"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: str) -> None:
    """Write ``obj`` as JSON to ``path`` atomically (tmp + rename)."""
    tmp = f"{path}.part"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _firms_cache_covers(meta_path: str, requested: dict, eps: float = 1e-6) -> bool:
    """True if a cached FIRMS slice's AOI + time window covers ``requested``.

    The streamed FIRMS slice is specific to the bounding box and date window it was
    fetched for, so a rerun with a wider AOI (larger ``--buffer``/``--resolution``
    or a different ``--crs``) or a longer window must re-fetch rather than silently
    reuse an under-covered cache. A cache written without this sidecar (older
    versions) is treated as not-covering, so it is refreshed once.
    """
    try:
        with open(meta_path) as fh:
            cached = json.load(fh)
        spatial = (
            requested["min_lon"] >= cached["min_lon"] - eps
            and requested["min_lat"] >= cached["min_lat"] - eps
            and requested["max_lon"] <= cached["max_lon"] + eps
            and requested["max_lat"] <= cached["max_lat"] + eps
        )
        temporal = (
            datetime.fromisoformat(requested["t_start"])
            >= datetime.fromisoformat(cached["t_start"])
            and datetime.fromisoformat(requested["t_end"])
            <= datetime.fromisoformat(cached["t_end"])
        )
        return spatial and temporal
    except (OSError, ValueError, KeyError):
        return False


def _fetch_firms_area(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    t_start: datetime,
    t_end: datetime,
    map_key: str,
) -> pd.DataFrame:
    """Fetch FIRMS active-fire points for a bounding box and date range.

    Streams only the requested slice from the NASA FIRMS Area API instead of
    downloading the full archive. The Area API limits each request to a 5-day
    window, so longer events are fetched in consecutive chunks; standard
    processing (archive) and near-real-time sources are both queried and merged.

    Args:
        min_lon, min_lat, max_lon, max_lat: Bounding box in EPSG:4326.
        t_start, t_end: Inclusive datetime range to fetch.
        map_key: NASA FIRMS MAP_KEY.

    Returns:
        DataFrame with the raw FIRMS Area API columns (may be empty).
    """
    area = f"{min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}"
    start_date = t_start.date()
    end_date = t_end.date()

    frames: list[pd.DataFrame] = []
    cur = start_date
    while cur <= end_date:
        span = min(FIRMS_API_MAX_DAYS, (end_date - cur).days + 1)
        for source in FIRMS_API_SOURCES:
            url = f"{FIRMS_API_BASE}/{map_key}/{source}/{area}/{span}/{cur.isoformat()}"
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    text = resp.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                log.warning(f"FIRMS API HTTP {e.code} for {source} {cur}: {e.reason}")
                continue
            except urllib.error.URLError as e:
                log.warning(f"FIRMS API request failed for {source} {cur}: {e.reason}")
                continue

            header = text.lstrip().split("\n", 1)[0].lower()
            if "latitude" not in header or "longitude" not in header:
                # Not a CSV payload -> error message (bad/expired key, quota, etc.)
                log.warning(
                    f"FIRMS API returned a non-data response for {source} {cur}: "
                    f"{text.strip()[:200]}"
                )
                continue

            df = pd.read_csv(StringIO(text))
            if len(df):
                frames.append(df)
        cur = cur + timedelta(days=span)

    if not frames:
        # Empty-but-typed so the cached CSV round-trips (header row, 0 rows).
        return pd.DataFrame(columns=FIRMS_USECOLS)
    return pd.concat(frames, ignore_index=True)


def _firms_add_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add a parsed datetime column ``t`` from FIRMS acq_date/acq_time.

    Drops the source acq_date/acq_time columns. Works for both the bundled
    archive CSVs and the Area API response, which share the FIRMS_USECOLS schema.
    """
    df = df.copy()
    df['acq_time_str'] = df['acq_time'].astype(str).str.zfill(4)
    df['t'] = pd.to_datetime(
        df['acq_date'].astype(str) + ' ' +
        df['acq_time_str'].str[:2] + ':' +
        df['acq_time_str'].str[2:]
    )
    return df.drop(columns=['acq_time_str', 'acq_date', 'acq_time'])


def clear_frp_cache() -> None:
    """Clear cached FIRMS and Firepix data to free memory.

    Call this when done processing fires or when memory is needed.
    """
    _firms_cache.clear()
    _firepix_cache.clear()
    log.info("Cleared FIRMS and Firepix cache")


def _get_perimeter_masks_from_data(
    perimeter_data: DataLayer,
) -> tuple[list[np.ndarray], list[datetime]]:
    """Extract perimeter masks and timestamps from DataLayer.

    Converts the output of process_feds25mtbs into the format needed
    for FRP masking.

    Args:
        perimeter_data: DataLayer from process_feds25mtbs.

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
    task_info: ProcessingTask,
    firms_dir: str = DEFAULT_FIRMS_DIR,
    buffer_deg: float = 0.01,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Load FIRMS data for fires from 2025 onwards (and any fire without firepix).

    Filters FIRMS data by spatial and temporal bounds defined in ProcessingTask.

    When the FIRMS_MAP_KEY environment variable is set, only the AOI + date
    window is streamed from the NASA FIRMS Area API (no full-archive download)
    and cached per-event under the cache root's fixed ``FIRMS`` subfolder
    (``<cache_dir>/FIRMS/``). Otherwise any user-placed full-archive CSVs in
    ``firms_dir`` (``datasets/FIRMS/``) are used as a fallback.

    Args:
        task_info: Task configuration with bounds and time range.
        firms_dir: Directory holding optional user-placed FIRMS archive CSVs.
        buffer_deg: Buffer in degrees to add around bounds for spatial filtering.
        cache_dir: Cache root for on-the-fly downloads; streamed FIRMS slices
            are cached under its fixed ``FIRMS`` subfolder.

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

    # Gather candidate source frames (each carrying the FIRMS_USECOLS schema plus
    # a parsed ``t`` column). Prefer streaming the exact slice from the FIRMS Area
    # API when a MAP_KEY is configured; otherwise fall back to the bundled archive
    # CSVs so existing local setups keep working.
    source_frames: list[pd.DataFrame] = []
    map_key = _firms_map_key()

    if map_key:
        # Streaming path: download only this event's bbox + date window and cache
        # the (small) result in the software cache so repeated runs never re-fetch.
        firms_cache_dir = os.path.join(cache_dir, FIRMS_CACHE_NAME)
        cache_path = os.path.join(firms_cache_dir, f"{task_info.event_id}.csv")
        meta_path = os.path.join(firms_cache_dir, f"{task_info.event_id}.meta.json")
        # The streamed slice is specific to this bbox + window; record them in a
        # sidecar so a later, wider rerun re-fetches instead of reusing a slice
        # that does not cover the new AOI/window.
        requested = {
            "min_lon": min_lon, "min_lat": min_lat,
            "max_lon": max_lon, "max_lat": max_lat,
            "t_start": t_start.isoformat(), "t_end": t_end.isoformat(),
        }
        if os.path.exists(cache_path) and _firms_cache_covers(meta_path, requested):
            log.info(f"Using cached FIRMS slice: {cache_path}")
            try:
                raw = pd.read_csv(cache_path)
            except pd.errors.EmptyDataError:
                raw = pd.DataFrame(columns=FIRMS_USECOLS)
        else:
            log.info(
                f"Streaming FIRMS data from Area API for bbox "
                f"({min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}) "
                f"{t_start.date()}..{t_end.date()}"
            )
            raw = _fetch_firms_area(
                min_lon, min_lat, max_lon, max_lat, t_start, t_end, map_key
            )
            os.makedirs(firms_cache_dir, exist_ok=True)
            _atomic_write_csv(raw, cache_path)
            _atomic_write_json(requested, meta_path)
            log.info(f"  Cached {len(raw):,} FIRMS points to {cache_path}")

        if len(raw):
            missing = [c for c in FIRMS_USECOLS if c not in raw.columns]
            if missing:
                log.warning(f"FIRMS Area API response missing columns: {missing}")
            else:
                source_frames.append(_firms_add_time(raw[FIRMS_USECOLS]))
    else:
        # Fallback path: filter the bundled full-archive CSVs.
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
                df = _firms_add_time(pd.read_csv(filepath, usecols=FIRMS_USECOLS))

                # Cache for future use
                _firms_cache[filepath] = df
                log.info(f"  Cached {len(df):,} points from {firms_file}")

            source_frames.append(df)

    if not source_frames:
        if not map_key:
            log.warning(
                "No FIRMS source available. Set the FIRMS_MAP_KEY environment "
                "variable to stream from the NASA FIRMS Area API "
                "(https://firms.modaps.eosdis.nasa.gov/api/map_key/), or place "
                f"archive CSVs in {firms_dir}/ ({', '.join(FIRMS_FILES)})."
            )
        log.warning("No FIRMS data found for the specified bounds and time range")
        return pd.DataFrame(columns=['Lat', 'Lon', 'FRP', 'Confidence', 'DNFlag', 't', 'Event_ID'])

    # Apply exact spatial + temporal filtering to every source frame.
    all_data = []
    for df in source_frames:
        spatial_mask = (
            (df['latitude'] >= min_lat) &
            (df['latitude'] <= max_lat) &
            (df['longitude'] >= min_lon) &
            (df['longitude'] <= max_lon)
        )
        temporal_mask = (df['t'] >= t_start) & (df['t'] <= t_end)
        filtered = df[spatial_mask & temporal_mask].copy()
        if len(filtered) > 0:
            log.info(f"  Found {len(filtered)} fire points")
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
    task_info: ProcessingTask,
    firepix_dir: str = DEFAULT_FIREPIX_DIR,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Load pre-processed firepix data for fires before 2025.

    Returns *every* point for the event, including detections past the active
    window (the archive runs to the MTBS ``Ted``); the caller
    (:func:`process_frp`) applies the ``[t_start, t_end]`` time clip. (Contrast
    :func:`_load_firms_data`, which clips internally.)

    Args:
        task_info: Task configuration with event ID and year.
        firepix_dir: Directory containing Firepix CSV files.
        cache_dir: Cache root for on-the-fly downloads; firepix CSVs pulled from
            the Zenodo archive are cached under its fixed ``FEDS25MTBS/firepix``
            subfolder.

    Returns:
        DataFrame with columns [Lat, Lon, FRP, Confidence, DNFlag, t, Event_ID].
    """
    # Prefer the user archive (datasets/), then the software cache (cache/).
    firepix_cache_dir = os.path.join(cache_dir, FIREPIX_CACHE_NAME)
    filepath = os.path.join(firepix_dir, f"Firepix_{task_info.year}.csv")
    if not os.path.exists(filepath):
        cache_path = os.path.join(
            firepix_cache_dir, f"Firepix_{task_info.year}.csv")
        if not os.path.exists(cache_path):
            # On-demand: range-pull this year's firepix from the FEDS-MTBS Zenodo
            # archive into the cache (fail-soft -> still missing if disabled /
            # offline / absent).
            from firedataforge.remote_archive import maybe_fetch_firepix_year
            maybe_fetch_firepix_year(task_info.year, firepix_cache_dir)
        filepath = cache_path
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
    """Rasterize points using Gaussian splatting (mass-preserving).

    Each point's FRP value is spread over a Gaussian footprint sized to the source
    sensor's pixel (e.g. 375 m for VIIRS), with the weights normalized to sum to one.
    The total radiative power is therefore conserved: summing the rasterized footprint
    recovers the point's observed FRP, and a point whose footprint lies within the
    grid contributes exactly its value to the raster total. Per-pixel values are thus
    shares of the detection's FRP (much smaller than the observed value), not the
    observed value itself.

    Normalization is over the whole kernel window rather than only its in-bounds
    part, so the grid edge is lossy rather than accumulative: a detection near the
    boundary contributes exactly the in-grid fraction of its Gaussian mass, and one
    whose window falls entirely off the grid contributes nothing. Rescaling by the
    in-bounds weights instead would deposit a clipped detection's *full* FRP onto
    the boundary cells.

    Args:
        x: X coordinates in target CRS.
        y: Y coordinates in target CRS.
        values: Values to rasterize (e.g., FRP).
        bounds: Bounding box (minx, miny, maxx, maxy).
        shape: Output shape (height, width).
        resolution: Target pixel resolution in meters.
        source_resolution: Source sensor pixel resolution in meters (default 375m for VIIRS).

    Returns:
        Rasterized array whose footprint sum equals the input FRP (mass-preserving).
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

        # Pass 1: accumulate the Gaussian weights over the WHOLE kernel window,
        # in bounds or not, while collecting only the cells that land on the grid.
        # Normalizing by the in-bounds sum alone would rescale a clipped footprint
        # back up to the detection's full FRP -- piling the off-grid mass onto the
        # boundary cells, and inflating every in-bounds cell of that footprint by
        # the same factor. Interior detections are unaffected: their window lies
        # entirely in bounds, so this sum is identical to the in-bounds one.
        cells = []
        wsum = 0.0
        for dy in range(-kernel_radius, kernel_radius + 1):
            for dx in range(-kernel_radius, kernel_radius + 1):
                row = py_center + dy
                col = px_center + dx

                # Distance from point center to pixel center
                dist_x = (col + 0.5) - px[i]
                dist_y = (row + 0.5) - py[i]
                dist_sq = dist_x**2 + dist_y**2
                w = np.exp(-dist_sq / (2 * sigma_pixels**2))
                wsum += w

                if 0 <= row < height and 0 <= col < width:
                    cells.append((row, col, w))

        # Pass 2: deposit val * w / sum(w) so the footprint weights sum to one and
        # the point's full FRP is conserved (mass-preserving): a detection whose
        # footprint lies entirely within the grid contributes exactly ``val`` to the
        # raster total, a clipped footprint contributes exactly the in-grid fraction
        # of its Gaussian mass, and a detection whose window is wholly off the grid
        # contributes nothing.
        if wsum > 0:
            for row, col, w in cells:
                raster[row, col] += val * w / wsum

    return raster.astype(np.float32)


def _rasterize_point_max(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    resolution: int,
) -> np.ndarray:
    """Per-pixel maximum of raw point values, with no spreading.

    Each point's value is written to the cell it falls in, keeping the maximum on
    collision. Unlike the mass-preserving splat (whose pixels are shares of a
    detection's FRP), this preserves the observed FRP intensity in MW, which is what
    the nearby-max query in :func:`firedataforge.sources.feds.process_fireline_max_frp`
    needs.
    """
    minx, miny, maxx, maxy = bounds
    height, width = shape
    raster = np.zeros(shape, dtype=np.float64)
    col = np.floor((np.asarray(x) - minx) / resolution).astype(int)
    row = np.floor((maxy - np.asarray(y)) / resolution).astype(int)
    inb = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    np.maximum.at(raster, (row[inb], col[inb]), np.asarray(values, float)[inb])
    return raster.astype(np.float32)


def _rasterize_fire_points(
    df: pd.DataFrame,
    task_info: ProcessingTask,
    perimeter_masks: list[np.ndarray],
    perimeter_timestamps: list[datetime],
    time_interval_hours: int = 12,
    point_max: bool = False,
) -> tuple[list[np.ndarray], list[datetime]]:
    """Rasterize fire points and apply perimeter masking.

    Groups fire points by time intervals and creates FRP rasters using the
    mass-preserving Gaussian splat, or, when ``point_max`` is set, a per-pixel
    maximum of the raw FRP (for ``fireline_max_frp``, which needs true MW intensity).

    Args:
        df: DataFrame with fire point data.
        task_info: Task configuration.
        perimeter_masks: List of fire perimeter masks.
        perimeter_timestamps: List of perimeter timestamps.
        time_interval_hours: Time interval for grouping points.
        point_max: If True, keep the per-pixel max raw FRP instead of splatting.

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

        # Rasterize: per-pixel max raw FRP, or the mass-preserving Gaussian splat.
        rasterize = _rasterize_point_max if point_max else _gaussian_splat_rasterize
        raster = rasterize(
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


def _load_frp_points(
    task_info: ProcessingTask,
    perimeter_data: Optional[DataLayer] = None,
    time_of_day: Literal['all', 'day', 'night'] = 'all',
    firms_dir: str = DEFAULT_FIRMS_DIR,
    firepix_dir: str = DEFAULT_FIREPIX_DIR,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Load and filter the VIIRS active-fire points for an event.

    Selects the source (FEDS25MTBS firepix for pre-2025 fires with a perimeter, else
    NASA FIRMS), clips to the task window ``[t_start, t_end]``, and applies the VIIRS
    day/night overpass filter (on the ``DNFlag``, which is correct regardless of
    clock hour). Shared by the splatted FRP layers (:func:`process_frp`) and the
    point-max ``fireline_max_frp`` input (:func:`process_fireline_frp_points`).

    Returns a DataFrame with columns ``[Lat, Lon, FRP, Confidence, DNFlag, t, ...]``.
    """
    if perimeter_data is not None and task_info.year < 2025:
        df = _load_firepix_data(task_info, firepix_dir, cache_dir=cache_dir)
    else:
        df = _load_firms_data(task_info, firms_dir, cache_dir=cache_dir)

    if not df.empty:
        df = df[(df["t"] >= task_info.t_start) & (df["t"] <= task_info.t_end)].copy()
    if time_of_day in ('day', 'night') and not df.empty:
        flag = "D" if time_of_day == 'day' else "N"
        df = df[df["DNFlag"].astype(str).str.upper().str.startswith(flag)].copy()
    log.info(f"Loaded {len(df)} FRP points ({time_of_day}) for {task_info.event_id}")
    return df


def process_frp(
    task_info: ProcessingTask,
    perimeter_data: Optional[DataLayer] = None,
    time_interval_hours: int = 12,
    time_of_day: Literal['all', 'day', 'night'] = 'all',
    firms_dir: str = DEFAULT_FIRMS_DIR,
    firepix_dir: str = DEFAULT_FIREPIX_DIR,
    cache_dir: str = CACHE_DIR,
) -> DataLayer:
    """Process VIIRS active-fire points into a rasterized FRP (MW) time series.

    Source selection:
      * With a FEDS perimeter (``perimeter_data`` given), the paper's behavior:
        pre-2025 fires use the bundled FEDS25MTBS firepix archive (which now
        covers 2012-2024), 2025+ fires use the NASA FIRMS Area API, and every
        frame is masked to the perimeter.
      * Without a FEDS perimeter, FRP is sourced from NASA FIRMS for any year and
        left *unmasked* (``perimeter_masked=False`` in the metadata), so the layer
        is still produced when the FEDS archive is absent.

    Rasterization uses a mass-preserving Gaussian splat (see
    :func:`_gaussian_splat_rasterize`).

    Args:
        task_info: Task configuration (event details, grid, time window).
        perimeter_data: Perimeter DataLayer from :func:`process_feds25mtbs` used for
            masking; ``None`` disables masking and forces the FIRMS source.
        time_interval_hours: Interval for grouping fire points (default 12 hours).
        time_of_day: ``'all'`` (default), ``'day'``, or ``'night'`` -- selected
            via the VIIRS day/night flag (DNFlag), not a clock time.
        firms_dir: Directory containing the FIRMS streaming cache / archive CSVs.
        firepix_dir: Directory containing the FEDS25MTBS firepix CSVs.
        cache_dir: Cache root for on-the-fly FIRMS/firepix downloads (each under
            its own fixed subfolder).

    Either way, frames are restricted to the task window
    ``[task_info.t_start, task_info.t_end]`` so FRP shares the same temporal
    extent as the perimeter/fireline layers.

    Returns:
        DataLayer named ``frp`` / ``frp_daytime`` / ``frp_nighttime`` with one
        raster per time step, parallel ``timestamps``, ``unit="MW"``, and a
        ``note`` recording the source and whether masking was applied.
    """
    # Determine output name and source suffix based on time_of_day
    if time_of_day == 'day':
        output_name = "frp_daytime"
        source_suffix = " - Day"
        observation_time = "day (VIIRS daytime overpass)"
        log.info(f"Processing DAYTIME FRP for event: {task_info.event_id}")
    elif time_of_day == 'night':
        output_name = "frp_nighttime"
        source_suffix = " - Night"
        observation_time = "night (VIIRS nighttime overpass)"
        log.info(f"Processing NIGHTTIME FRP for event: {task_info.event_id}")
    else:
        output_name = "frp"
        source_suffix = ""
        observation_time = None
        log.info(f"Processing FRP for event: {task_info.event_id}")

    log.info(f"Year: {task_info.year}, Time range: {task_info.t_start} to {task_info.t_end}")

    # Perimeter masks (empty when no FEDS perimeter is provided -> unmasked FRP).
    masked = perimeter_data is not None
    if perimeter_data is not None:
        perimeter_masks, perimeter_timestamps = _get_perimeter_masks_from_data(perimeter_data)
        log.info(f"Using {len(perimeter_masks)} perimeter frames for masking")
    else:
        perimeter_masks, perimeter_timestamps = [], []
        log.info("No FEDS perimeter available; producing unmasked FRP from FIRMS")

    # Load the source-selected, window-clipped, day/night-filtered points. The
    # source string mirrors the selection _load_frp_points applies (firepix for
    # pre-2025 fires with a perimeter, else NASA FIRMS).
    source = (f"FEDS25MTBS Firepix (VIIRS Active Fire){source_suffix}"
              if masked and task_info.year < 2025
              else f"NASA FIRMS (VIIRS Active Fire){source_suffix}")
    df = _load_frp_points(task_info, perimeter_data, time_of_day, firms_dir,
                          firepix_dir, cache_dir=cache_dir)

    n_points = len(df)
    if time_of_day == 'all':
        log.info(f"Total fire points: {n_points}")

    # Build note dictionary
    note: dict = {
        "event_id": task_info.event_id,
        "year": task_info.year,
        "n_points": n_points,
        "time_interval_hours": time_interval_hours,
        "rasterization_method": "gaussian_splatting_mass_preserving",
        "perimeter_masked": masked,
    }
    if observation_time:
        note["observation_time"] = observation_time

    if df.empty:
        log.warning(f"No fire points found for {output_name}")
        return DataLayer(
            name=output_name,
            data=[],
            timestamps=[],
            source=source,
            native_resolution=375,
            unit="MW",
            note=note,
        )

    # Rasterize fire points to grid with perimeter masking
    rasters, timestamps = _rasterize_fire_points(
        df, task_info, perimeter_masks, perimeter_timestamps, time_interval_hours
    )

    log.info(f"Created {len(rasters)} time step rasters")

    note["n_perimeter_frames"] = len(perimeter_masks)

    return DataLayer(
        name=output_name,
        data=rasters,
        timestamps=timestamps,
        source=source,
        native_resolution=375,
        unit="MW",
        note=note,
    )


def process_fireline_frp_points(
    task_info: ProcessingTask,
    perimeter_data: Optional[DataLayer] = None,
    time_of_day: Literal['all', 'day', 'night'] = 'day',
    time_interval_hours: int = 24,
    firms_dir: str = DEFAULT_FIRMS_DIR,
    firepix_dir: str = DEFAULT_FIREPIX_DIR,
    cache_dir: str = CACHE_DIR,
) -> DataLayer:
    """Per-pixel max raw FRP (MW) time series used as the ``fireline_max_frp`` source.

    Loads the same day-overpass points and applies the same perimeter masking as the
    splatted :func:`process_frp` layers, but keeps each detection's raw FRP at its grid
    cell (max on collision) instead of mass-preservingly splatting it, so the values stay
    true MW intensities for the nearby-max query in
    :func:`firedataforge.sources.feds.process_fireline_max_frp`.

    Returns a DataLayer named ``frp_points_max`` (not persisted as an output layer).
    """
    df = _load_frp_points(task_info, perimeter_data, time_of_day, firms_dir,
                          firepix_dir, cache_dir=cache_dir)
    if df.empty:
        return DataLayer(name="frp_points_max", data=[], timestamps=[],
                         source="VIIRS Active Fire (per-pixel max)",
                         native_resolution=375, unit="MW")
    if perimeter_data is not None:
        perimeter_masks, perimeter_timestamps = _get_perimeter_masks_from_data(perimeter_data)
    else:
        perimeter_masks, perimeter_timestamps = [], []
    rasters, timestamps = _rasterize_fire_points(
        df, task_info, perimeter_masks, perimeter_timestamps,
        time_interval_hours, point_max=True)
    return DataLayer(name="frp_points_max", data=rasters, timestamps=timestamps,
                     source="VIIRS Active Fire (per-pixel max)",
                     native_resolution=375, unit="MW")
