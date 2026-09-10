"""FEDS25MTBS: GeoPackage discovery, progression dates, perimeter/fireline,
and perimeter interpolation."""

import glob
import logging
import os
from bisect import bisect_left
from dataclasses import replace
from datetime import datetime
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import GeometryCollection, MultiPolygon

from firedataforge.constants import (
    CACHE_DIR, FEDS_CACHE_DIR, FEDS_CACHE_NAME, FEDS_DIR,
    FEDS_MTBS_CHEN_DOI, FEDS_MTBS_ZENODO_DOI,
)
from firedataforge.schemas import DataLayer, ProcessingTask

log = logging.getLogger(__name__)

# Where to look for a fire's GeoPackage: the user-managed archive first, then the
# software cache. On-demand fetches only ever write to the cache (FEDS_CACHE_DIR).
_FEDS_SEARCH_DIRS = (FEDS_DIR, FEDS_CACHE_DIR)

# Provenance string shared by the perimeter/fireline layers (single source of truth
# for the FEDS algorithm + dataset DOIs).
_FEDS_SOURCE = (
    f"FEDS-MTBS; Chen et al. 2022, doi:{FEDS_MTBS_CHEN_DOI}; "
    f"dataset doi:{FEDS_MTBS_ZENODO_DOI}"
)


def find_event_gpkg(
    event_id: str,
    year: Optional[int] = None,
    search_dirs: Optional[tuple[str, ...]] = None,
    cache_dir: str = CACHE_DIR,
) -> Optional[str]:
    """Locate ``<event_id>.gpkg`` in the user archive or the software cache.

    Tries the conventional ``<dir>/<year>/<event_id>.gpkg`` location first in each
    of ``search_dirs`` (the user ``datasets/`` archive, then the ``cache/``), then
    falls back to a recursive search so the GeoPackage is found regardless of which
    year subfolder (or archive-name variant) it lives in. If still not found, and
    on-demand fetching is enabled, the single fire is range-pulled from the
    FEDS-MTBS Zenodo archive into the cache (fail-soft: returns ``None`` when the
    fire is absent from the archive or the network is unavailable).

    ``search_dirs`` defaults to the user archive plus the ``FEDS25MTBS`` subfolder
    of ``cache_dir``; the same cache subfolder is where any on-demand fetch lands.
    """
    if search_dirs is None:
        search_dirs = (FEDS_DIR, os.path.join(cache_dir, FEDS_CACHE_NAME))
    if year is not None:
        for d in search_dirs:
            fast = os.path.join(d, str(year), f"{event_id}.gpkg")
            if os.path.exists(fast):
                return fast
    for d in search_dirs:
        matches = glob.glob(os.path.join(d, "**", f"{event_id}.gpkg"),
                            recursive=True)
        if matches:
            return matches[0]
    # Imported lazily so the network path is never touched in offline setups.
    from firedataforge.remote_archive import maybe_fetch_gpkg
    return maybe_fetch_gpkg(
        event_id, year, feds_dir=os.path.join(cache_dir, FEDS_CACHE_NAME))


def index_event_gpkgs(
    search_dirs: tuple[str, ...] = _FEDS_SEARCH_DIRS,
) -> dict[str, str]:
    """Map every locally available ``<event_id> -> <gpkg path>`` in one tree walk.

    :func:`find_event_gpkg` walks the whole archive on *every* call -- fine for a
    single lookup, but quadratic when resolving tens of thousands of events in a
    row (e.g. :func:`build_firelist`, which otherwise re-scans the entire tree per
    fire). Building this index once and doing O(1) lookups turns that bulk scan
    from minutes into seconds. Earlier ``search_dirs`` win, matching
    ``find_event_gpkg``'s archive-before-cache precedence.
    """
    index: dict[str, str] = {}
    for d in search_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.gpkg"), recursive=True):
            event_id = os.path.splitext(os.path.basename(path))[0]
            index.setdefault(event_id, path)
    return index


def geometries_are_equal(geom1, geom2, threshold: float = 1e-4) -> bool:
    """Check if two geometries are equal within a tolerance.

    Uses symmetric difference to handle floating-point precision issues where
    geometries are identical but .equals() returns False due to tiny coordinate
    differences (machine epsilon).

    Args:
        geom1: First geometry.
        geom2: Second geometry.
        threshold: Maximum symmetric difference ratio to consider equal.
                   Default 1e-4 filters floating-point noise while keeping real changes.

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


def read_perimeter_gdf(gpkg_path: str) -> gpd.GeoDataFrame:
    """Read a GeoPackage's ``perimeter`` layer.

    A single entry point so the event-resolution path can read the layer once and
    share it across the progression-date and bounds helpers (which would otherwise
    each re-read the same file).
    """
    return gpd.read_file(gpkg_path, layer='perimeter')


def get_fire_progression_dates(
    event_id: str,
    year: int,
    gpkg_path: Optional[str] = None,
    cache_dir: str = CACHE_DIR,
    perimeter_gdf: Optional[gpd.GeoDataFrame] = None,
) -> tuple[datetime, datetime]:
    """Find the actual fire progression dates from FEDS25MTBS perimeter data.

    Analyzes consecutive perimeters to find when the fire actually starts
    progressing (first change) and when it stops (last change). Uses a
    tolerance-based comparison to filter out floating-point noise.

    Args:
        event_id: Event ID to look up.
        year: Year of the fire event.
        gpkg_path: Resolved GeoPackage path, if the caller already located it
            (e.g. a bulk build holding a prebuilt index); skips the per-fire scan.
        cache_dir: Cache root for an on-demand GeoPackage fetch (under its fixed
            ``FEDS25MTBS`` subfolder) when the fire is not present locally.
        perimeter_gdf: Pre-read ``perimeter`` layer; pass it to avoid re-reading
            the file when the caller already holds it (see :func:`read_perimeter_gdf`).

    Returns:
        Tuple of (t_start, t_end) representing the actual fire progression period.

    Raises:
        FileNotFoundError: If no matching GeoPackage exists under the data root.
    """
    if perimeter_gdf is None:
        if gpkg_path is None:
            gpkg_path = find_event_gpkg(event_id, year, cache_dir=cache_dir)
        if gpkg_path is None:
            raise FileNotFoundError(
                f"No FEDS perimeter GeoPackage for {event_id} under "
                f"{FEDS_DIR}/ or {FEDS_CACHE_DIR}/"
            )
        perimeter_gdf = read_perimeter_gdf(gpkg_path)

    # Sort by timestamp to ensure correct consecutive comparisons
    gdf = perimeter_gdf.sort_values('t').reset_index(drop=True)

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


def get_perimeter_bounds(
    event_id: str,
    year: Optional[int] = None,
    cache_dir: str = CACHE_DIR,
    gpkg_path: Optional[str] = None,
    perimeter_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Optional[tuple[float, float, float, float]]:
    """Return the lon/lat extent (minx, miny, maxx, maxy) of a FEDS GeoPackage.

    The union of every perimeter frame's envelope, reprojected to EPSG:4326. This
    is the authoritative bounding box for an event whose metadata is otherwise
    resolved from the MTBS service: the MTBS Landsat burn-boundary bbox can be
    tighter than the FEDS VIIRS perimeter and would clip it, so the perimeter
    extent supersedes it whenever a local GeoPackage exists.

    Pass ``gpkg_path`` and/or ``perimeter_gdf`` to reuse an already-resolved path
    or already-read layer instead of re-locating and re-reading the file.

    Returns ``None`` when no GeoPackage is found or it carries no usable geometry,
    so callers can fall back to whatever bbox their metadata source provided.
    """
    if perimeter_gdf is None:
        if gpkg_path is None:
            gpkg_path = find_event_gpkg(event_id, year, cache_dir=cache_dir)
        if gpkg_path is None:
            return None
        try:
            perimeter_gdf = read_perimeter_gdf(gpkg_path)
        except Exception:
            return None
    gdf = perimeter_gdf[perimeter_gdf.geometry.notna()]
    if gdf.empty:
        return None
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    if any(pd.isna(v) for v in (minx, miny, maxx, maxy)):
        return None
    return (float(minx), float(miny), float(maxx), float(maxy))


def get_perimeter_hull_wkt(
    event_id: str,
    year: Optional[int] = None,
    cache_dir: str = CACHE_DIR,
    gpkg_path: Optional[str] = None,
    perimeter_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Optional[str]:
    """Return the perimeter's convex hull as WKT in EPSG:4326, or ``None``.

    :func:`get_perimeter_bounds` reduces the perimeter to a lon/lat *envelope*,
    which discards the fire's shape. That matters because a lon/lat-aligned box is
    a rotated quadrilateral in a projected CRS -- the grid convergence reaches
    ~13.6 degrees at the western edge of EPSG:5070 -- so its axis-aligned extent in
    the target CRS is much larger than the fire and sits off-centre within it.
    Keeping the geometry lets :func:`~firedataforge.events.get_task_info` take the
    extent *after* projecting, which is both tight and centred.

    The convex hull is sufficient and far cheaper than the full geometry: the
    bounding box of a projected geometry equals that of its projected hull, and
    building it from a ``GeometryCollection`` avoids a boolean union over every
    perimeter frame.

    Pass ``gpkg_path`` and/or ``perimeter_gdf`` to reuse an already-resolved path
    or already-read layer, exactly as :func:`get_perimeter_bounds` does.
    """
    if perimeter_gdf is None:
        if gpkg_path is None:
            gpkg_path = find_event_gpkg(event_id, year, cache_dir=cache_dir)
        if gpkg_path is None:
            return None
        try:
            perimeter_gdf = read_perimeter_gdf(gpkg_path)
        except Exception:
            return None
    gdf = perimeter_gdf[perimeter_gdf.geometry.notna()]
    if gdf.empty:
        return None
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    try:
        hull = GeometryCollection(list(gdf.geometry)).convex_hull
    except Exception:
        return None
    if hull.is_empty or hull.geom_type not in ("Polygon", "LineString", "Point"):
        return None
    return hull.wkt


def perimeter_frames(
    gpkg_path: str,
    t_start: datetime,
    t_end: datetime,
) -> tuple[list[datetime], list[MultiPolygon]]:
    """The ``burn_perimeter`` time axis: in-window frames whose geometry changed.

    Single source of truth for that axis. :func:`process_feds25mtbs` rasterizes
    exactly these frames and :func:`process_fireline` resamples onto these
    timestamps, so the two layers cannot drift apart. They used to run separate
    filters over the same GeoPackage -- an area-threshold comparison here, an
    exact-coordinate one for the fireline -- which left the layers on different
    axes for about a third of the archive's fires, usually off by one frame at
    ignition (the first perimeter has no fireline: nothing has advanced yet).

    Args:
        gpkg_path: FEDS25MTBS GeoPackage for one fire.
        t_start: Start of the event window; frames before it are dropped.
        t_end: End of the event window; frames after it are dropped.

    Returns:
        ``(timestamps, geometries)``, in ascending time order and the same
        length. Geometries are ``MultiPolygon`` in EPSG:4326.
    """
    gdf = gpd.read_file(gpkg_path, layer='perimeter')
    gdf = gdf.sort_values('t').reset_index(drop=True)

    timestamps: list[datetime] = []
    geometries: list[MultiPolygon] = []
    last_imported_geom = None
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if geom.geom_type == 'MultiPolygon':
            pass
        elif geom.geom_type == 'Polygon':
            geom = MultiPolygon([geom])
        else:
            log.warning(f"Unexpected geometry type: {geom.geom_type}")
            continue

        timestamp = pd.to_datetime(row['t']).to_pydatetime()
        if not (t_start <= timestamp <= t_end):
            continue

        # Keep only frames that differ from the last one kept; the threshold
        # comparison filters floating-point noise.
        if last_imported_geom is None or not geometries_are_equal(
                geom, last_imported_geom):
            timestamps.append(timestamp)
            geometries.append(geom)
            last_imported_geom = geom  # only update when actually imported

    return timestamps, geometries


def perimeter_timestamps(
    task_info: ProcessingTask,
    cache_dir: str = CACHE_DIR,
) -> list[datetime]:
    """The ``burn_perimeter`` axis for a task, without rasterizing anything.

    Lets a caller that does not build the perimeter layer (e.g. a fireline-only
    run) still put its output on the perimeter's clock.
    """
    gpkg_path = find_event_gpkg(task_info.event_id, task_info.year, cache_dir=cache_dir)
    assert gpkg_path is not None and os.path.exists(gpkg_path), (
        f"Error: FEDS25MTBS GeoPackage for {task_info.event_id} not found "
        f"under {FEDS_DIR}/ or {FEDS_CACHE_DIR}/"
    )
    return perimeter_frames(gpkg_path, task_info.t_start, task_info.t_end)[0]


def _rasterize(geometries, task_info: ProcessingTask) -> np.ndarray:
    """Burn ``geometries`` (already in the task CRS) onto the task grid."""
    t_minx, t_miny, t_maxx, t_maxy = task_info.bounds
    transform = from_origin(t_minx, t_maxy, task_info.resolution, task_info.resolution)
    raster = features.rasterize(
        shapes=[(geom, 1) for geom in geometries],
        out_shape=task_info.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    assert raster.shape == task_info.shape, (
        "Rasterized shape does not match target shape"
    )
    return raster.astype(np.bool_)


def process_feds25mtbs(
    task_info: ProcessingTask,
    cache_dir: str = CACHE_DIR,
) -> DataLayer:
    """Process FEDS25MTBS fire perimeter data into rasterized time series.

    Reads the GeoPackage file for the fire event and rasterizes each timestep's
    perimeter polygon to match the task grid specification. Only includes frames
    within the task_info time range (t_start to t_end) where the perimeter
    actually changed from the previous frame (see :func:`perimeter_frames`).

    Args:
        task_info: Task configuration with event details and grid parameters.
        cache_dir: Cache root for an on-demand GeoPackage fetch (under its fixed
            ``FEDS25MTBS`` subfolder) when the fire is not present locally.

    Returns:
        DataLayer containing boolean rasters for each timestep.

    Raises:
        AssertionError: If the GeoPackage file doesn't exist.
    """
    log.info(f"Processing FEDS25MTBS for event_id: {task_info.event_id}")

    # Resolve across the user archive + software cache (and fetch on demand).
    gpkg_path = find_event_gpkg(task_info.event_id, task_info.year, cache_dir=cache_dir)
    assert gpkg_path is not None and os.path.exists(gpkg_path), (
        f"Error: FEDS25MTBS GeoPackage for {task_info.event_id} not found "
        f"under {FEDS_DIR}/ or {FEDS_CACHE_DIR}/"
    )

    timestamps, geometries = perimeter_frames(
        gpkg_path, task_info.t_start, task_info.t_end)
    log.info(
        f"Unique frames imported: {len(timestamps)} "
        f"(t_start={task_info.t_start}, t_end={task_info.t_end})"
    )
    log.info(f"Target Grid: {task_info.shape} pixels @ {task_info.resolution}m resolution")

    log.info(
        f"Reprojecting geometries from EPSG:4326 to target CRS {task_info.crs}"
    )
    gdf = gpd.GeoDataFrame({'geometry': geometries}, crs="EPSG:4326")
    gdf = gdf.to_crs(task_info.crs)

    processed_rasters = [_rasterize([geom], task_info) for geom in gdf.geometry]

    return DataLayer(
        name="burn_perimeter",
        data=processed_rasters,
        timestamps=timestamps,
        source=_FEDS_SOURCE,
        native_resolution=375,
    )


def process_fireline(
    task_info: ProcessingTask,
    width: float = 375.0,
    cache_dir: str = CACHE_DIR,
    timestamps: Optional[Sequence[datetime]] = None,
) -> DataLayer:
    """Process FEDS25MTBS fireline data into rasterized time series.

    The fireline is not an independent cadence: FEDS derives it from the same
    12-hourly snapshots as the perimeter, and each line marks the perimeter
    segments that advanced since the previous snapshot. This layer therefore
    rides the ``burn_perimeter`` axis rather than one of its own -- frame ``k``
    holds every fireline observed in ``(timestamps[k-1], timestamps[k]]``,
    buffered by half of ``width`` into a corridor and rasterized onto the task
    grid. Frame 0 also takes anything observed at or before ``timestamps[0]``.
    A frame with no observation in its interval is an empty mask, which is what
    the fireline layer means: nothing advanced.

    Sharing the axis is what keeps the two layers paired. Filtering the fireline
    on its own (the previous behaviour) dropped different frames than the
    perimeter filter did, so for about a third of the archive's fires the two
    ``.npz`` files came out on different clocks -- typically off by one frame,
    because a fire's first perimeter has no fireline at all.

    Args:
        task_info: Task configuration with event details and grid parameters.
        width: Total width of the fireline corridor in meters.
        cache_dir: Cache root for an on-demand GeoPackage fetch (under its fixed
            ``FEDS25MTBS`` subfolder) when the fire is not present locally.
        timestamps: Axis to emit on. Defaults to the ``burn_perimeter`` axis read
            from the same GeoPackage. Pass the *observed* perimeter timestamps,
            not an SDF-interpolated axis: a synthetic in-between perimeter has no
            observed fireline, and emitting empty masks there would assert a
            quiet front that was never observed. On a subset axis the layer stays
            correct for the "most recent frame <= t" cursor every consumer uses.

    Returns:
        DataLayer containing boolean rasters, one per timestamp on the axis.
        Empty (no frames) when the fire has no fireline in the window at all.

    Raises:
        AssertionError: If the GeoPackage file doesn't exist.
    """
    log.info(f"Processing fireline for event_id: {task_info.event_id}")

    # Resolve across the user archive + software cache (and fetch on demand).
    gpkg_path = find_event_gpkg(task_info.event_id, task_info.year, cache_dir=cache_dir)
    assert gpkg_path is not None and os.path.exists(gpkg_path), (
        f"Error: FEDS25MTBS GeoPackage for {task_info.event_id} not found "
        f"under {FEDS_DIR}/ or {FEDS_CACHE_DIR}/"
    )

    axis = (list(timestamps) if timestamps is not None
            else perimeter_frames(gpkg_path, task_info.t_start, task_info.t_end)[0])
    assert all(a <= b for a, b in zip(axis, axis[1:])), (
        "fireline axis must be sorted ascending (bucketing uses bisect)")

    gdf = gpd.read_file(gpkg_path, layer='fireline')
    gdf = gdf.sort_values('t').reset_index(drop=True)

    obs_timestamps: list[datetime] = []
    obs_geometries = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        timestamp = pd.to_datetime(row['t']).to_pydatetime()
        if task_info.t_start <= timestamp <= task_info.t_end:
            obs_timestamps.append(timestamp)
            obs_geometries.append(geom)

    log.info(
        f"Fireline observations in window: {len(obs_timestamps)} of {len(gdf)} "
        f"(t_start={task_info.t_start}, t_end={task_info.t_end}); "
        f"burn_perimeter axis: {len(axis)} frames"
    )

    if not axis or not obs_geometries:
        return DataLayer(
            name="fireline", data=[], timestamps=[],
            source=_FEDS_SOURCE, native_resolution=375,
            note={"width_m": width, "axis": "burn_perimeter"},
        )

    log.info(f"Target Grid: {task_info.shape} pixels @ {task_info.resolution}m resolution")
    log.info(
        f"Reprojecting fireline geometries from EPSG:4326 to target CRS {task_info.crs}"
    )
    gdf_lines = gpd.GeoDataFrame({'geometry': obs_geometries}, crs="EPSG:4326")
    gdf_lines = gdf_lines.to_crs(task_info.crs)

    # Buffer line geometries by half the width to create corridors
    corridors = gdf_lines.geometry.buffer(width / 2.0).tolist()

    # Bucket each observation into the axis frame that closes its interval:
    # frame k covers (axis[k-1], axis[k]], and frame 0 also takes anything at or
    # before axis[0]. bisect_left gives that index directly.
    buckets: list[list] = [[] for _ in axis]
    n_after_axis = 0
    for timestamp, corridor in zip(obs_timestamps, corridors):
        index = bisect_left(axis, timestamp)
        if index >= len(axis):
            n_after_axis += 1  # observed after the last perimeter frame
            continue
        buckets[index].append(corridor)

    if n_after_axis:
        log.info(f"Dropped {n_after_axis} fireline observation(s) past the last "
                 f"burn_perimeter frame ({axis[-1]})")

    processed_rasters = [
        _rasterize(shapes, task_info) if shapes
        else np.zeros(task_info.shape, dtype=np.bool_)
        for shapes in buckets
    ]
    n_empty = sum(1 for shapes in buckets if not shapes)
    log.info(f"Fireline frames on the burn_perimeter axis: {len(processed_rasters)} "
             f"({n_empty} with no observed fireline)")

    return DataLayer(
        name="fireline",
        data=processed_rasters,
        timestamps=list(axis),
        source=_FEDS_SOURCE,
        native_resolution=375,
        note={"width_m": width, "axis": "burn_perimeter",
              "n_observations": len(obs_geometries), "n_empty_frames": n_empty},
    )


def process_fireline_max_frp(
    task_info: ProcessingTask,
    fireline_data: DataLayer,
    frp_data: DataLayer,
) -> DataLayer:
    """Assign per-segment maximum nearby FRP to fireline pixels.

    For each fireline timestep, identifies disconnected fireline segments
    via connected-component labelling.  For each segment, dilates its mask
    by 375 m to capture nearby FRP values, takes the maximum FRP within
    that dilated region, and assigns it uniformly to every pixel of that
    segment.  Different segments receive independent max-FRP values.
    Pixels outside the fireline mask are set to zero.

    Args:
        task_info: Task configuration with grid parameters.
        fireline_data: DataLayer from process_fireline (boolean masks).
        frp_data: per-pixel max raw FRP rasters (MW) from
                  process_fireline_frp_points -- true observed intensities, not the
                  mass-preserving splat whose pixels are shares of a detection's FRP.

    Returns:
        DataLayer containing float32 rasters with per-segment
        max-FRP values, in MW.
    """
    from scipy.ndimage import binary_dilation, label

    log.info(f"Processing fireline_max_frp for event_id: {task_info.event_id}")

    fireline_masks = fireline_data.data
    fireline_ts = fireline_data.timestamps or []
    frp_rasters = frp_data.data
    frp_ts = frp_data.timestamps or []

    if not fireline_masks or not frp_rasters:
        log.warning("No fireline or FRP data available for fireline_max_frp")
        return DataLayer(
            name="fireline_max_frp",
            data=[],
            timestamps=[],
            source="Derived from fireline + FRP",
            native_resolution=375,
            unit="MW",
        )

    # Build a circular structuring element with radius ~375 m
    width_m = fireline_data.note.get("width_m", 375.0)
    radius_px = max(int(round(width_m / task_info.resolution)), 1)
    diam = 2 * radius_px + 1
    yy, xx = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    struct = (xx * xx + yy * yy) <= radius_px * radius_px

    log.info(
        f"Search radius: {radius_px} px ({radius_px * task_info.resolution}m), "
        f"structuring element: {diam}x{diam}"
    )

    result_rasters = []
    result_timestamps = []

    for fl_mask, fl_time in zip(fireline_masks, fireline_ts):
        # Find the closest FRP raster in time
        best_idx = None
        best_diff = None
        for idx, ft in enumerate(frp_ts):
            diff = abs((ft - fl_time).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = idx

        if best_idx is None:
            result_rasters.append(np.zeros(task_info.shape, dtype=np.float32))
            result_timestamps.append(fl_time)
            continue

        frp_raster = frp_rasters[best_idx].astype(np.float32)

        # Label disconnected fireline segments
        labeled, n_segments = label(fl_mask.astype(np.uint8))

        out = np.zeros(task_info.shape, dtype=np.float32)

        for seg_id in range(1, n_segments + 1):
            seg_mask = labeled == seg_id

            # Dilate this segment's mask by 375 m to capture nearby FRP
            search_mask = binary_dilation(seg_mask, structure=struct)

            # Max FRP within the dilated region of this segment
            masked_frp = frp_raster[search_mask]
            seg_max = float(masked_frp.max()) if masked_frp.size > 0 else 0.0

            # Assign uniform value to this segment's pixels
            out[seg_mask] = seg_max

        result_rasters.append(out)
        result_timestamps.append(fl_time)

    log.info(f"Created {len(result_rasters)} fireline_max_frp frames")

    return DataLayer(
        name="fireline_max_frp",
        data=result_rasters,
        timestamps=result_timestamps,
        source="Derived from fireline + FRP",
        native_resolution=375,
        unit="MW",
        note={
            "description": "Per-segment max FRP along fireline (within 375m search radius)",
            "width_m": width_m,
            "search_radius_px": radius_px,
        },
    )


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


def interpolate_burn_perimeter(
    data: DataLayer,
    multiplier: int
) -> DataLayer:
    """Interpolate additional frames between existing burn perimeter timesteps.

    Uses SDF-based shape interpolation to create smooth temporal transitions
    between fire perimeter snapshots.

    Args:
        data: Burn perimeter data with timestamps.
        multiplier: Number of intermediate frames to insert between each pair.

    Returns:
        DataLayer with interpolated frames and timestamps.
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
