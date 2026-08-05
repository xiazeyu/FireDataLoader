"""Quantitative validation metrics for FireDataForge outputs.

Each function operates on an event's output directory (``output/<event_id>/``)
produced by ``firedataforge.process_single_fire`` and returns a small dict of
numbers:

    * ``reprojection_roundtrip_error`` -- self-contained (needs only the saved grid
      coordinates).
    * ``frp_conservation`` -- reloads the event's VIIRS active-fire points to verify
      the Gaussian splat conserves total radiative power; needs the FRP source (the
      bundled firepix archive, or a configured NASA FIRMS key for 2024+ fires).
    * ``categorical_agreement`` and ``continuous_rmse`` -- compare the pipeline output
      against a native-resolution reference (native WorldCover tiles, native 3DEP DEM)
      re-fetched from Earth Engine and aggregated to the target grid by
      :func:`_native_reference`; they need a configured GEE project and return ``{}``
      otherwise.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Optional

import numpy as np
from pyproj import Transformer

import firedataforge as fdf
from schemas import DataLayer


def _load(event_dir: str, name: str) -> Optional[DataLayer]:
    """Load ``<event_dir>/<name>.npy`` as a DataLayer, or None if absent."""
    path = os.path.join(event_dir, f"{name}.npy")
    return fdf.load_numpy(path) if os.path.exists(path) else None


def _task_from_event(event_dir: str):
    """Reconstruct the :class:`ProcessingTask` from an event's ``task_info.npy``."""
    ti = _load(event_dir, "task_info")
    if ti is None or not ti.data or not isinstance(ti.data[0], dict):
        return None
    try:
        from schemas import ProcessingTask
        return ProcessingTask(**ti.data[0])
    except (TypeError, ImportError):
        return None


# Native-resolution source for each validatable layer: (band, GEE collection id).
_NATIVE_SOURCES = {
    "landcover": ("Map", "ESA/WorldCover/v200"),   # categorical, 10 m native
    "elevation": ("elevation", "USGS/3DEP/1m"),    # continuous, 1 m native
}

# Output-grid tile size (pixels) for fetching the native reference; bounds the
# per-request footprint so large events stay under Earth Engine's reprojection cap.
_GEE_TILE = 512


def _native_reference(layer: str, georef, reducer: str):
    """Native-resolution source for ``layer`` aggregated onto ``georef``'s grid.

    Re-fetches the layer's native source over the event footprint and aggregates it
    to the output grid with ``reducer`` -- ``"mode"`` (majority) for categorical
    layers, ``"mean"`` for continuous ones -- via Earth Engine's area-weighted
    :meth:`reduceResolution`. Because this is an *independent* aggregation of the
    native pixels, comparing it against the pipeline output (which point-samples the
    source through nearest/bilinear resampling) measures real resampling fidelity
    instead of being tautological.

    ``wui`` is dispatched to :func:`_wui_native_reference` (GlobalWUI raster tiles,
    not Earth Engine); ``landcover`` / ``elevation`` use the Earth Engine path below.

    Returns the aggregated array (same shape as the grid), or ``None`` if the layer
    has no registered native source or its backend is unavailable / unconfigured.
    """
    if reducer not in ("mode", "mean") or georef is None:
        return None
    if layer == "wui":
        return _wui_native_reference(georef)
    if layer not in _NATIVE_SOURCES:
        return None
    try:
        import ee
        import geemap
        from firedataforge.sources.gee import (
            GEEProjectNotConfiguredError, _ensure_ee_initialized,
        )
    except ImportError:
        return None
    try:
        _ensure_ee_initialized()
    except GEEProjectNotConfiguredError:
        return None

    band, collection_id = _NATIVE_SOURCES[layer]
    roi = ee.Geometry.Rectangle(list(georef.bounds), georef.crs, False)
    collection = ee.ImageCollection(collection_id).filterBounds(roi)
    native_proj = collection.first().select(band).projection()
    ee_reducer = ee.Reducer.mode() if reducer == "mode" else ee.Reducer.mean()

    # reduceResolution must be immediately followed by reproject to the target grid.
    # bestEffort lets GEE aggregate from an overview pyramid when reading the true
    # native grid would blow Earth Engine's per-request reprojection cap (e.g. 1 m
    # 3DEP over a multi-km footprint); a small maxPixels keeps the categorical
    # majority exact (its native factor is tiny) while serving a stable mean for the
    # fine continuous source.
    image = (collection.mosaic().select(band)
             .setDefaultProjection(native_proj)
             .reduceResolution(reducer=ee_reducer, maxPixels=64, bestEffort=True)
             .reproject(crs=georef.crs, scale=georef.resolution))

    # Fetch in tiles: a large footprint at native scale exceeds Earth Engine's
    # per-request reprojection cap, so pull the grid in blocks and stitch them.
    H, W = georef.shape
    minx, _, _, maxy = georef.bounds
    res = georef.resolution
    out = np.full((H, W), np.nan, dtype=float)
    for r0 in range(0, H, _GEE_TILE):
        for c0 in range(0, W, _GEE_TILE):
            r1, c1 = min(r0 + _GEE_TILE, H), min(c0 + _GEE_TILE, W)
            tile_roi = ee.Geometry.Rectangle(
                [minx + c0 * res, maxy - r1 * res, minx + c1 * res, maxy - r0 * res],
                georef.crs, False)
            block = np.squeeze(geemap.ee_to_numpy(image, region=tile_roi), axis=2)
            out[r0:r1, c0:c1] = block[:r1 - r0, :c1 - c0]
    return np.around(out).astype(np.int16) if reducer == "mode" else out


def _wui_native_reference(georef):
    """Majority-aggregated native GlobalWUI reference on ``georef``'s grid.

    Loads the same native 10 m GlobalWUI tiles the pipeline uses, then reprojects
    them to the target grid with majority (``Resampling.mode``) resampling instead of
    the pipeline's nearest-neighbour, so the overall-accuracy comparison is not
    tautological. Returns ``None`` if the tiles or GlobalWUI dependencies are unavailable.
    """
    try:
        import rioxarray  # noqa: F401  (.rio accessor)
        from rasterio.enums import Resampling
        from rasterio.transform import from_origin
        from firedataforge.sources.wui import (
            DEFAULT_GLOBALWUI_DIR, GLOBALWUI_CACHE_DIR,
            _get_equi7_tiles_for_bounds, _stream_globalwui_tile,
        )
    except ImportError:
        return None

    datasets = []
    for tile_id in _get_equi7_tiles_for_bounds(georef.bounds, georef.crs):
        # Prefer the user archive (datasets/), then the software cache (cache/).
        archive_path = os.path.join(DEFAULT_GLOBALWUI_DIR, tile_id, "WUI.tif")
        cache_path = os.path.join(GLOBALWUI_CACHE_DIR, tile_id, "WUI.tif")
        if os.path.exists(archive_path):
            path = archive_path
        elif os.path.exists(cache_path):
            path = cache_path
        else:
            path = cache_path
            try:
                if not _stream_globalwui_tile(tile_id, path):
                    continue
            except Exception:
                continue
        try:
            datasets.append(rioxarray.open_rasterio(path))
        except Exception:
            continue
    if not datasets:
        return None

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        from rioxarray.merge import merge_arrays
        merged = merge_arrays(datasets)

    minx, _, _, maxy = georef.bounds
    transform = from_origin(minx, maxy, georef.resolution, georef.resolution)
    ref = merged.rio.reproject(
        dst_crs=georef.crs,
        shape=tuple(georef.shape),
        transform=transform,
        resampling=Resampling.mode,  # majority, vs the pipeline's nearest-neighbour
    ).values
    if ref.ndim == 3:
        ref = ref[0]
    if ref.shape != tuple(georef.shape):
        return None
    return np.around(ref).astype(np.int16)


def reprojection_roundtrip_error(event_dir: str) -> dict:
    """Round-trip reprojection error of the output grid, in meters.

    Projects every pixel-center from the target CRS to EPSG:4326 and back, then
    reports the displacement statistics. A correct transform pipeline yields
    sub-millimeter residuals; this catches CRS/axis-order regressions.

    Returns ``{"max_m", "mean_m", "rmse_m", "n_points"}`` or ``{}`` if the grid
    coordinates are unavailable.
    """
    coords = _load(event_dir, "coordinates")
    if coords is None or coords.georeference is None:
        return {}
    crs = coords.georeference.crs
    x, y = coords.data  # 1-D pixel-center arrays
    xx, yy = np.meshgrid(x, y)
    fwd = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    inv = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon, lat = fwd.transform(xx.ravel(), yy.ravel())
    x2, y2 = inv.transform(lon, lat)
    disp = np.hypot(x2 - xx.ravel(), y2 - yy.ravel())
    return {
        "max_m": float(np.max(disp)),
        "mean_m": float(np.mean(disp)),
        "rmse_m": float(np.sqrt(np.mean(disp ** 2))),
        "n_points": int(disp.size),
    }


def _expected_ingrid_splat(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                           bounds: tuple[float, float, float, float],
                           shape: tuple[int, int], resolution: int,
                           source_resolution: float = 375.0) -> np.ndarray:
    """Oracle for the Gaussian splat, independent of ``sources/frp.py``.

    Normalizes each detection over the **whole** kernel window, so a detection
    inside the grid contributes exactly its value and a clipped one contributes
    exactly the in-grid fraction of its Gaussian mass. Reimplemented here rather
    than imported so :func:`frp_conservation` cannot be graded by the code it is
    checking.
    """
    minx, miny, maxx, maxy = bounds
    height, width = shape
    raster = np.zeros(shape, dtype=np.float64)

    sigma_pixels = (source_resolution / resolution) / 2.0
    kernel_radius = int(np.ceil(3 * sigma_pixels))
    px = (x - minx) / resolution
    py = (maxy - y) / resolution

    for i in range(len(values)):
        px_center, py_center = int(np.round(px[i])), int(np.round(py[i]))
        cells, wsum = [], 0.0
        for dy in range(-kernel_radius, kernel_radius + 1):
            for dx in range(-kernel_radius, kernel_radius + 1):
                row, col = py_center + dy, px_center + dx
                dist_sq = ((col + 0.5) - px[i]) ** 2 + ((row + 0.5) - py[i]) ** 2
                w = np.exp(-dist_sq / (2 * sigma_pixels ** 2))
                wsum += w
                if 0 <= row < height and 0 <= col < width:
                    cells.append((row, col, w))
        if wsum > 0:
            for row, col, w in cells:
                raster[row, col] += values[i] * w / wsum
    return raster


def frp_conservation(event_dir: str) -> dict:
    """Radiative-power conservation of the mass-preserving Gaussian splat.

    Reloads the event's VIIRS active-fire points (the same source the pipeline used),
    re-splats them onto the grid *without* perimeter masking to isolate the splat
    operator, and compares the rasterized total against two references.

    ``rel_error`` is measured against the summed point FRP, so it is the fraction
    lost at the grid edge -- non-zero whenever a footprint is clipped, which is the
    common case (the AOI margin is routinely smaller than the 570 m kernel radius).
    On its own it cannot distinguish a correct edge loss from a broken operator: an
    implementation that renormalizes by the in-bounds weights alone drives it to
    ~0 by piling the off-grid mass back onto the boundary cells.

    ``rel_error_vs_expected`` is therefore the real assertion. It compares the
    raster against ``expected_ingrid_mw`` -- each detection's FRP times the in-grid
    fraction of its Gaussian mass, computed independently in
    :func:`_expected_ingrid_splat` -- and must be ~0.

    Returns ``{"point_sum_mw", "raster_integral_mw", "rel_error",
    "expected_ingrid_mw", "rel_error_vs_expected"}`` or ``{}`` when the task
    metadata or fire points are unavailable.
    """
    task = _task_from_event(event_dir)
    if task is None:
        return {}
    try:
        from firedataforge.sources.frp import (
            _load_firepix_data, _load_firms_data, _rasterize_fire_points,
        )
    except ImportError:
        return {}
    # Mirror the pipeline's source selection: with a FEDS perimeter, pre-2025 fires
    # use the bundled firepix archive; otherwise NASA FIRMS.
    has_perim = os.path.exists(os.path.join(event_dir, "burn_perimeter.npy"))
    try:
        df = (_load_firepix_data(task) if has_perim and task.year < 2025
              else _load_firms_data(task))
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    df = df[(df["t"] >= task.t_start) & (df["t"] <= task.t_end)]
    if df.empty:
        return {}
    point_sum = float(df["FRP"].sum())
    # Empty mask lists -> unmasked, isolating the splat from perimeter masking.
    rasters, _ = _rasterize_fire_points(df, task, [], [])
    raster_sum = float(sum(np.nansum(r) for r in rasters))

    # Independent oracle: the in-grid share of each detection's Gaussian mass.
    try:
        transformer = Transformer.from_crs("EPSG:4326", task.crs, always_xy=True)
        x, y = transformer.transform(df["Lon"].values, df["Lat"].values)
        expected = float(np.nansum(_expected_ingrid_splat(
            np.asarray(x), np.asarray(y), df["FRP"].to_numpy(dtype=float),
            task.bounds, task.shape, task.resolution)))
    except Exception:
        expected = None

    return {
        "point_sum_mw": point_sum,
        "raster_integral_mw": raster_sum,
        "rel_error": abs(raster_sum - point_sum) / point_sum if point_sum else None,
        "expected_ingrid_mw": expected,
        "rel_error_vs_expected": (
            abs(raster_sum - expected) / expected if expected else None),
    }


def categorical_agreement(event_dir: str, layer: str = "landcover") -> dict:
    """Overall accuracy between a pipeline categorical layer and a native reference.

    Compares the pipeline output against a majority-aggregated native-resolution
    reference on the same grid (so the check is *not* tautological for
    nearest-neighbor resampling). Overall accuracy is the fraction of pixels where the
    nearest-neighbor pick matches the dominant native class; its complement
    ``1 - overall_accuracy`` is the resampling disagreement rate that this check targets.

    We report overall accuracy rather than Cohen's kappa: kappa's chance correction is
    confounded by class prevalence (Pontius & Millones 2011, "Death to Kappa"), which
    deflates the score on homogeneous fire-region grids in a way unrelated to resampling
    fidelity, whereas overall accuracy is exactly the disagreement probability we want.

    Returns ``{"overall_accuracy", "n"}`` or ``{}`` when no reference is
    available (e.g. Earth Engine unconfigured, or no grid georeference).
    """
    out = _load(event_dir, layer)
    coords = _load(event_dir, "coordinates")
    if out is None or coords is None:
        return {}
    ref = _native_reference(layer, coords.georeference, "mode")
    if ref is None:
        return {}  # no native reference available
    a = np.asarray(out.data[0]).ravel()
    b = np.asarray(ref).ravel()
    # Overall accuracy: fraction of pixels where the pipeline class matches the
    # majority-aggregated native class.
    return {"overall_accuracy": float(np.mean(a == b)), "n": int(a.size)}


def continuous_rmse(event_dir: str, layer: str = "elevation") -> dict:
    """RMSE/MAE of a continuous layer vs. a native reference aggregated to the grid.

    Compares the pipeline output (a bilinear point-sample of the native source)
    against the native 3DEP DEM mean-aggregated to the target resolution; the
    residual reflects how much sub-pixel terrain variance the resampling discards.

    Returns ``{"rmse", "mae", "n"}`` or ``{}`` when no reference is available
    (e.g. Earth Engine unconfigured, or no grid georeference).
    """
    out = _load(event_dir, layer)
    coords = _load(event_dir, "coordinates")
    if out is None or coords is None:
        return {}
    ref = _native_reference(layer, coords.georeference, "mean")
    if ref is None:
        return {}
    a = np.asarray(out.data[0], dtype=float)
    b = np.asarray(ref, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    diff = a[mask] - b[mask]
    return {"rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mae": float(np.mean(np.abs(diff))),
            "n": int(mask.sum())}


def _phase_cross_shift(a, b):
    """Sub-pixel (row, col) shift that aligns ``b`` onto ``a`` via phase correlation.

    A windowed cross-power spectrum gives the integer peak; a parabolic fit on its
    neighbours refines it to sub-pixel. Shifts are wrapped into ``[-N/2, N/2)``.
    """
    a = np.nan_to_num(np.asarray(a, float) - np.nanmean(a))
    b = np.nan_to_num(np.asarray(b, float) - np.nanmean(b))
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))  # suppress edges
    fa, fb = np.fft.fft2(a * win), np.fft.fft2(b * win)
    cross = fa * np.conj(fb)
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-12)).real
    pr, pc = (int(i) for i in np.unravel_index(np.argmax(corr), corr.shape))
    h, w = corr.shape

    def _refine(c0, cm, cp):
        denom = cm - 2 * c0 + cp
        return 0.5 * (cm - cp) / denom if denom else 0.0

    sr = pr + _refine(corr[pr, pc], corr[(pr - 1) % h, pc], corr[(pr + 1) % h, pc])
    sc = pc + _refine(corr[pr, pc], corr[pr, (pc - 1) % w], corr[pr, (pc + 1) % w])
    return sr - h if sr > h / 2 else sr, sc - w if sc > w / 2 else sc


def registration_shift(event_dir: str, layer: str = "elevation") -> dict:
    """Sub-pixel registration error (meters) of a pipeline layer vs. its native source.

    Phase-correlates the pipeline layer against the independently re-fetched native
    reference (same grid) and reports the alignment shift, scaled to meters by the
    pixel size. A near-zero shift is direct evidence that the harmonized grid is
    correctly co-registered with the source -- a stronger check than a CRS round
    trip. Uses ``elevation`` by default for its terrain texture.

    Returns ``{"shift_x_m", "shift_y_m", "shift_m"}`` or ``{}`` when no reference is
    available.
    """
    out = _load(event_dir, layer)
    coords = _load(event_dir, "coordinates")
    if out is None or coords is None:
        return {}
    georef = coords.georeference
    ref = _native_reference(layer, georef, "mean")
    if ref is None:
        return {}
    a = np.asarray(out.data[0], float)
    b = np.asarray(ref, float)
    if a.shape != b.shape:
        return {}
    sr, sc = _phase_cross_shift(a, b)
    res = georef.resolution
    return {"shift_x_m": float(sc * res), "shift_y_m": float(sr * res),
            "shift_m": float(np.hypot(sr, sc) * res)}


ALL_METRICS = {
    "reprojection": reprojection_roundtrip_error,
    "frp_conservation": frp_conservation,
    "categorical_landcover": partial(categorical_agreement, layer="landcover"),
    "categorical_wui": partial(categorical_agreement, layer="wui"),
    "continuous_elevation": partial(continuous_rmse, layer="elevation"),
    "registration_elevation": partial(registration_shift, layer="elevation"),
}
