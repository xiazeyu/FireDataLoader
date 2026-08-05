"""Convert saved ``.npz`` layers into GeoTIFFs for GIS desktop tools.

The ``.npz`` outputs are the canonical product: compressed, self-describing, and
a one-line ``np.load`` for downstream ML code. But they are opaque to QGIS and
ArcGIS, which is where a lot of fire-science work happens. This module is a
post-hoc converter -- it reads an event directory the pipeline already produced
and writes ``<event_id>/geotiff/<name>.tif`` beside it, so the pipeline's own
outputs are unchanged. See ``to_geotiff.py`` for the command-line entry point.

The conversion is lossy in one direction only: GeoTIFF carries the pixels, the
CRS, and the affine transform, but not the rich :class:`~firedataforge.schemas.DataLayer`
envelope. Timestamps survive as per-band descriptions and a few scalar fields
land in GDAL metadata tags; anything needing the full envelope (``note``,
``categories``, provenance) should read the ``.npz``.

Every frame of a time-series layer becomes one band, so a ``frp_daytime`` with
9 frames is a 9-band GeoTIFF whose band descriptions are the observation times.
"""

import logging
import os
from typing import Any, Optional

import numpy as np

from firedataforge.schemas import DataLayer, ProcessingTask

log = logging.getLogger(__name__)

# rasterio has no bool or 64-bit integer raster type; map those to the nearest
# type it can write. Everything else passes through unchanged.
_DTYPE_FALLBACK = {
    "bool": "uint8",
    "int64": "int32",
    "uint64": "uint32",
}


def _to_band_stack(frames: list[Any]) -> Optional[tuple[np.ndarray, list[str]]]:
    """Flatten a layer's frames into ``(bands, H, W)`` plus per-band labels.

    Returns ``None`` for layers that are not rasters on a 2-D grid (the
    ``task_info`` config payload, the 1-D ``coordinates`` axes).
    """
    if not frames or not all(isinstance(f, np.ndarray) for f in frames):
        return None
    if len({f.shape for f in frames}) != 1:
        return None
    shape = frames[0].shape

    if len(shape) == 2:  # (H, W) per frame -> one band each
        return np.stack(frames), [f"frame {i}" for i in range(len(frames))]
    if len(shape) == 3 and shape[2] in (3, 4):  # (H, W, C) -> C bands each
        channels = "RGBA"[:shape[2]]
        stack = np.concatenate([np.moveaxis(f, 2, 0) for f in frames])
        labels = [f"frame {i} {channels[c]}"
                  for i in range(len(frames)) for c in range(shape[2])]
        return stack, labels
    return None


def export_layer(
    task_info: ProcessingTask,
    layer: DataLayer,
    output_dir: str = "output",
) -> Optional[str]:
    """Write one layer as a multi-band GeoTIFF under ``<event_id>/geotiff/``.

    Args:
        task_info: Task configuration; supplies the event id, CRS, and extent.
        layer: The layer to convert.
        output_dir: Base output directory.

    Returns:
        Path to the written ``.tif``, or ``None`` if the layer holds no
        griddable raster (``task_info``, ``coordinates``).
    """
    import rasterio
    from rasterio.transform import from_bounds

    packed = _to_band_stack(layer.data)
    if packed is None:
        return None
    stack, labels = packed

    dtype = _DTYPE_FALLBACK.get(stack.dtype.name, stack.dtype.name)
    stack = stack.astype(dtype, copy=False)

    # Derive the transform from the layer's own array shape rather than the task
    # grid: the HRRR weather layers share these bounds but sit on a coarser grid.
    minx, miny, maxx, maxy = task_info.bounds
    bands, height, width = stack.shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    out_dir = os.path.join(output_dir, task_info.event_id, "geotiff")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{layer.name}.tif")

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": bands,
        "dtype": dtype, "crs": task_info.crs, "transform": transform,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "compress": "deflate", "predictor": 2 if dtype.startswith("f") else 1,
    }
    if dtype.startswith("float"):
        profile["nodata"] = float("nan")

    # Prefer the timestamps as band names; fall back to positional labels for
    # static layers and for the extra channels of an RGB frame.
    if layer.timestamps is not None and len(layer.timestamps) == bands:
        labels = [t.isoformat() for t in layer.timestamps]

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        for i, label in enumerate(labels, start=1):
            dst.set_band_description(i, label)
        tags = {"layer": layer.name}
        for field in ("source", "unit", "native_resolution", "current_resolution"):
            value = getattr(layer, field, None)
            if value is not None:
                tags[field] = str(value)
        dst.update_tags(**tags)

    log.info(f"Saved {layer.name} GeoTIFF to {path}")
    return path


def export_event_dir(event_dir: str, output_dir: Optional[str] = None) -> int:
    """Convert every layer in one event directory to GeoTIFF.

    Args:
        event_dir: Path to one ``output/<event_id>/`` directory.
        output_dir: Base directory to write under; defaults to ``event_dir``'s
            parent, i.e. the GeoTIFFs land in ``<event_dir>/geotiff/``.

    Returns:
        Number of layers converted.
    """
    from firedataforge.io import layer_files, load_numpy

    if output_dir is None:
        output_dir = os.path.dirname(os.path.normpath(event_dir)) or "."

    info = load_numpy(os.path.join(event_dir, "task_info.npz"))
    task_info = ProcessingTask(**info.data[0])

    count = 0
    for path in layer_files(event_dir):
        if export_layer(task_info, load_numpy(path), output_dir) is not None:
            count += 1
    return count


def export_tree(root: str, output_dir: Optional[str] = None) -> dict[str, int]:
    """Convert every event directory under ``root``.

    Accepts either a single event directory or a whole ``output/`` tree; an
    event directory is anything containing a ``task_info`` layer.

    Args:
        root: Event directory, or a directory of event directories.
        output_dir: Base directory to write under; defaults to ``root``'s own
            layout (each event's GeoTIFFs land beside its ``.npz`` files).

    Returns:
        Mapping of event id to the number of layers converted.
    """
    from firedataforge.io import resolve_path

    def is_event_dir(path: str) -> bool:
        return os.path.exists(resolve_path(os.path.join(path, "task_info.npz")))

    root = os.path.normpath(root)
    if is_event_dir(root):
        candidates = [root]
    else:
        candidates = sorted(
            os.path.join(root, name) for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
            and is_event_dir(os.path.join(root, name))
        )

    results: dict[str, int] = {}
    for event_dir in candidates:
        event_id = os.path.basename(event_dir)
        try:
            results[event_id] = export_event_dir(event_dir, output_dir)
        except Exception as exc:  # keep the batch going
            log.error(f"[{event_id}] GeoTIFF conversion failed: {exc!r}")
            results[event_id] = 0
    return results
