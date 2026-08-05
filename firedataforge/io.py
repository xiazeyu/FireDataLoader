"""Persistence of harmonized layers (``.npz``) and grid coordinates.

Each layer is one compressed ``.npz`` under ``output_dir/<event_id>/``, holding
raw arrays alongside a single JSON metadata string. Nothing is pickled, so
consumers load an event with a plain ``np.load`` -- no ``allow_pickle``, no need
to import this package, and no code execution from data files.

Inside a layer file:

- ``data`` -- the frames stacked into one ``(T, ...)`` array, when every frame
  shares a shape and dtype (the common case: ``(T, H, W)``).
- ``data_000``, ``data_001``, ... -- one array per frame, when frames differ in
  shape (e.g. ``coordinates``, whose x and y axes have different lengths).
- ``meta`` -- a 0-d string array holding the JSON-encoded :class:`DataLayer`
  metadata (``timestamps``, ``source``, ``categories``, ``georeference``,
  ``note``, ...) plus a ``layout`` field naming which of the two above applies.
  Layers whose payload is not an array at all (``task_info``) carry it here too,
  under ``layout="json"``.

``datetime``, ``tuple``, and integer-keyed ``dict`` values round-trip exactly
via a small tagged-JSON codec (see :func:`_encode` / :func:`_decode`), so a
loaded :class:`DataLayer` is indistinguishable from the saved one.

Reading a layer without this package installed:

    import json, numpy as np
    f = np.load("output/<event_id>/elevation.npz")
    cube = f["data"]                       # (T, H, W)
    meta = json.loads(str(f["meta"]))      # source, unit, note, ...

FireDataForge 0.1 wrote a pickled ``.npy`` per layer instead. Those files are
still readable (see :func:`load_numpy`); :func:`convert_legacy_dir` rewrites an
existing output tree in the current format.
"""

import glob
import json
import logging
import os
from dataclasses import asdict, fields, replace
from datetime import datetime
from typing import Any

import numpy as np

from firedataforge.schemas import SCHEMA_VERSION, DataLayer, GeoReference, ProcessingTask

log = logging.getLogger(__name__)

#: Extension for layer files written by :func:`save_numpy`.
DATA_EXT = ".npz"

#: FireDataForge 0.1 extension: a pickled ``asdict(DataLayer)`` in a 0-d object
#: array. Still readable, so existing output trees keep working.
LEGACY_EXT = ".npy"

#: Layers that describe the event rather than carrying a raster of their own.
NON_RASTER_LAYERS = ("task_info", "coordinates")

# Marker key for the tagged-JSON codec. Namespaced so it cannot collide with a
# real key in a source's free-form ``note`` dict.
_TYPE_KEY = "__fdf_type__"


# --------------------------------------------------------------------------
# Tagged-JSON codec: exact round-trip for the types that appear in DataLayer
# metadata, using only JSON primitives (no pickle).
# --------------------------------------------------------------------------

def _encode(obj: Any) -> Any:
    """Convert ``obj`` into JSON-serializable form, tagging non-JSON types."""
    if isinstance(obj, datetime):
        return {_TYPE_KEY: "datetime", "value": obj.isoformat()}
    if isinstance(obj, np.ndarray):
        return {_TYPE_KEY: "ndarray", "dtype": obj.dtype.str,
                "shape": list(obj.shape), "value": obj.tolist()}
    if isinstance(obj, np.generic):  # numpy scalar -> its Python equivalent
        return _encode(obj.item())
    if isinstance(obj, tuple):
        return {_TYPE_KEY: "tuple", "value": [_encode(v) for v in obj]}
    if isinstance(obj, dict):
        # JSON object keys are always strings; a dict keyed by anything else
        # (``categories`` is keyed by int) is encoded as an item list instead.
        if all(isinstance(k, str) for k in obj):
            return {k: _encode(v) for k, v in obj.items()}
        return {_TYPE_KEY: "dict",
                "items": [[_encode(k), _encode(v)] for k, v in obj.items()]}
    if isinstance(obj, list):
        return [_encode(v) for v in obj]
    return obj  # str / int / float / bool / None


def _decode(obj: Any) -> Any:
    """Inverse of :func:`_encode`."""
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if isinstance(obj, dict):
        kind = obj.get(_TYPE_KEY)
        if kind == "datetime":
            return datetime.fromisoformat(obj["value"])
        if kind == "tuple":
            return tuple(_decode(v) for v in obj["value"])
        if kind == "ndarray":
            return np.asarray(obj["value"],
                              dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        if kind == "dict":
            return {_decode(k): _decode(v) for k, v in obj["items"]}
        return {k: _decode(v) for k, v in obj.items()}
    return obj


def _pack_frames(frames: list[Any]) -> tuple[dict[str, np.ndarray], str]:
    """Split a ``DataLayer.data`` list into ``.npz`` arrays plus a layout tag."""
    if frames and all(isinstance(f, np.ndarray) for f in frames):
        first = frames[0]
        if all(f.shape == first.shape and f.dtype == first.dtype for f in frames):
            return {"data": np.stack(frames)}, "stacked"
        return {f"data_{i:03d}": f for i, f in enumerate(frames)}, "ragged"
    # Non-array payloads (e.g. task_info's config dict) ride along in the JSON.
    return {}, "json"


def _write_layer(path: str, layer: DataLayer) -> None:
    """Write ``layer`` to ``path`` as a compressed, pickle-free ``.npz``."""
    # Build the metadata dict field-by-field rather than via asdict(), which
    # would deep-copy every frame just to have it thrown away by _pack_frames.
    meta = {f.name: getattr(layer, f.name) for f in fields(layer) if f.name != "data"}
    if layer.georeference is not None:
        meta["georeference"] = asdict(layer.georeference)

    arrays, layout = _pack_frames(layer.data)
    meta["layout"] = layout
    meta["n_frames"] = len(layer.data)
    if layout == "json":
        meta["data"] = layer.data

    np.savez_compressed(path, meta=np.array(json.dumps(_encode(meta))), **arrays)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def save_numpy(
    task_info: ProcessingTask,
    data: DataLayer,
    output_dir: str = 'output'
) -> None:
    """Save a processed layer to a compressed ``.npz``.

    Creates a directory structure: ``output_dir/event_id/data_name.npz``

    Args:
        task_info: Task configuration containing event_id.
        data: Data to save with metadata.
        output_dir: Base output directory.
    """
    event_id = task_info.event_id
    output_path = os.path.join(output_dir, event_id)
    os.makedirs(output_path, exist_ok=True)

    output_path = os.path.join(output_path, f"{data.name}{DATA_EXT}")
    # Stamp the resolution of the grid this layer currently sits on, so each file
    # is self-describing (native_resolution is the source's own resolution). Most
    # layers are resampled to the task grid; a builder that targets a different
    # grid (e.g. the coarser HRRR weather grid) sets current_resolution itself,
    # so only fill it in when the builder left it unset.
    if data.current_resolution is None:
        data = replace(data, current_resolution=task_info.resolution)

    _write_layer(output_path, data)

    log.info(f"Saved {data.name} data to {output_path}")


def resolve_path(filepath: str) -> str:
    """Return ``filepath``, falling back to its sibling ``.npz``/``.npy`` twin.

    Lets callers (and users with 0.1 output trees) name a layer with either
    extension and get whichever one is actually on disk.
    """
    if os.path.exists(filepath):
        return filepath
    stem, ext = os.path.splitext(filepath)
    if ext in (DATA_EXT, LEGACY_EXT):
        for alt in (DATA_EXT, LEGACY_EXT):
            if os.path.exists(stem + alt):
                return stem + alt
    return filepath  # let the loader raise a normal FileNotFoundError


def layer_files(event_dir: str, include_non_raster: bool = False) -> list[str]:
    """List an event directory's layer files, current format preferred.

    Args:
        event_dir: Path to one ``output_dir/<event_id>/`` directory.
        include_non_raster: Keep ``task_info`` / ``coordinates``, which describe
            the event rather than holding a raster of their own.

    Returns:
        Paths sorted by layer name. When both a ``.npz`` and a legacy ``.npy``
        exist for the same layer, only the ``.npz`` is returned.
    """
    found: dict[str, str] = {}
    for ext in (LEGACY_EXT, DATA_EXT):  # .npz second, so it wins
        for path in sorted(glob.glob(os.path.join(event_dir, f"*{ext}"))):
            name = os.path.splitext(os.path.basename(path))[0]
            if not include_non_raster and name in NON_RASTER_LAYERS:
                continue
            found[name] = path
    return [found[name] for name in sorted(found)]


def _load_legacy_npy(filepath: str) -> DataLayer:
    """Load a FireDataForge 0.1 pickled ``.npy`` layer.

    Kept so 0.1 output trees stay readable. This path requires
    ``allow_pickle=True`` and therefore only belongs on files you generated
    yourself -- never on a layer file from an untrusted source. Re-run the event,
    or use :func:`convert_legacy_dir`, to move to the pickle-free ``.npz``.
    """
    loaded_dict = np.load(filepath, allow_pickle=True).item()
    # asdict() flattened nested dataclasses to dicts on save; rehydrate the
    # typed GeoReference here so loads round-trip to the same types as saves.
    geo = loaded_dict.get("georeference")
    if isinstance(geo, dict):
        loaded_dict["georeference"] = GeoReference(**geo)
    return DataLayer(**loaded_dict)


def load_numpy(filepath: str) -> DataLayer:
    """Load a processed layer from disk.

    Reads the ``.npz`` format written by :func:`save_numpy` **without**
    ``allow_pickle``. FireDataForge 0.1's pickled ``.npy`` files are still
    accepted (see :func:`_load_legacy_npy`); either extension resolves to
    whichever file exists, so ``load_numpy('.../elevation.npy')`` finds
    ``elevation.npz``.

    Args:
        filepath: Path to the layer file.

    Returns:
        DataLayer object with loaded data.
    """
    path = resolve_path(filepath)
    if path.endswith(LEGACY_EXT):
        return _load_legacy_npy(path)

    with np.load(path) as handle:  # note: no allow_pickle
        meta = _decode(json.loads(handle["meta"].item()))
        layout = meta.pop("layout", "stacked")
        meta.pop("n_frames", None)
        if layout == "stacked":
            # Iterating the stacked cube yields per-frame views, restoring the
            # list-of-frames shape DataLayer.data promises.
            meta["data"] = list(handle["data"])
        elif layout == "ragged":
            keys = sorted(k for k in handle.files if k.startswith("data_"))
            meta["data"] = [handle[k] for k in keys]
        elif "data" not in meta:
            meta["data"] = []

    geo = meta.get("georeference")
    if isinstance(geo, dict):
        meta["georeference"] = GeoReference(**geo)
    return DataLayer(**meta)


def convert_legacy_dir(root: str, remove_originals: bool = False) -> int:
    """Rewrite every FireDataForge 0.1 ``.npy`` layer under ``root`` as ``.npz``.

    Walks ``root`` recursively, so it accepts either one event directory or a
    whole ``output/`` tree. Layers that already have a ``.npz`` are skipped.

    Args:
        root: Event directory, or a directory of event directories.
        remove_originals: Delete each ``.npy`` once its ``.npz`` is written.

    Returns:
        Number of layers converted.
    """
    converted = 0
    for path in sorted(glob.glob(os.path.join(root, "**", f"*{LEGACY_EXT}"),
                                 recursive=True)):
        target = os.path.splitext(path)[0] + DATA_EXT
        if os.path.exists(target):
            continue
        # Re-stamp the envelope version: the fields are unchanged from 1.1, but
        # the file now uses the 2.0 container, and a consumer reads `version` to
        # know which to expect.
        _write_layer(target, replace(_load_legacy_npy(path),
                                     version=SCHEMA_VERSION))
        if remove_originals:
            os.remove(path)
        converted += 1
        log.info(f"Converted {path} -> {target}")
    return converted


def save_coordinates(
    task_info: ProcessingTask,
    output_dir: str = 'output'
) -> None:
    """Save pixel-center coordinate arrays and CRS for the task grid.

    Writes ``coordinates.npz`` into ``output_dir/event_id/`` containing a
    :class:`DataLayer` with:

    - ``data[0]``: 1-D array of x (easting/longitude) pixel-center coordinates,
      shape ``(width,)``.
    - ``data[1]``: 1-D array of y (northing/latitude) pixel-center coordinates,
      shape ``(height,)``, ordered top-to-bottom to match raster row order.
    - ``georeference``: a :class:`~firedataforge.schemas.GeoReference` with ``crs`` (short id,
      e.g. ``"EPSG:5070"``), ``crs_wkt`` / ``crs_proj4`` / ``crs_epsg``
      (self-contained CRS definitions for archival / custom-CRS use),
      ``bounds`` (minx, miny, maxx, maxy), ``shape`` (height, width),
      ``resolution``, and ``transform`` (affine coefficients
      ``a, b, c, d, e, f``).

    The two axes have different lengths, so this layer uses the ``ragged``
    layout on disk: ``data_000`` is x and ``data_001`` is y.

    These coordinates correspond to the same grid every other layer is sampled
    on, so researchers can wrap arrays directly into ``xarray`` or re-project
    them using the saved CRS for publication-quality figures.

    Args:
        task_info: Task configuration with ``bounds``, ``shape``, and ``crs``.
        output_dir: Base output directory.
    """
    minx, miny, maxx, maxy = task_info.bounds
    height, width = task_info.shape

    px = (maxx - minx) / width if width else 0.0
    py = (maxy - miny) / height if height else 0.0

    # Pixel-center coordinates. Y is top-to-bottom (north -> south) to match
    # rasterio/numpy row-major raster ordering used elsewhere in the pipeline.
    x = minx + (np.arange(width) + 0.5) * px
    y = maxy - (np.arange(height) + 0.5) * py

    # Affine transform (from_origin equivalent): pixel (col, row) -> (x, y).
    transform = (px, 0.0, minx, 0.0, -py, maxy)

    # Self-contained CRS info so consumers don't need an EPSG lookup or
    # network access to reconstruct the projection (useful for archival,
    # custom CRSes, or environments without a PROJ database).
    try:
        from pyproj import CRS as _CRS
        _crs_obj = _CRS.from_user_input(task_info.crs)
        crs_wkt = _crs_obj.to_wkt()
        crs_proj4 = _crs_obj.to_proj4()
        crs_epsg = _crs_obj.to_epsg()
    except Exception:  # pragma: no cover - defensive only
        crs_wkt = None
        crs_proj4 = None
        crs_epsg = None

    coords = DataLayer(
        name="coordinates",
        data=[x, y],
        timestamps=None,
        source="Derived from ProcessingTask grid",
        native_resolution=task_info.resolution,
        unit="CRS units",
        georeference=GeoReference(
            crs=task_info.crs,
            bounds=task_info.bounds,
            shape=task_info.shape,
            resolution=task_info.resolution,
            transform=transform,
            crs_wkt=crs_wkt,
            crs_proj4=crs_proj4,
            crs_epsg=crs_epsg,
        ),
        note={"axes": ["x (width)", "y (height, top-to-bottom)"]},
    )

    save_numpy(task_info, coords, output_dir)
