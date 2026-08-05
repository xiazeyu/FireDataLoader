"""
Data schemas for FireDataForge.

This module defines the core data contract used throughout the wildfire data
processing pipeline. The classes are dependency-light (standard library only)
so they can be imported independently in other modules, notebooks, or by
downstream consumers of the framework's outputs.

Example:
    from schemas import FireEvent, ProcessingTask, DataLayer, GeoReference
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

SCHEMA_VERSION = "1.1"


@dataclass
class FireEvent:
    """Information about a wildfire event, keyed by its MTBS Event ID.

    Attributes:
        event_id: MTBS Event ID for the fire event.
        name: Human-readable name of the fire incident.
        year: Year the fire occurred.
        acres_burned: Total acres burned by the fire.
        t_start: Start datetime of the fire.
        t_end: End datetime of the fire.
        bounds: Bounding box as (minx, miny, maxx, maxy) in the specified CRS.
        crs: Coordinate reference system for the bounds (default: EPSG:4326).
        aoi_wkt: Convex hull of the fire's true perimeter as a WKT string in
            ``crs``, when a FEDS GeoPackage supplied one.
    """
    event_id: str
    name: str
    year: int
    acres_burned: int
    t_start: datetime
    t_end: datetime
    bounds: tuple[float, float, float, float]
    crs: str = "EPSG:4326"
    aoi_wkt: Optional[str] = None


@dataclass
class ProcessingTask:
    """Processing task configuration for a fire event.

    Defines the spatial extent, resolution, and projection for data processing.

    Attributes:
        event_id: MTBS Event ID for the fire event.
        name: Human-readable name of the fire incident.
        year: Year the fire occurred.
        t_start: Start datetime of the fire.
        t_end: End datetime of the fire.
        resolution: Spatial resolution in meters.
        bounds: Bounding box as (minx, miny, maxx, maxy) in the target CRS.
        shape: Output raster dimensions as (height, width) in pixels.
        crs: Target coordinate reference system.
        t_end_estimated: ``True`` when ``t_end`` is a fallback estimate
            (``t_start`` + a fixed window) because neither a FEDS perimeter nor an
            explicit end date was available; ``False`` when it is observation-derived.
        buffer: Margin in meters added around the fire's extent.
        aoi_mode: How the extent was derived.
    """
    event_id: str
    name: str
    year: int
    t_start: datetime
    t_end: datetime
    resolution: int
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]
    crs: str
    t_end_estimated: bool = False
    buffer: int = 100
    aoi_mode: str = "bbox"


@dataclass
class GeoReference:
    """Georeferencing for the common output grid.

    Captures everything a consumer needs to place the raster arrays in space
    without an external EPSG lookup or network access. Almost every raster layer
    in an event directory is sampled on this exact grid, so a single
    ``GeoReference`` (persisted in ``coordinates.npy``) georeferences all the
    other ``.npy`` layers; per-layer envelopes leave
    :attr:`DataLayer.georeference` as ``None`` and inherit it. The HRRR weather
    layers (``r2``/``u10``/``v10``) are the exception: they share these
    ``bounds`` and ``crs`` but sit on a coarser grid (see
    :attr:`DataLayer.current_resolution`), so derive their grid from the bounds
    and the array's own shape rather than from this ``GeoReference``.

    Attributes:
        crs: Short CRS identifier (e.g. ``"EPSG:5070"``).
        bounds: Grid extent as (minx, miny, maxx, maxy) in ``crs`` units.
        shape: Grid dimensions as (height, width) in pixels.
        resolution: Pixel size in ``crs`` units (meters for projected CRSes).
        transform: Affine coefficients (a, b, c, d, e, f) mapping
            pixel (col, row) -> (x, y), following the rasterio/GDAL convention.
        crs_wkt: Full WKT2 definition; a self-contained CRS for archival or
            custom CRSes (works without a PROJ database). ``None`` if the CRS
            could not be resolved.
        crs_proj4: Legacy PROJ.4 string, or ``None``.
        crs_epsg: EPSG code if the CRS maps to one, else ``None``.
    """
    crs: str
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]
    resolution: int
    transform: tuple[float, float, float, float, float, float]
    crs_wkt: Optional[str] = None
    crs_proj4: Optional[str] = None
    crs_epsg: Optional[int] = None


@dataclass
class DataLayer:
    """Universal output envelope for a processed data layer plus its metadata.

    Every persisted output---a raster time series, a static raster, the grid
    coordinate arrays, or a configuration payload---is wrapped in this one type
    so that consumers can load and inspect any layer uniformly, without
    per-source special cases.

    Invariants:
        * ``data`` is a list of payloads. For raster layers each element is an
          array sampled on the common grid (see :class:`GeoReference`); static
          layers hold a single element.
        * When ``timestamps`` is not ``None`` it is *parallel* to ``data``:
          ``data[i]`` is the frame observed at ``timestamps[i]`` and
          ``len(timestamps) == len(data)`` (enforced in :meth:`__post_init__`).
          This pairing is what the consumption-time temporal cursor relies on
          to expose each source at its most recent valid observation.
          ``timestamps`` is ``None`` for layers with no temporal dimension
          (e.g. ``coordinates``).

    Attributes:
        name: Layer identifier; also the output file stem (``<name>.npy``).
        data: List of payloads (see invariants above).
        version: Schema version of this envelope (see :data:`SCHEMA_VERSION`).
        timestamps: Per-frame datetimes parallel to ``data``, or ``None``.
        source: Human-readable provenance / source description.
        native_resolution: The source's native spatial resolution in meters
            (the layer's true information content); ``None`` when not applicable
            (e.g. the ``task_info`` payload).
        current_resolution: Spatial resolution in meters of the grid the array is
            currently sampled on. Usually :attr:`ProcessingTask.resolution` (the
            common output grid); the HRRR weather layers keep their own coarser
            grid (~500 m) and record that here instead. Lets a single ``.npy``
            describe both its true and grid resolution without loading
            ``coordinates.npy``.
        unit: Unit of measurement for the data values.
        categories: For categorical layers, a mapping from integer pixel value
            to class label (e.g. land-cover or WUI classes); ``None`` otherwise.
        georeference: Grid georeferencing; populated for the layer that
            georeferences the event (``coordinates``), ``None`` for layers that
            inherit it.
        note: Free-form, source-specific metadata that has no dedicated field
            (e.g. processing parameters, data-gap logs, descriptions).
    """
    name: str
    data: list[Any]
    version: str = SCHEMA_VERSION
    timestamps: Optional[list[datetime]] = None
    source: Optional[str] = None
    native_resolution: Optional[int] = None
    current_resolution: Optional[int] = None
    unit: Optional[str] = None
    categories: Optional[dict[int, str]] = None
    georeference: Optional[GeoReference] = None
    note: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamps is not None and len(self.timestamps) != len(self.data):
            raise ValueError(
                f"DataLayer('{self.name}'): timestamps and data must be "
                f"parallel, got {len(self.timestamps)} timestamps for "
                f"{len(self.data)} data frame(s)."
            )


@dataclass
class ProcessingArgs:
    """Arguments for fire data processing.

    Encapsulates all configuration parameters needed for processing fire events.
    Used by both single-event and batch processing functions.

    Attributes:
        resolution: Spatial resolution in meters.
        buffer: Margin in meters added around the fire's extent.
        aoi_mode: ``"tight"`` builds the grid from the fire's true projected
            perimeter; ``"bbox"`` uses the legacy lon/lat envelope, whose
            axis-aligned extent in the target CRS is inflated by the grid
            convergence. Events without perimeter geometry always use ``"bbox"``.
        crs: Target coordinate reference system.
        output_dir: Output directory for saved data.
        interpolation: Number of intermediate frames to interpolate.
        cache_dir: Root directory for all on-the-fly downloads. Each source
            caches under its own fixed subfolder of this root (e.g.
            ``<cache_dir>/herbie``, ``<cache_dir>/FIRMS``,
            ``<cache_dir>/FEDS25MTBS``), so pointing this at a new location
            relocates the entire software-managed cache at once.
        verbose: Enable verbose logging output.
        only: Optional list of feature names to process (None = all features).
        layer_workers: Max concurrent layer downloads within a single event.
    """
    resolution: int = 30
    buffer: int = 600
    aoi_mode: str = "tight"
    crs: str = "EPSG:5070"
    output_dir: str = "output"
    interpolation: int = 0
    cache_dir: str = "cache"
    verbose: bool = False
    only: Optional[list[str]] = None
    layer_workers: int = 5
