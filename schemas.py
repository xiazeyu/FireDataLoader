"""
Data schemas for Fire Data Loader.

This module contains the core data classes used throughout the fire data
processing pipeline. These classes can be imported independently for use
in other modules or scripts.

Example:
    from schemas import FireInfo, TaskInfo, DataWithMetadata
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class FireInfo:
    """Information about a wildfire event from the FEDS25MTBS dataset.
    
    Attributes:
        event_id: Unique identifier for the fire event.
        name: Human-readable name of the fire incident.
        year: Year the fire occurred.
        acres_burned: Total acres burned by the fire.
        t_start: Start datetime of the fire.
        t_end: End datetime of the fire.
        bounds: Bounding box as (minx, miny, maxx, maxy) in the specified CRS.
        crs: Coordinate reference system for the bounds (default: EPSG:4326).
    """
    event_id: str
    name: str
    year: int
    acres_burned: int
    t_start: datetime
    t_end: datetime
    bounds: tuple[float, float, float, float]
    crs: str = "EPSG:4326"


@dataclass
class TaskInfo:
    """Processing task configuration for a fire event.
    
    Defines the spatial extent, resolution, and projection for data processing.
    
    Attributes:
        event_id: Unique identifier for the fire event.
        name: Human-readable name of the fire incident.
        year: Year the fire occurred.
        t_start: Start datetime of the fire.
        t_end: End datetime of the fire.
        resolution: Spatial resolution in meters.
        bounds: Bounding box as (minx, miny, maxx, maxy) in the target CRS.
        shape: Output raster dimensions as (height, width) in pixels.
        crs: Target coordinate reference system.
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


@dataclass
class DataWithMetadata:
    """Container for processed geospatial data with metadata.
    
    Attributes:
        name: Name identifier for the data layer.
        data: List of numpy arrays (one per timestamp or single static layer).
        timestamps: Optional list of datetimes corresponding to each data array.
        source: Description of the data source.
        resolution: Native resolution of the source data in meters.
        unit: Unit of measurement for the data values.
        note: Additional metadata as key-value pairs.
    """
    name: str
    data: list[Any]
    timestamps: Optional[list[datetime]] = None
    source: Optional[str] = None
    resolution: Optional[int] = None
    unit: Optional[str] = None
    note: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingArgs:
    """Arguments for fire data processing.
    
    Encapsulates all configuration parameters needed for processing fire events.
    Used by both single-event and batch processing functions.
    
    Attributes:
        resolution: Spatial resolution in meters.
        buffer: Buffer distance around fire bounds in meters.
        crs: Target coordinate reference system.
        output_dir: Output directory for saved data.
        interpolation: Number of intermediate frames to interpolate.
        herbie_cache_dir: Directory for caching HRRR GRIB files.
        verbose: Enable verbose logging output.
    """
    resolution: int = 30
    buffer: int = 100
    crs: str = "EPSG:5070"
    output_dir: str = "output"
    interpolation: int = 0
    herbie_cache_dir: str = "./datasets/herbie"
    verbose: bool = False
