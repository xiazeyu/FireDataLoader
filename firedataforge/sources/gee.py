"""Google Earth Engine layers: 3DEP, LANDFIRE, GBA, WorldCover, LAI,
Sentinel-2 mosaic, and the derived terrain RGB."""

import logging
import math
import os
from dataclasses import replace
from datetime import timedelta
from typing import Literal

import ee
import geemap
import numpy as np
from pyproj import Transformer

from firedataforge.schemas import DataLayer, ProcessingTask

log = logging.getLogger(__name__)


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
    # Use the project persisted during setup (.env / `earthengine set_project`).
    project = os.environ.get("EARTHENGINE_PROJECT") or None

    def _init() -> None:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()

    try:
        # Check if already initialized
        ee.Number(1).getInfo()
    except Exception:
        log.info("Earth Engine not initialized. Attempting to initialize...")
        try:
            _init()
        except Exception as e:
            error_msg = str(e).lower()
            if "project" in error_msg or "quota" in error_msg or "credentials" in error_msg:
                log.info("Authentication required. A URL will be printed; "
                         "open it, authorize, and paste the code back.")
                # auth_mode="notebook" avoids the gcloud CLI (absent on headless
                # /HPC nodes) and works without a local browser.
                ee.Authenticate(auth_mode="notebook")
                try:
                    _init()
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
    task_info: ProcessingTask,
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
    task_info: ProcessingTask,
    dataset_name: str,
    imagecollection: str | list[ee.Image],
    band: str,
    resample: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear'
) -> DataLayer:
    """Download data from a Google Earth Engine ImageCollection.
    
    Args:
        task_info: Task configuration with bounds and resolution.
        dataset_name: Name for the output data layer.
        imagecollection: GEE ImageCollection path or list of ee.Image objects.
        band: Band name to extract.
        resample: Resampling method ('nearest', 'bilinear', or 'bicubic').
    
    Returns:
        DataLayer containing the downloaded array.
    
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

    return DataLayer(
        name=dataset_name,
        data=[data_array],
        timestamps=[task_info.t_start],
        source=f"Google Earth Engine: {imagecollection}",
    )


# =============================================================================
# Data Download Functions
# =============================================================================

def download_usgs(task_info: ProcessingTask) -> DataLayer:
    """Download USGS 3DEP 1m elevation data.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataLayer containing elevation in meters (int16).
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
        native_resolution=1,
        unit="m",
    )


def download_landfire(task_info: ProcessingTask) -> list[DataLayer]:
    """Download LANDFIRE fuel data (Canopy Bulk Density and Canopy Cover).
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        List of DataLayer for CBD and CC layers.
    """
    log.info(f"Downloading LANDFIRE data for event_id: {task_info.event_id}")
    payload = []

    data = download_gee_task(
        task_info,
        dataset_name="canopy_bulk_density",
        band="CBD",
        imagecollection="projects/sat-io/open-datasets/landfire/FUEL/CBD",
        resample='bilinear',
    )
    payload.append(replace(
        data,
        data=[np.around(data.data[0]).astype(np.int16)],
        native_resolution=30,
        unit="100kg/m^3",
    ))

    data = download_gee_task(
        task_info,
        dataset_name="canopy_cover",
        band="CC",
        imagecollection="projects/sat-io/open-datasets/landfire/FUEL/CC",
        resample='bilinear',
    )
    payload.append(replace(
        data,
        data=[np.around(data.data[0]).astype(np.int16)],
        native_resolution=30,
        unit="%",
    ))

    return payload


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


def download_building_height(task_info: ProcessingTask) -> DataLayer:
    """Download building heights from the Global Building Atlas.
    
    Uses area-weighted averaging when multiple buildings fall within a single
    pixel, where larger footprint buildings contribute more to the average.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataLayer containing building heights in meters.
    
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

    # Filter to ROI and valid buildings. The Global Building Atlas has a dense
    # noise floor of spurious low-height footprints; heights at or below ~3.97 m
    # are dominated by this reconstruction artifact rather than real low buildings,
    # so dropping them (and null heights) removes a uniform background bias without
    # losing genuine structures.
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

    return DataLayer(
        name=dataset_name,
        data=[data_array],
        timestamps=[task_info.t_start],
        source=f"Global Building Atlas (Tiles: {len(tile_paths)})",
        native_resolution=3,
        unit="m",
        note={'aggregation': 'area-weighted average'},
    )


def download_eca(task_info: ProcessingTask) -> DataLayer:
    """Download ESA WorldCover land cover classification.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataLayer containing land cover classes (int16).
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
        native_resolution=10,
        unit="class",
        categories={
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
        },
    )


def download_tc(task_info: ProcessingTask) -> DataLayer:
    """Download Tree Canopy Leaf Area Index (LAI) data.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataLayer containing LAI values in m²/m².
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
        'LAI_Grid_30deg_56_2020-07-02',
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

    # The source asset stores LAI scaled by 10 (integer 0-~80); divide to recover
    # physical LAI in m2/m2 (typically 0-8).
    return replace(
        data,
        data=[data_array.astype(np.float32) / 10.0 for data_array in data.data],
        native_resolution=10,
        unit="m2/m2",
    )


# =============================================================================
# Satellite Imagery Functions
# =============================================================================

def download_sentinel2_rgb(task_info: ProcessingTask) -> DataLayer:
    """Download sentinel2_rgb imagery (RGB) from the Sentinel-2 L2A Cloudless Mosaic.
    
    Builds a cloud-free RGB composite from the Copernicus Sentinel-2 Surface
    Reflectance (L2A) Harmonized collection at 10m native resolution by
    filtering low-cloud scenes around the fire start date and taking the
    per-pixel median.
    
    Args:
        task_info: Task configuration with bounds and resolution.
    
    Returns:
        DataLayer containing RGB sentinel2_rgb imagery as (H, W, 3) array.
    """
    log.info(f"Downloading Sentinel-2 L2A cloudless mosaic for event_id: {task_info.event_id}")
    _ensure_ee_initialized()
    
    roi = ee.Geometry.Rectangle(task_info.bounds, task_info.crs, False)
    native_res = 10
    source = "Copernicus Sentinel-2 L2A Cloudless Mosaic"
    
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

    if s2.size().getInfo() == 0:
        # No scene covers this event (e.g. pre-2015 fires predate Sentinel-2, or
        # no sufficiently cloud-free scene exists). Skip cleanly rather than
        # failing on the empty-collection median. Imported here to avoid a
        # gee <-> pipeline import cycle.
        from firedataforge.pipeline import LayerUnavailable
        raise LayerUnavailable(
            f"No Sentinel-2 imagery for {task_info.event_id} ({task_info.year}); "
            "the Sentinel-2 archive starts 2015"
        )

    image = s2.median().rename(['R', 'G', 'B'])
    # Scale Sentinel-2 values to 0-255 range (divide by 3000 for brighter visualization)
    image = image.divide(3000).multiply(255).clamp(0, 255)
    
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
    
    log.info(f"✓ Downloaded sentinel2_rgb imagery: {rgb_array.shape}")
    
    return DataLayer(
        name="sentinel2_rgb",
        data=[rgb_array],
        timestamps=[task_info.t_start],
        source=source,
        native_resolution=native_res,
        unit="RGB (0-255)",
        note={'description': 'True color sentinel2_rgb imagery'},
    )


_TERRAIN_RGB_PALETTE: list[str] = [
    '3a8d52',  # low green
    '6cb27a',
    'a3c585',
    'e8e1a8',  # tan
    'c2a384',
    '8b6f4e',  # brown
    'd9d9d9',
    'ffffff',  # snow
]


def download_terrain_rgb(task_info: ProcessingTask) -> DataLayer:
    """Download a colored shaded-relief terrain visualization (Google-Maps style).

    Combines USGS 3DEP elevation (hypsometric color tint) with a hillshade
    (multiply blend) to produce an RGB image similar to the Google Maps
    "Terrain" view. Returns a uint8 array of shape (H, W, 3).

    Args:
        task_info: Task configuration with bounds and resolution.

    Returns:
        DataLayer containing an RGB array (H, W, 3) of uint8 values.
    """
    log.info(f"Downloading terrain RGB for event_id: {task_info.event_id}")
    _ensure_ee_initialized()

    roi = ee.Geometry.Rectangle(task_info.bounds, task_info.crs, False)

    collection = ee.ImageCollection("USGS/3DEP/1m").filterBounds(roi)
    native_proj = collection.first().select('elevation').projection()
    elevation = collection.mosaic().select('elevation').setDefaultProjection(native_proj)
    native_res = 1

    # Compute local elevation stretch so the palette uses the full dynamic
    # range visible in the scene, not the global one.
    stats = elevation.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=roi,
        scale=max(task_info.resolution, 30),
        maxPixels=int(1e10),
        bestEffort=True,
    ).getInfo() or {}
    emin = stats.get('elevation_min')
    emax = stats.get('elevation_max')
    if emin is None or emax is None or emax <= emin:
        # Fall back to a sensible default range for very flat areas.
        emin = float(emin) if emin is not None else 0.0
        emax = emin + 1.0
    log.info(f"Terrain RGB elevation stretch: [{emin:.1f}, {emax:.1f}] m")

    # Hypsometric tint: visualize() produces a 3-band uint8 RGB image.
    tint = elevation.visualize(
        min=float(emin),
        max=float(emax),
        palette=_TERRAIN_RGB_PALETTE,
    )

    # Standard cartographic lighting (azimuth=315° NW, sun elevation=45°).
    hillshade = ee.Terrain.hillshade(elevation, azimuth=315, elevation=45)

    # Multiply blend: shaded = tint * (hillshade / 255). Done per-band so the
    # output stays a 3-band uint8 image.
    shade = hillshade.divide(255.0)
    shaded = tint.multiply(shade).clamp(0, 255).rename(['R', 'G', 'B'])
    shaded = shaded.resample('bilinear')

    rgb_data = []
    for band in ['R', 'G', 'B']:
        band_image = shaded.select(band).reproject(
            crs=task_info.crs,
            scale=task_info.resolution,
        )
        data_array = _download_processed_image(band_image, task_info, band)
        rgb_data.append(data_array)

    rgb_array = np.stack(rgb_data, axis=-1).clip(0, 255).astype(np.uint8)

    log.info(f"✓ Downloaded terrain RGB: {rgb_array.shape}")

    return DataLayer(
        name="terrain_rgb",
        data=[rgb_array],
        timestamps=[task_info.t_start],
        source="Computed from USGS 3DEP via Google Earth Engine",
        native_resolution=native_res,
        unit="RGB (0-255)",
        note={
            'description': (
                'Colored shaded-relief (hypsometric tint × hillshade), '
                'Google-Maps terrain style'
            ),
            'elevation_stretch_m': [float(emin), float(emax)],
            'hillshade_azimuth_deg': 315,
            'hillshade_elevation_deg': 45,
            'palette': _TERRAIN_RGB_PALETTE,
        },
    )
