from affine import Affine
from dataclasses import asdict, dataclass, replace, field
from datetime import datetime, timedelta
from herbie import Herbie
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box, MultiPolygon
from tqdm import tqdm
from typing import Any, Literal, Optional
import argparse
import ee
import geemap
import geopandas as gpd
import logging
import math
import numpy as np
import os
import pandas as pd
import rioxarray
import xarray as xr

xr.set_options(use_new_combine_kwarg_defaults=True)

log = logging.getLogger(__name__)
GEE_PROJECT_ID = "annular-haven-474021-v1"
HERBIE_CACHE_DIR = "./datasets/herbie"


@dataclass
class FireInfo:
    event_id: str
    name: str
    year: int
    acres_burned: int
    t_start: datetime
    t_end: datetime
    # (minx, miny, maxx, maxy)
    bounds: tuple[float, float, float, float]
    crs: str = "EPSG:4326"


@dataclass
class TaskInfo:
    event_id: str
    name: str
    year: int
    t_start: datetime
    t_end: datetime
    resolution: int
    # (minx, miny, maxx, maxy)
    bounds: tuple[float, float, float, float]
    # (height, width)
    shape: tuple[int, int]
    crs: str


@dataclass
class DataWithMetadata:
    name: str
    data: list[Any]
    timestamps: Optional[list[datetime]] = None
    source: Optional[str] = None
    resolution: Optional[int] = None
    unit: Optional[str] = None
    note: dict[str, Any] = field(default_factory=dict)


def get_fire_info(event_id: str, firelist_path: str = 'datasets/FEDS25MTBS/fireslist2012-2023.csv') -> FireInfo:
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


def get_task_info(fire_info: FireInfo,
                  resolution: int = 30,
                  buffer: int = 100,
                  crs: str = "EPSG:5070",
                  ) -> TaskInfo:

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
        t_start=fire_info.t_start,
        t_end=fire_info.t_end,
        resolution=resolution,
        bounds=target_bounds,
        shape=(height, width),
        crs=crs,
    )


def save_numpy(task_info: TaskInfo, data: DataWithMetadata, output_dir: str = 'output') -> None:
    event_id = task_info.event_id
    output_path = os.path.join(output_dir, event_id)
    os.makedirs(output_path, exist_ok=True)

    output_path = os.path.join(output_path, f"{data.name}.npy")
    np.save(output_path, asdict(data))

    log.info(f"Saved {data.name} data to {output_path}")


def load_numpy(filepath: str) -> DataWithMetadata:
    loaded_dict = np.load(filepath, allow_pickle=True).item()
    obj = DataWithMetadata(**loaded_dict)
    return obj


def process_feds25mtbs(task_info: TaskInfo, base_dir: str = 'datasets/FEDS25MTBS') -> DataWithMetadata:
    log.info(f"Processing FEDS25MTBS for event_id: {task_info.event_id}")

    data_dir = os.path.join(base_dir, str(
        task_info.year), task_info.event_id + '.gpkg')

    assert os.path.exists(
        data_dir), f"Error: FEDS25MTBS data not found at {data_dir}"

    gdf = gpd.read_file(data_dir, layer='perimeter')

    data_list = []
    timestamps = []

    for _, row in gdf.iterrows():
        timestamp = pd.to_datetime(row['t'])
        geom = row.geometry

        if geom is None:
            continue

        if geom.geom_type == 'MultiPolygon':
            data_list.append(geom)
        elif geom.geom_type == 'Polygon':
            data_list.append(MultiPolygon([geom]))
        else:
            log.warning(
                f"Unexpected geometry type: {geom.geom_type} for event_id: {task_info.event_id}")
            continue

        timestamps.append(timestamp)

    # Calculate Grid Dimensions
    t_minx, t_miny, t_maxx, t_maxy = task_info.bounds

    res = task_info.resolution
    transform = from_origin(t_minx, t_maxy, res, res)

    log.info(f"Target Grid: {task_info.shape} pixels @ {res}m resolution")

    # Process each timestep
    processed_rasters = []

    gdf = gpd.GeoDataFrame({
        'geometry': data_list,
        'timestamp': timestamps
    }, crs="EPSG:4326")

    log.info(
        f"Reprojecting geometries from EPSG:4326 to target CRS {task_info.crs}")
    gdf = gdf.to_crs(task_info.crs)

    # Rasterize
    for _, row in gdf.iterrows():
        # Rasterize creates a numpy array where geometry exists
        # We burn a value of '1' where the polygon is, '0' otherwise
        shapes = [(row.geometry, 1)]

        raster = features.rasterize(
            shapes=shapes,
            out_shape=task_info.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True  # If a pixel touches the polygon, it counts
        )
        assert raster.shape == task_info.shape, "Error: Rasterized shape does not match target shape"
        raster = raster.astype(np.bool)
        processed_rasters.append(raster)

    return DataWithMetadata(
        name="burn_perimeters",
        data=processed_rasters,
        timestamps=timestamps,
        source="'FEDS25MTBS; https://doi.org/10.1038/s41597-022-01343-0; requested via SharePoint by Huilin'",
        resolution=375,
    )


def signed_bwdist(im: np.ndarray) -> np.ndarray:
    '''
    Computes the Signed Distance Field (SDF) for a binary image.

    Pixels inside the shape are positive (distance to boundary).
    Pixels outside the shape are negative (distance to boundary).
    '''
    # Ensure boolean
    im = im.astype(bool)

    # distance_transform_edt calculates distance to the closest ZERO.
    # So for the inner distance, we use the image itself.
    # For the outer distance, we use the inverse of the image.
    inner_dist = distance_transform_edt(im)
    outer_dist = distance_transform_edt(~im)

    # Combine: Inside is positive, Outside is negative.
    # We subtract outer_dist from inner_dist.
    # (Note: Boundary pixels might be close to 0)
    return inner_dist - outer_dist


def interp_shape(array_a: np.ndarray, array_b: np.ndarray, precision: float = 0.5) -> np.ndarray:
    '''
    Interpolate between two contours (boolean masks)

    ref: https://stackoverflow.com/questions/48818373/interpolate-between-two-images

    Input: array_a (top)
            [X,Y] bool - Image of first shape
           array_b (bottom)
            [X,Y] bool - Image of second shape
           precision
             float - 0.0 to 1.0 (0.0 = array_a, 1.0 = array_b)

    Output: out
            [X,Y] bool - Interpolated shape
    '''
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


def interpolate_burn_perimeters(data: DataWithMetadata, multiplier: int) -> DataWithMetadata:
    # Validation: Cannot interpolate if fewer than 2 frames
    if not data.data or len(data.data) < 2:
        return data

    n_original = len(data.data)
    new_data_list = []

    assert (data.timestamps is not None) and (
        len(data.timestamps) == n_original)
    new_timestamps = []

    # Iterate through pairs of existing data
    for i in range(n_original - 1):
        # 1. Get current pair
        curr_frame = data.data[i]
        next_frame = data.data[i+1]

        # 2. Append the start frame of this pair
        new_data_list.append(curr_frame)

        # 3. Handle start timestamp
        curr_time = data.timestamps[i]
        next_time = data.timestamps[i+1]
        new_timestamps.append(curr_time)
        time_diff = next_time - curr_time

        # 4. Generate Intermediates
        # We need 'multiplier' steps.
        # Denominator is multiplier + 1 (e.g., 1 intermediate = 1/2 split)
        steps = multiplier + 1

        for step in range(1, steps):
            # Calculate precision factor (t) between 0 and 1
            t = step / steps

            # Interpolate Shape
            interp_frame = interp_shape(curr_frame, next_frame, precision=t)
            new_data_list.append(interp_frame)

            # Interpolate Time
            # Calculate delta and add to current time
            interp_time = curr_time + (time_diff * t)
            new_timestamps.append(interp_time)

    # 5. Append the very last original frame/timestamp (the end of the chain)
    new_data_list.append(data.data[-1])
    new_timestamps.append(data.timestamps[-1])

    # 6. Return new object, preserving other metadata
    # We use replace() to create a copy with updated fields
    return replace(
        data,
        data=new_data_list,
        timestamps=new_timestamps,
        note=data.note | {'interpolate': multiplier}
    )


def _ensure_ee_initialized() -> None:
    try:
        # Check if initialized by trying a simple operation
        ee.Number(1).getInfo()
    except Exception:
        log.info("Earth Engine not initialized. Attempting to authenticate...")
        try:
            ee.Initialize(project=GEE_PROJECT_ID)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT_ID)


def _download_processed_image(image: ee.Image, task_info: TaskInfo, band_name: str) -> np.ndarray:
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


def download_gee_task(task_info: TaskInfo, dataset_name: str, imagecollection: str | list[ee.Image], band: str, resample: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear') -> DataWithMetadata:
    # only works for image collections in GEE, not for feature collections

    log.info(
        f"Downloading {dataset_name} data for event_id: {task_info.event_id} from Google Earth Engine")

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


def download_usgs(task_info: TaskInfo) -> DataWithMetadata:
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


def clip_hrrr_to_task(hrrr_data: xr.Dataset, task_info: TaskInfo, target_resolution: int = 500) -> xr.Dataset:

    assert "x" in hrrr_data.dims and "y" in hrrr_data.dims, "HRRR data must have 'x' and 'y' dimensions"

    assert hrrr_data.rio.crs is not None, "HRRR data must have a valid CRS for reprojection"

    # Retrieve the CRS from Herbie accessor
    # Herbie usually attaches the valid CRS here.
    crs = hrrr_data.herbie.crs

    # Create a transformer from Lat/Lon (4326) to the HRRR CRS
    # Note: HRRR is Lambert Conformal Conic
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Project the 2D Lat/Lon grids to get 2D Projected coords
    # We use the existing 2D lat/lon arrays in the dataset
    xx, yy = transformer.transform(
        hrrr_data.longitude.values, hrrr_data.latitude.values)

    # Extract 1D axes (HRRR is a rectilinear grid in its own projection)
    # We can take the first row for x and first column for y
    x_coords = xx[0, :]
    y_coords = yy[:, 0]

    # Assign these new coordinates to the dataset
    hrrr_data = hrrr_data.assign_coords(x=x_coords, y=y_coords)

    # Ensure the CRS is written to the xarray object for rioxarray
    hrrr_data = hrrr_data.rio.write_crs(crs)

    # Clip to task bounds in target CRS
    minx, miny, maxx, maxy = task_info.bounds

    # Calculate the shape (width/height) based on the target resolution
    width = int((maxx - minx) / target_resolution)
    height = int((maxy - miny) / target_resolution)

    # Define the Affine Transform (scales pixels to map coordinates)
    # This tells the code exactly where pixels sit in the target CRS
    target_transform = Affine.translation(
        minx, maxy) * Affine.scale(target_resolution, -target_resolution)

    # REPROJECT (Crop + Resample)
    hrrr_data = hrrr_data.rio.reproject(
        dst_crs=task_info.crs,
        shape=(height, width),
        transform=target_transform,
        resampling=Resampling.bilinear,
    )

    return hrrr_data


def download_hrrr(task_info: TaskInfo, delta_hour: int = 1) -> list[DataWithMetadata]:
    log.info(f"Downloading HRRR data for event_id: {task_info.event_id}")

    data_buffer: dict[str, list[np.ndarray]] = {
        'r2': [],  # Humidity
        'u10': [],  # Wind U
        'v10': [],  # Wind V
    }

    # Pre-calculate the list of timestamps to iterate over
    timestamps_iter = []
    current_time = task_info.t_start
    while current_time <= task_info.t_end:
        timestamps_iter.append(current_time)
        current_time += timedelta(hours=delta_hour)

    if not timestamps_iter:
        raise ValueError("No time range defined.")

    timestamps: list[datetime] = []

    pbar = tqdm(timestamps_iter)
    for current_time in pbar:
        log.info(f"Processing HRRR data for timestamp: {current_time}")
        pbar.set_description(
                f"Processing HRRR on {current_time.strftime('%Y-%m-%d %H:%M')}")

        try:
            H = Herbie(
                current_time,
                model='hrrr',
                product='sfc',
                fxx=0,
                save_dir=HERBIE_CACHE_DIR,
                verbose=log.level >= logging.INFO,
            )
            ds_rh = None
            ds_wind = None
            try:
                ds_rh = H.xarray(":RH:2 m", remove_grib=False)
                ds_rh = clip_hrrr_to_task(ds_rh, task_info)
                data_buffer['r2'].append(ds_rh.r2.values)
            except Exception as e_rh:
                log.error(f"⚠️ PARTIAL: Could not retrieve/process r2 for {current_time}. Error: {e_rh}")
            try:
                ds_wind = H.xarray(":(?:UGRD|VGRD):10 m", remove_grib=False)
                ds_wind = clip_hrrr_to_task(ds_wind, task_info)
                data_buffer['u10'].append(ds_wind.u10.values)
                data_buffer['v10'].append(ds_wind.v10.values)
            except Exception as e_wind:
                log.error(f"⚠️ PARTIAL: Could not retrieve/process u10/v10 for {current_time}. Error: {e_wind}")
            
            if (ds_rh is None) and (ds_wind is None):
                raise ValueError("Incomplete data retrieval for this timestamp.")
            else:
                timestamps.append(current_time)

        except Exception as e:
            # CATCH-ALL: If retrieval (primary AND fallback) OR processing fails
            log.error(f"❌ DATA GAP: Could not retrieve/process {current_time}. Error: {e}")
            continue


    if not timestamps:
        raise ValueError("No HRRR data downloaded.")

    payload = []
    for var_name, data_list in data_buffer.items():
        payload.append(DataWithMetadata(
            name=var_name,
            data=data_list,
            timestamps=timestamps,
            source="HRRR via Herbie",
            resolution=3000,
            unit="%" if var_name == 'r2' else "m/s",
        ))

    return payload


def _format_lat_lon_string(val: int, is_lon: bool) -> str:
    """Helper to format lat/lon integers for the filename (e.g. -120 -> w120, 35 -> n35)."""
    if is_lon:
        prefix = 'e' if val >= 0 else 'w'
        return f"{prefix}{abs(val):03d}"
    else:
        prefix = 'n' if val >= 0 else 's'
        return f"{prefix}{abs(val):02d}"


def _get_gba_tile_ids(bounds: tuple[float, float, float, float]) -> list[str]:
    """
    Calculates the Global Building Atlas 5x5 degree tile IDs needed to cover the bounds.
    Format example: w120_n35_w115_n30
    Structure: {WestLon}_{NorthLat}_{EastLon}_{SouthLat}
    bounds in CRS EPSG:4326
    """
    min_x, min_y, max_x, max_y = bounds

    # The grid aligns to 5 degree increments.
    # We find the 'floor' 5-degree lines for the min bounds.
    start_x = math.floor(min_x / 5.0) * 5
    start_y = math.floor(min_y / 5.0) * 5

    # We define the max iteration range (using ceil to ensure we cover the edge)
    # end_x = math.ceil(max_x / 5.0) * 5
    # end_y = math.ceil(max_y / 5.0) * 5

    tile_paths = []
    base_path = "projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS"

    # Iterate through the 5x5 grid cells that overlap the bounds
    # Python range is exclusive at the end, so we add 5 to end_x/end_y if we rely on range
    # But simpler is to loop with while/integers

    curr_x = start_x
    while curr_x < max_x:  # Iterate West to East
        curr_y = start_y
        while curr_y < max_y:  # Iterate South to North

            # Tile definitions for this 5x5 cell
            # The naming convention puts the West and North coordinate first usually,
            # let's replicate the example: w120_n35_w115_n30
            # That example tile covers: Lon [-120, -115], Lat [30, 35]

            # The integers used in the filename are:
            # 1. West Edge (curr_x)
            # 2. North Edge (curr_y + 5)
            # 3. East Edge (curr_x + 5)
            # 4. South Edge (curr_y)

            tile_w = int(curr_x)
            tile_s = int(curr_y)
            tile_e = int(curr_x + 5)
            tile_n = int(curr_y + 5)

            part1 = _format_lat_lon_string(tile_w, is_lon=True)
            part2 = _format_lat_lon_string(tile_n, is_lon=False)
            part3 = _format_lat_lon_string(tile_e, is_lon=True)
            part4 = _format_lat_lon_string(tile_s, is_lon=False)

            tile_id = f"{part1}_{part2}_{part3}_{part4}"
            tile_paths.append(f"{base_path}/{tile_id}")

            curr_y += 5
        curr_x += 5

    return tile_paths


def download_building_height(task_info: TaskInfo):
    dataset_name = "building_height"
    log.info(
        f"Downloading {dataset_name} data for event_id: {task_info.event_id}")
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

    # Load and Merge Collections
    collection = None

    for path in tile_paths:
        try:
            col = ee.FeatureCollection(path)
            # Verify the collection exists by grabbing metadata or checking size roughly
            # (Optional: this might fail if the tile doesn't exist in GEE yet)
            if collection is None:
                collection = col
            else:
                collection = collection.merge(col)
        except Exception as e:
            log.warning(
                f"Could not load GBA tile: {path}. It might not exist or there is an access issue. Error: {e}")

    if collection is None:
        raise ValueError(
            "Could not load any building atlas tiles for the requested region.")

    # Filter and Clip
    # Filter bounds first to reduce processing load before intersection
    clipped = collection.filterBounds(roi)

    clipped = clipped.filter(
        ee.Filter.And(
            ee.Filter.neq('height', None),
            ee.Filter.gt('height', 3.971)
        )
    )

    count = clipped.size().getInfo()
    log.info(f"Buildings in region: {count}")

    height_raster = clipped.reduceToImage(
        properties=["height"],
        reducer=ee.Reducer.max()
    ).unmask(0).rename(dataset_name)

    # Download
    data_array = _download_processed_image(
        height_raster, task_info, band_name=dataset_name)

    return DataWithMetadata(
        name=dataset_name,
        data=[data_array],
        timestamps=[task_info.t_start],
        source=f"Global Building Atlas (Tiles: {len(tile_paths)})",
        resolution=3,
        unit="m",
    )


def download_eca(task_info: TaskInfo) -> DataWithMetadata:
    log.info(f"Downloading ECA data for event_id: {task_info.event_id}")
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


def download_tc(task_info: TaskInfo):
    log.info(
        f"Downloading LAI data for event_id: {task_info.event_id}")

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

    # Build Image Objects
    # --------------------------------------
    # We must construct ee.Image objects and rename the band to 'LAI'
    # *before* passing to download_gee_task.
    # The generic function expects a collection where .select(band) works immediately.

    def prepare_lai_image(filename):
        # Load image, select the first band (index 0), and normalize name to 'lai'
        return ee.Image(ASSET_ROOT + filename).select([0]).rename('lai')

    # Create a Python list of ee.Image objects
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


def main() -> None:

    global GEE_PROJECT_ID
    global HERBIE_CACHE_DIR

    parser = argparse.ArgumentParser(description="Fire Data Loader")
    parser.add_argument("event_id", type=str, help="Event ID to process")
    parser.add_argument("-r", "--resolution", type=int,
                        default=30, help="Resolution for task info")
    parser.add_argument("-b", "--buffer", type=int,
                        default=20, help="Buffer for task info")
    parser.add_argument("-c", "--crs", type=str,
                        default="EPSG:5070", help="CRS for task info")
    parser.add_argument("-o", "--output_dir", type=str,
                        default="output", help="Output directory for saved data")
    parser.add_argument("-p", "--gee_project_id", type=str,
                        default=GEE_PROJECT_ID, help="Google Earth Engine project ID")
    parser.add_argument("-t", "--interpolation", type=int,
                        default=0, help="Interpolate burn perimeters with given multiplier")
    parser.add_argument("--herbie_cache_dir", type=str,
                        default=HERBIE_CACHE_DIR, help="Directory for Herbie cache")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")

    args = parser.parse_args()

    GEE_PROJECT_ID = args.gee_project_id
    HERBIE_CACHE_DIR = args.herbie_cache_dir

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)  # Simplified format

    event_id = args.event_id
    fire_info = get_fire_info(event_id)

    log.info(f"Retrieved fire info for event_id: {event_id}")
    log.debug(f"Generated fire info: {fire_info}")

    task_info = get_task_info(
        fire_info, resolution=args.resolution, buffer=args.buffer, crs=args.crs)

    log.debug(f"Generated task info: {task_info}")
    save_numpy(task_info, DataWithMetadata(
        name="task_info", data=[task_info]), args.output_dir)

    feds25mtbs = process_feds25mtbs(task_info)
    log.debug(f"Processed FEDS25MTBS data: {feds25mtbs}")
    
    if args.interpolation > 0:
        log.info(
            f"Interpolating burn perimeters with multiplier: {args.interpolation}")
        feds25mtbs = interpolate_burn_perimeters(
            feds25mtbs, multiplier=args.interpolation)
        log.debug(f"Interpolated FEDS25MTBS data: {feds25mtbs}")
    
    save_numpy(task_info, feds25mtbs, args.output_dir)

    elevation = download_usgs(task_info)
    log.debug(f"Downloaded elevation data: {elevation}")
    save_numpy(task_info, elevation, args.output_dir)

    landfire = download_landfire(task_info)
    for lf_data in landfire:
        log.debug(f"Downloaded LANDFIRE data: {lf_data}")
        save_numpy(task_info, lf_data, args.output_dir)

    hrrr = download_hrrr(task_info)

    for hrrr_data in hrrr:
        log.debug(f"Downloaded HRRR data: {hrrr_data}")
        save_numpy(task_info, hrrr_data, args.output_dir)

    building_height = download_building_height(task_info)
    log.debug(f"Processed building height data: {building_height}")
    save_numpy(task_info, building_height, args.output_dir)

    landcover = download_eca(task_info)
    log.debug(f"Processed landcover data: {landcover}")
    save_numpy(task_info, landcover, args.output_dir)

    lai = download_tc(task_info)
    log.debug(f"Processed LAI data: {lai}")
    save_numpy(task_info, lai, args.output_dir)


def plot(data, *args, **kwargs):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap='viridis', interpolation='nearest', *args, **kwargs)
    plt.colorbar(label='Value')
    plt.show()


if __name__ == "__main__":

    main()
