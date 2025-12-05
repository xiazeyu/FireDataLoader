from affine import Affine
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from herbie import Herbie
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import box, MultiPolygon
from tqdm import tqdm
from typing import Any, Literal, Optional
import argparse
import ee
import geopandas as gpd
import io
import logging
import math
import numpy as np
import os
import pandas as pd
import requests
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


def download_gee_task(task_info: TaskInfo, dataset_name: str, imagecollection: str, band: str, resample: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear') -> DataWithMetadata:
    # only works for image collections in GEE, not for feature collections

    log.info(
        f"Downloading {dataset_name} data for event_id: {task_info.event_id} from Google Earth Engine")

    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except Exception as e:
        log.info("Earth Engine not initialized. Attempting to authenticate...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)

    roi = ee.Geometry.Rectangle(task_info.bounds, task_info.crs, False)

    collection = ee.ImageCollection(imagecollection).filterBounds(roi)

    if collection.size().getInfo() == 0:
        error_msg = (f"The requested ROI is outside the coverage of {imagecollection}. "
                     f"Images found: 0. Bounds: {task_info.bounds}")
        raise ValueError(error_msg)

    native_proj = collection.first().select(band).projection()
    image = collection.mosaic().select(band)
    image = image.setDefaultProjection(native_proj)

    if resample != 'nearest':
        image = image.resample(resample)

    try:
        url = image.getDownloadURL({
            'scale': task_info.resolution,
            'crs': task_info.crs,
            'region': roi,
            'format': 'NPY'
        })
    except Exception as e:
        log.error(f"Error generating download URL: {e}")
        raise e

    log.info(f"Download URL: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        log.error(f"Error downloading data: HTTP {response.status_code}")
        log.error(f"Response content: {response.content}")
        response.raise_for_status()

    data = np.load(io.BytesIO(response.content), allow_pickle=True)
    log.info(f"Downloaded {dataset_name} data shape: {data.shape}")

    data = data[band]

    return DataWithMetadata(
        name=dataset_name,
        data=[data],
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
    data.data[0] = np.around(data.data[0]).astype(np.int16)
    data.resolution = 1
    data.unit = "m"
    return data


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
    data.data[0] = np.around(data.data[0]).astype(np.int16)
    data.resolution = 30
    data.unit = "100kg/m^3"
    payload.append(data)

    data = download_gee_task(
        task_info,
        dataset_name="cc",
        band="CC",
        imagecollection="projects/sat-io/open-datasets/landfire/FUEL/CC",
        resample='bilinear',
    )
    data.data[0] = np.around(data.data[0]).astype(np.int16)
    data.resolution = 30
    data.unit = "%"
    payload.append(data)

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

        try:
            pbar.set_description(
                f"Processing HRRR on {current_time.strftime('%Y-%m-%d %H:%M')}")
            H = Herbie(
                current_time,
                model='hrrr',
                product='sfc',
                fxx=0,
                save_dir=HERBIE_CACHE_DIR,
                verbose=log.level >= logging.INFO,
            )

            try:
                ds_rh = H.xarray(":RH:2 m", remove_grib=False)
                ds_wind = H.xarray(":(?:UGRD|VGRD):10 m", remove_grib=False)
            except ValueError as e:
                if "No index file" in str(e):
                    tqdm.write(
                        f"⚠️ Index missing for {current_time}. Attempting full download...")
                    H.download()
                    ds_rh = H.xarray(":RH:2 m", remove_grib=False)
                    ds_wind = H.xarray(
                        ":(?:UGRD|VGRD):10 m", remove_grib=False)
                else:
                    raise e

            ds_rh = clip_hrrr_to_task(ds_rh, task_info)
            data_buffer['r2'].append(ds_rh.r2.values)

            ds_wind = clip_hrrr_to_task(ds_wind, task_info)
            data_buffer['u10'].append(ds_wind.u10.values)
            data_buffer['v10'].append(ds_wind.v10.values)
            timestamps.append(current_time)
        except Exception as e:
            # CATCH-ALL: If data is missing from ALL sources
            log.error(
                f"❌ DATA GAP: Could not retrieve {current_time} on HRRR via Herbie. Skipping. Error: {e}")
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


def download_building_height(task_info: TaskInfo):
    log.info(
        f"Downloading building height data for event_id: {task_info.event_id}")

    # TODO

def download_tc(task_info: TaskInfo):
    log.info(
        f"Downloading LAI data for event_id: {task_info.event_id}")

    # TODO


def main() -> None:
    
    # TODO LANDCOVER: resample using nearest neighbor

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
    parser.add_argument("--herbie_cache_dir", type=str,
                        default=HERBIE_CACHE_DIR, help="Directory for Herbie cache")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")

    args = parser.parse_args()

    GEE_PROJECT_ID = args.gee_project_id
    HERBIE_CACHE_DIR = args.herbie_cache_dir

    log_level = logging.INFO if args.verbose else logging.WARNING
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

    download_building_height(task_info)
    download_tc(task_info)


if __name__ == "__main__":
    main()
