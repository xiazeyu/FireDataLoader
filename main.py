from dataclasses import asdict, dataclass
from datetime import datetime
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box, MultiPolygon
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
import rasterio
import requests

log = logging.getLogger(__name__)
GEE_PROJECT_ID = "annular-haven-474021-v1"


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


def get_fire_info(event_id: str, firelist_path='datasets/FEDS25MTBS/fireslist2012-2023.csv') -> FireInfo:
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
    

def process_feds25mtbs(task_info: TaskInfo, base_dir='datasets/FEDS25MTBS') -> DataWithMetadata:
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
    
    log.info(f"Downloading {dataset_name} data for event_id: {task_info.event_id} from Google Earth Engine")

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
    return data


def download_landfire(task_info: TaskInfo) -> DataWithMetadata:
    log.info(f"Downloading LANDFIRE data for event_id: {task_info.event_id}")

    # TODO


def download_hrrr(task_info: TaskInfo):
    log.info(f"Downloading HRRR data for event_id: {task_info.event_id}")

    # TDOO


def download_building_height(task_info: TaskInfo):
    log.info(
        f"Downloading building height data for event_id: {task_info.event_id}")

    # TODO


def main() -> None:
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
                        default="annular-haven-474021-v1", help="Google Earth Engine project ID")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")

    args = parser.parse_args()
    
    global GEE_PROJECT_ID
    GEE_PROJECT_ID = args.gee_project_id

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)  # Simplified format

    event_id = args.event_id
    fire_info = get_fire_info(event_id)

    log.info(f"Retrieved fire info for event_id: {event_id}")
    log.debug(f"Generated fire info: {fire_info}")

    task_info = get_task_info(
        fire_info, resolution=args.resolution, buffer=args.buffer, crs=args.crs)

    log.debug(f"Generated task info: {task_info}")
    save_numpy(task_info, DataWithMetadata(name="task_info", data=[task_info]), args.output_dir)

    feds25mtbs = process_feds25mtbs(task_info)
    log.debug(f"Processed FEDS25MTBS data: {feds25mtbs}")
    save_numpy(task_info, feds25mtbs, args.output_dir)

    elevation = download_usgs(task_info)
    log.debug(f"Downloaded elevation data: {elevation}")
    save_numpy(task_info, elevation, args.output_dir)
    
    download_landfire(task_info)
    download_hrrr(task_info)
    download_building_height(task_info)


if __name__ == "__main__":
    main()
