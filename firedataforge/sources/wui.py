"""Global WUI: byte-range streaming of Equi7 GeoTIFF tiles + rasterization."""

import logging
import math
import os
import shutil
import struct
import urllib.error
import urllib.request
import zlib
from typing import Optional

import numpy as np
import rioxarray  # noqa: F401  (.rio accessor)
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from tqdm import tqdm

from firedataforge.constants import CACHE_DIR, DATASETS_DIR, GLOBALWUI_CACHE_NAME
from firedataforge.progress import download_to_file
from firedataforge.schemas import DataLayer, ProcessingTask

log = logging.getLogger(__name__)


GLOBAL_WUI_CLASSES = {
    1: "Forest/Shrub/Wetland Intermix WUI",
    2: "Forest/Shrub/Wetland Interface WUI",
    3: "Grassland Intermix WUI",
    4: "Grassland Interface WUI",
    5: "Non-WUI: Forest/Shrub/Wetland",
    6: "Non-WUI: Grassland",
    7: "Non-WUI: Urban",
    8: "Non-WUI: Other",
}

# User-managed full archive (optional pre-staged download) vs the software cache
# that on-the-fly streamed tiles land in. Readers check the archive first.
DEFAULT_GLOBALWUI_DIR = os.path.join(DATASETS_DIR, "GlobalWUI")
GLOBALWUI_CACHE_DIR = os.path.join(CACHE_DIR, GLOBALWUI_CACHE_NAME)

# The Global WUI data is distributed only as per-continent zips (no per-tile
# URLs). The SILVIS geoserver serves them over plain HTTP with byte-range
# support, so individual ~32 KB tiles can be streamed out of the zip via range
# requests (read the central directory, then just the one entry) instead of
# downloading the full ~3.8 GB archive. The EQUI7 tile params below are the
# North America grid, so only the NA archive is used.
GLOBALWUI_ZIP_URL = "https://geoserver.silvis.forest.wisc.edu/geodata/globalwui/NA.zip"
GLOBALWUI_ZIP_PREFIX = "NA"  # internal path prefix inside the zip (e.g. NA/X..._Y.../WUI.tif)

# Parsed central directories, keyed by zip URL: {name: (method, csize, local_offset)}.
_globalwui_zip_index: dict[str, dict[str, tuple[int, int, int]]] = {}


def _http_range(url: str, start: int, end: Optional[int] = None) -> bytes:
    """Fetch a byte range from ``url`` via an HTTP Range request.

    ``start`` may be negative (suffix range, e.g. -65557 for the last 65557
    bytes); ``end`` is inclusive when given.
    """
    if start < 0:
        rng = f"bytes={start}"
    else:
        rng = f"bytes={start}-{end if end is not None else ''}"
    req = urllib.request.Request(url, headers={"Range": rng})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _zip_central_directory(url: str) -> dict[str, tuple[int, int, int]]:
    """Read a remote zip's central directory via range requests.

    Returns a mapping of entry name -> (compression_method, compressed_size,
    local_header_offset). Cached per URL.
    """
    if url in _globalwui_zip_index:
        return _globalwui_zip_index[url]

    # The End Of Central Directory record sits within the last 64 KB (+22 byte
    # record + up to 65535 byte comment). This dataset's zips are < 4 GB, so the
    # classic (non-ZIP64) format applies.
    tail = _http_range(url, -(65557))
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError(f"Zip End Of Central Directory not found for {url}")
    (_, _, _, _, _, cd_size, cd_off, _) = struct.unpack(
        "<IHHHHIIH", tail[eocd:eocd + 22]
    )

    cd = _http_range(url, cd_off, cd_off + cd_size - 1)
    entries: dict[str, tuple[int, int, int]] = {}
    p = 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, _, method, _, _, _, csize, _,
         nlen, elen, clen, _, _, _, loff) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", cd[p:p + 46]
        )
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        entries[name] = (method, csize, loff)
        p += 46 + nlen + elen + clen

    _globalwui_zip_index[url] = entries
    return entries


def _stream_zip_entry(
    url: str, method: int, csize: int, local_offset: int
) -> bytes:
    """Download and decompress a single zip entry via range requests."""
    # The local file header has its own (possibly different) name/extra lengths,
    # so read it to find where the compressed data actually begins.
    header = _http_range(url, local_offset, local_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local file header at offset {local_offset}")
    nlen, elen = struct.unpack("<HH", header[26:30])
    data_off = local_offset + 30 + nlen + elen
    comp = _http_range(url, data_off, data_off + csize - 1)
    if method == 0:          # stored
        return comp
    if method == 8:          # deflate
        return zlib.decompress(comp, -15)
    raise ValueError(f"Unsupported zip compression method {method}")


def _stream_globalwui_tile(tile_id: str, dest_path: str) -> bool:
    """Stream a single Global WUI tile from the remote archive to ``dest_path``.

    Reads only the relevant ~32 KB entry from the continent zip (no full
    download). Returns False if the tile is not present in the archive (e.g.
    outside North America coverage / over ocean).
    """
    entries = _zip_central_directory(GLOBALWUI_ZIP_URL)
    entry_name = f"{GLOBALWUI_ZIP_PREFIX}/{tile_id}/WUI.tif"
    if entry_name not in entries:
        return False

    method, csize, loff = entries[entry_name]
    raw = _stream_zip_entry(GLOBALWUI_ZIP_URL, method, csize, loff)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as f:
        f.write(raw)
    os.replace(tmp_path, dest_path)
    return True


def download_globalwui_archive(dest_dir: str = DEFAULT_GLOBALWUI_DIR) -> str:
    """Download and unpack the *entire* Global WUI North America archive.

    The opposite of the per-tile streaming path: pulls the full ~3.8 GB
    ``NA.zip`` and extracts every ``WUI.tif`` into ``dest_dir`` as
    ``<tile_id>/WUI.tif`` (matching the layout :func:`download_globalwui` reads),
    so no tiles ever need streaming. Use it to trade disk for bandwidth; the
    default on-demand per-tile streaming into ``cache/`` is otherwise plenty.

    Returns ``dest_dir``.
    """
    import zipfile

    os.makedirs(dest_dir, exist_ok=True)
    tmp_zip = os.path.join(dest_dir, "_NA.zip")
    log.info(f"Downloading the full Global WUI archive from {GLOBALWUI_ZIP_URL}")
    prefix = f"{GLOBALWUI_ZIP_PREFIX}/"
    try:
        # SILVIS serves NA.zip with byte-range support, so two parallel streams
        # bypass any per-connection throttle (falls back to one stream if not).
        download_to_file(
            GLOBALWUI_ZIP_URL, tmp_zip, desc="Global WUI archive", connections=2)
        with zipfile.ZipFile(tmp_zip) as zf:
            # Only the per-tile WUI.tif members are kept (the layout download_globalwui
            # reads), so the bar counts tiles extracted rather than every zip member.
            members = [n for n in zf.namelist() if n.endswith("/WUI.tif")]
            for name in tqdm(members, desc="Extracting Global WUI", unit=" tile",
                             disable=None, leave=False):
                # Strip the leading "NA/" so the local layout is <tile>/WUI.tif.
                rel = name[len(prefix):] if name.startswith(prefix) else name
                out_path = os.path.join(dest_dir, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
    log.info(f"Full Global WUI archive unpacked into {dest_dir}")
    return dest_dir


def _get_equi7_tile_params() -> dict:
    """Get EQUI7 grid parameters for North America tile system.
    
    The Global WUI data uses the EQUI7 Azimuthal Equidistant projection
    with 100km x 100km tiles.
    
    Returns:
        Dictionary with projection and tile parameters.
    """
    return {
        'crs': 'PROJCS["Azimuthal_Equidistant",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Azimuthal_Equidistant"],PARAMETER["false_easting",8264722.17686],PARAMETER["false_northing",4867518.35323],PARAMETER["longitude_of_center",-97.5],PARAMETER["latitude_of_center",52.0],UNIT["Meter",1.0]]',
        'tile_size': 100000,  # 100km tiles
        'origin_x': 0,
        'origin_y': 9900000,
    }


def _get_equi7_tiles_for_bounds(
    bounds: tuple[float, float, float, float],
    source_crs: str,
) -> list[str]:
    """Calculate EQUI7 tile IDs needed to cover the given bounds.
    
    Args:
        bounds: Bounding box (minx, miny, maxx, maxy) in source_crs.
        source_crs: CRS of the input bounds.
    
    Returns:
        List of tile IDs in format 'X0065_Y0040'.
    """
    equi7_params = _get_equi7_tile_params()
    
    # Transform bounds to EQUI7 projection
    transformer = Transformer.from_crs(source_crs, equi7_params['crs'], always_xy=True)
    
    # Transform all corners to handle projection distortion
    minx, miny, maxx, maxy = bounds
    corners = [
        (minx, miny), (minx, maxy),
        (maxx, miny), (maxx, maxy),
    ]
    
    transformed_x = []
    transformed_y = []
    for x, y in corners:
        tx, ty = transformer.transform(x, y)
        transformed_x.append(tx)
        transformed_y.append(ty)
    
    # Get bounding box in EQUI7 coordinates
    equi7_minx = min(transformed_x)
    equi7_miny = min(transformed_y)
    equi7_maxx = max(transformed_x)
    equi7_maxy = max(transformed_y)
    
    # Calculate tile indices
    tile_size = equi7_params['tile_size']
    origin_y = equi7_params['origin_y']
    
    # Tile index calculation: tile_x = floor(x / tile_size), tile_y = floor((origin_y - y) / tile_size)
    # Note: Y increases downward in the tile naming scheme
    min_tile_x = int(math.floor(equi7_minx / tile_size))
    max_tile_x = int(math.floor(equi7_maxx / tile_size))
    min_tile_y = int(math.floor((origin_y - equi7_maxy) / tile_size))
    max_tile_y = int(math.floor((origin_y - equi7_miny) / tile_size))
    
    tiles = []
    for tx in range(min_tile_x, max_tile_x + 1):
        for ty in range(min_tile_y, max_tile_y + 1):
            tile_id = f"X{tx:04d}_Y{ty:04d}"
            tiles.append(tile_id)
    
    return tiles


def download_globalwui(
    task_info: ProcessingTask,
    base_dir: str = DEFAULT_GLOBALWUI_DIR,
    cache_dir: str = CACHE_DIR,
) -> DataLayer:
    """Download Global Wildland-Urban Interface (WUI) data.

    The Global WUI dataset maps the interface between human settlements and
    wildland vegetation at 10m resolution. Data is organized in EQUI7 tiles.

    Only the EQUI7 tiles intersecting the event are needed. Any tile missing
    from ``base_dir`` is streamed directly out of the remote continent archive
    via HTTP range requests (~32 KB per tile) and cached locally, so the full
    ~3.8 GB archive never has to be downloaded.

    Reference:
        Schug, F. et al. (2023). The global wildland–urban interface.
        Nature. https://doi.org/10.1038/s41586-023-06320-0

    Args:
        task_info: Task configuration with bounds and resolution.
        base_dir: Base directory for the user-managed Global WUI archive.
        cache_dir: Cache root for on-the-fly downloads; streamed tiles are
            cached under its fixed ``GlobalWUI`` subfolder.

    Returns:
        DataLayer containing WUI class values (uint8, 1-8).

    Raises:
        FileNotFoundError: If required tiles are not found.
    """
    log.info(f"Processing Global WUI data for event_id: {task_info.event_id}")

    # Streamed tiles cache under the fixed ``GlobalWUI`` subfolder of the root.
    wui_cache_dir = os.path.join(cache_dir, GLOBALWUI_CACHE_NAME)

    # Get tiles that intersect with task bounds
    tiles_needed = _get_equi7_tiles_for_bounds(task_info.bounds, task_info.crs)
    log.info(f"Required EQUI7 tiles: {tiles_needed}")
    
    # Load and merge tiles using rioxarray
    tile_datasets = []
    missing_tiles = []
    
    for tile_id in tiles_needed:
        # Prefer the user archive (datasets/), then the software cache (cache/).
        archive_path = os.path.join(base_dir, tile_id, 'WUI.tif')
        cache_path = os.path.join(wui_cache_dir, tile_id, 'WUI.tif')
        if os.path.exists(archive_path):
            tile_path = archive_path
        elif os.path.exists(cache_path):
            tile_path = cache_path
        else:
            # Stream the tile out of the remote archive into the cache.
            tile_path = cache_path
            try:
                log.info(f"Streaming Global WUI tile {tile_id} from remote archive...")
                if not _stream_globalwui_tile(tile_id, tile_path):
                    log.warning(
                        f"Tile {tile_id} not in remote WUI archive "
                        f"(may be outside North America / WUI coverage)"
                    )
                    missing_tiles.append(tile_id)
                    continue
            except Exception as e:
                log.warning(f"Failed to stream WUI tile {tile_id}: {e}")
                missing_tiles.append(tile_id)
                continue

        try:
            ds = rioxarray.open_rasterio(tile_path)
            tile_datasets.append(ds)
            log.debug(f"Loaded tile: {tile_id}")
        except Exception as e:
            log.warning(f"Error loading tile {tile_id}: {e}")
            missing_tiles.append(tile_id)

    if not tile_datasets:
        raise FileNotFoundError(
            f"No Global WUI tiles found for region. "
            f"Expected tiles: {tiles_needed}. "
            f"Could not load them locally or stream them from {GLOBALWUI_ZIP_URL}."
        )
    
    if missing_tiles:
        log.warning(f"Missing tiles (may be outside WUI coverage): {missing_tiles}")
    
    # Merge tiles if multiple
    if len(tile_datasets) == 1:
        merged = tile_datasets[0]
    else:
        from rioxarray.merge import merge_arrays
        merged = merge_arrays(tile_datasets)
    
    # Reproject to task CRS and clip to bounds
    t_minx, t_miny, t_maxx, t_maxy = task_info.bounds
    
    # Create the target transform
    target_transform = from_origin(
        t_minx, t_maxy,
        task_info.resolution, task_info.resolution
    )
    
    # Reproject and resample
    reprojected = merged.rio.reproject(
        dst_crs=task_info.crs,
        shape=task_info.shape,
        transform=target_transform,
        resampling=Resampling.nearest,  # Use nearest neighbor for categorical data
    )
    
    # Extract data array
    data_array = reprojected.values
    
    # Handle band dimension if present
    if data_array.ndim == 3:
        data_array = data_array[0]  # Take first band
    
    assert data_array.shape == task_info.shape, (
        f"Shape mismatch: got {data_array.shape}, expected {task_info.shape}"
    )
    
    # Convert to uint8
    data_array = data_array.astype(np.uint8)
    
    log.info(f"✓ Processed Global WUI data: {data_array.shape}, "
             f"unique classes: {np.unique(data_array).tolist()}")
    
    return DataLayer(
        name="wui",
        data=[data_array],
        timestamps=[task_info.t_start],
        source="Global WUI; Schug et al. 2023; https://doi.org/10.1038/s41586-023-06320-0",
        native_resolution=10,
        unit="class",
        categories=GLOBAL_WUI_CLASSES,
        note={
            'description': 'Wildland-Urban Interface classification',
            'temporal_coverage': 'ca. 2020',
            'missing_tiles': missing_tiles if missing_tiles else None,
        }
    )
