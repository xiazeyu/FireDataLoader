# FireDataLoader

A tool for downloading and processing wildfire-related geospatial data from multiple sources for machine learning and analysis.

## Data Sources

| Dataset | Description | Resolution | Feature Name(s) |
|---------|-------------|------------|-----------------|
| FEDS25MTBS (MTBS-constrained FEDS 2.5) | Fire perimeters, firelines, and pre-2024 active-fire pixels | 375m | `burn_perimeter`, `fireline`, `frp_daytime`/`frp_nighttime` (<2024) |
| NASA FIRMS (VIIRS) | Active-fire FRP for events from 2024 onwards | 375m | `frp_daytime`, `frp_nighttime` (≥2024) |
| *Derived from `fireline` + FRP* | Per-segment max FRP painted onto fireline pixels | — | `fireline_frp` |
| USGS 3DEP | Elevation and colored shaded-relief visualization | 1m | `elevation`, `terrain_rgb` |
| LANDFIRE | Canopy Bulk Density (CBD), Canopy Cover (CC) | 30m | `canopy_bulk_density`, `canopy_cover` |
| HRRR | Weather: humidity (r2), wind (u10, v10) | 3km | `r2`, `u10`, `v10` |
| Global Building Atlas | Building heights | 3m | `building_height` |
| ESA WorldCover v200 | Land cover classification | 10m | `landcover` |
| LAI | Leaf Area Index (single 2020-07-02 snapshot) | 10m | `lai` |
| Sentinel-2 L2A Cloudless Mosaic | Satellite imagery (RGB) | 10m | `sentinel2_rgb` |
| Global WUI (Schug et al. 2023) | Wildland-Urban Interface classification | 10m | `wui` |

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Prerequisites

### Google Earth Engine Setup

Before running, configure your GEE project:

```bash
# Authenticate (one-time, opens browser)
earthengine authenticate

# Set your project ID
earthengine set_project YOUR-PROJECT-ID
```

To get a project ID:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project with [Earth Engine enabled](https://developers.google.com/earth-engine/guides/access)
3. Copy the project ID

## Quick Start

```bash
# Process a single fire event
python main.py CA3859812261820171009

# With verbose output
python main.py CA3859812261820171009 -v

# Batch process multiple fires
python main.py --batch events.txt --workers 4
```

## Usage

### Single Event

```bash
python main.py <event_id> [options]
```

### Batch Processing

```bash
# From a file (one event ID per line)
python main.py --batch events.txt [options]

# From comma-separated list  
python main.py --batch CA123,CA456,CA789 [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch` | Batch mode: file path or comma-separated event IDs | - |
| `-w, --workers` | Parallel workers for batch processing | 1 |
| `-r, --resolution` | Spatial resolution (meters) | 30 |
| `-b, --buffer` | Buffer around fire bounds (meters) | 100 |
| `-c, --crs` | Target coordinate reference system | EPSG:5070 |
| `-o, --output_dir` | Output directory | output |
| `-t, --interpolation` | Intermediate frames between timesteps | 0 |
| `--herbie_cache_dir` | HRRR cache directory | datasets/herbie |
| `--only` | Only process specific feature(s), comma-separated | all |
| `-v, --verbose` | Enable verbose logging | False |

#### Available Features for `--only`

| Feature | Description |
|---------|-------------|
| `burn_perimeter` | Fire perimeter time series from FEDS25MTBS |
| `fireline` | Active fireline derived from consecutive perimeter differences |
| `fireline_frp` | Per-pixel maximum FRP along the fireline |
| `frp_daytime` | Daytime Fire Radiative Power (FIRMS ≥2024 / FEDS25MTBS firepix <2024) |
| `frp_nighttime` | Nighttime Fire Radiative Power (FIRMS ≥2024 / FEDS25MTBS firepix <2024) |
| `elevation` | USGS 3DEP elevation |
| `landfire` | LANDFIRE CBD and Canopy Cover |
| `building_height` | Global Building Atlas heights |
| `landcover` | ESA WorldCover classification |
| `lai` | Leaf Area Index |
| `sentinel2_rgb` | RGB sentinel2_rgb imagery Sentinel-2 |
| `terrain_rgb` | Colored shaded-relief terrain (Google-Maps style, RGB) |
| `wui` | Wildland-Urban Interface classification |
| `hrrr` | Weather data (humidity, wind) |

### Examples

```bash
# High resolution processing
python main.py CA3859812261820171009 -r 10 -v

# With temporal interpolation (3 intermediate frames)
python main.py CA3859812261820171009 -t 3

# Custom output directory
python main.py CA3859812261820171009 -o ./my_output

# Batch process from file with 4 workers
python main.py --batch events.txt -w 4 -o results/

# Batch process specific events
python main.py --batch CA123,CA456,CA789 --workers 3

# Process only a single feature (for quick debugging)
python main.py CA3859812261820171009 --only frp_daytime

# Process multiple specific features
python main.py CA3859812261820171009 --only frp_daytime,frp_nighttime,elevation

# Regenerate only weather data
python main.py CA3859812261820171009 --only hrrr
```

The batch summary is saved to `output/batch_summary.json`.

## Output

Data is saved as `.npy` files in `output/<event_id>/`:

```
output/CA3859812261820171009/
├── task_info.npy         # Processing configuration
├── coordinates.npy       # Pixel-center x/y coordinates + CRS for the grid
├── burn_perimeter.npy   # Fire perimeter time series
├── frp_daytime.npy           # Daytime Fire Radiative Power (MW)
├── frp_nighttime.npy         # Nighttime Fire Radiative Power (MW)
├── elevation.npy         # Terrain elevation
├── canopy_bulk_density.npy               # Canopy Bulk Density
├── canopy_cover.npy                # Canopy Cover
├── r2.npy                # Relative humidity
├── u10.npy               # Wind U component
├── v10.npy               # Wind V component
├── building_height.npy   # Building heights
├── landcover.npy         # Land cover classes
├── lai.npy               # Leaf Area Index
├── sentinel2_rgb.npy         # RGB sentinel2_rgb imagery
├── terrain_rgb.npy       # Colored shaded-relief terrain RGB (H, W, 3)
└── wui.npy               # Wildland-Urban Interface classification
```

### Grid Coordinates (`coordinates.npy`)

Every event directory now also contains `coordinates.npy`, which stores the
pixel-center coordinates of the common output grid together with the CRS.
All other raster layers (`elevation.npy`, `frp_*.npy`, `wui.npy`, ...) are
sampled on this exact grid, so this file is the single source of truth for
georeferencing the arrays — useful for wrapping outputs into `xarray`
DataArrays or re-projecting them when preparing publication figures.

```python
from main import load_numpy

coords = load_numpy('output/CA3859812261820171009/coordinates.npy')
x, y = coords.data            # 1-D arrays, shape (width,) and (height,)
crs = coords.note['crs']      # e.g. 'EPSG:5070'
crs_wkt = coords.note['crs_wkt']    # full WKT2 (works without an EPSG db)
crs_proj4 = coords.note['crs_proj4']  # legacy PROJ string
bounds = coords.note['bounds']        # (minx, miny, maxx, maxy)
height, width = coords.note['shape']
a, b, c, d, e, f = coords.note['transform']  # affine: (col, row) -> (x, y)
```

For standard EPSG codes, `note['crs']` alone is enough for pyproj /
rasterio / cartopy. The extra `crs_wkt` and `crs_proj4` fields are
included so the file is fully self-describing — useful for archival,
custom CRSes (e.g. EQUI7, HRRR Lambert), or environments without a PROJ
database.

`y` is ordered top-to-bottom (north → south) to match the row order of the
saved rasters.

### Visualizing Data

```bash
# Plot all layers overview (saves to file)
python plot_data.py CA3859812261820171009

# Plot and display interactively
python plot_data.py CA3859812261820171009 --show

# Plot time series frames (e.g., burn perimeters)
python plot_data.py CA3859812261820171009 -t burn_perimeter

# Batch plot multiple events
python plot_data.py --batch events.txt

# Batch plot from comma-separated list
python plot_data.py --batch CA123,CA456,CA789
```

### Loading Data

```python
from main import load_numpy

# Load a data file
data = load_numpy('output/CA3859812261820171009/elevation.npy')
print(data.name)        # 'elevation'
print(data.data[0].shape)  # (height, width)
print(data.unit)        # 'm'
```

### Using Schemas Externally

```python
from schemas import FireInfo, TaskInfo, DataWithMetadata
from main import get_fire_info, get_task_info

# Get fire information
fire = get_fire_info('CA3859812261820171009')
print(f"{fire.name}: {fire.acres_burned} acres")

# Create task configuration
task = get_task_info(fire, resolution=30)
print(f"Grid: {task.shape}, CRS: {task.crs}")
```

## Available Fire Events

Fire events are listed in `datasets/FEDS25MTBS/fireslist2012-2023.csv`. Event IDs follow the pattern: `{STATE}{LAT}{LON}{DATE}` (e.g., `CA3859812261820171009`).

## FEDS25MTBS Dataset

The Fire Event Data Suite (FEDS) provides half-daily fire perimeter time series derived from VIIRS sentinel2_rgb observations using an object-based tracking system.

### Data Source

- **Publication**: Balch, J.K. et al. (2022). California wildfire spread derived using VIIRS sentinel2_rgb observations and an object-based tracking system. *Scientific Data*. https://doi.org/10.1038/s41597-022-01343-0
- **Data Repository**: [figshare](https://figshare.com/) (public data for 2012-2020)
- **Extended Data**: Our dataset (2012-2023) was accessed via request to the authors
- **Temporal Coverage**: 2012-2023 fire seasons
- **Resolution**: 375m (VIIRS native resolution)

### Data Setup

Place the GeoPackage files in `datasets/FEDS25MTBS/` organized by year:

```
datasets/FEDS25MTBS/
├── fireslist2012-2023.csv    # Fire event metadata
├── 2012/
│   └── CA3245811923420120801.gpkg
├── 2013/
│   └── ...
├── ...
└── 2023/
    └── ...
```

Each `.gpkg` file contains the `perimeter` layer with fire boundary polygons at each timestep.

### Event ID Format

Event IDs follow the pattern: `{STATE}{LAT}{LON}{DATE}`
- **STATE**: 2-letter state code (e.g., `CA`)
- **LAT**: Latitude × 100 (5 digits, e.g., `38598` for 38.598°)
- **LON**: Longitude × 100 (5 digits, e.g., `12261` for -122.61°)
- **DATE**: Fire start date as YYYYMMDD (e.g., `20171009`)

Example: `CA3859812261820171009` = California fire at (38.598°N, 122.618°W) starting October 9, 2017

## ESA WorldCover Dataset

The ESA WorldCover dataset provides global land cover classification at 10m resolution based on Sentinel-1 and Sentinel-2 data.

### Data Source

- **Provider**: European Space Agency (ESA)
- **Data Access**: Google Earth Engine (`ESA/WorldCover/v200`)
- **Temporal Coverage**: 2021
- **Resolution**: 10m
- **Coverage**: Global

### Land Cover Classes

| Value | Class | Color |
|-------|-------|-------|
| 10 | Tree Cover | Dark Green |
| 20 | Shrubland | Orange/Yellow |
| 30 | Grassland | Yellow |
| 40 | Cropland | Pink |
| 50 | Built-up | Red |
| 60 | Bare/Sparse Vegetation | Gray |
| 70 | Snow and Ice | White |
| 80 | Permanent Water Bodies | Blue |
| 90 | Herbaceous Wetland | Teal |
| 95 | Mangroves | Green |
| 100 | Moss and Lichen | Beige |

### Data Access

This dataset is automatically downloaded from Google Earth Engine during processing. No manual setup required.

## Global WUI Dataset

The Global Wildland-Urban Interface (WUI) dataset maps where buildings and wildland vegetation meet or intermingle at 10m resolution globally.

### Data Source

- **Publication**: Schug, F. et al. (2023). The global wildland–urban interface. *Nature*. https://doi.org/10.1038/s41586-023-06320-0
- **Data Repository**: Available from SILVIS Lab, University of Wisconsin-Madison
- **Temporal Coverage**: ca. 2020
- **Resolution**: 10m
- **Projection**: EQUI7 Azimuthal Equidistant

### WUI Classes

| Value | Class | Description |
|-------|-------|-------------|
| 1 | Forest/Shrub/Wetland Intermix WUI | Buildings intermixed with forest/shrub/wetland vegetation |
| 2 | Forest/Shrub/Wetland Interface WUI | Buildings adjacent to forest/shrub/wetland vegetation |
| 3 | Grassland Intermix WUI | Buildings intermixed with grassland vegetation |
| 4 | Grassland Interface WUI | Buildings adjacent to grassland vegetation |
| 5 | Non-WUI: Forest/Shrub/Wetland | Forest/shrub/wetland without WUI |
| 6 | Non-WUI: Grassland | Grassland without WUI |
| 7 | Non-WUI: Urban | Urban areas without wildland interface |
| 8 | Non-WUI: Other | Other land cover types |

### Data Setup

The Global WUI data uses the EQUI7 tiling grid. Download required tiles and place them in `datasets/globalwui/`:

```
datasets/globalwui/
├── X0065_Y0040/
│   └── WUI.tif
├── X0066_Y0040/
│   └── WUI.tif
└── ...
```

The script automatically identifies which tiles are needed based on the fire event location.

## License

See [LICENSE](LICENSE).
