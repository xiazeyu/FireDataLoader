# FireDataLoader

A tool for downloading and processing wildfire-related geospatial data from multiple sources for machine learning and analysis.

## Data Sources

| Dataset | Description | Resolution |
|---------|-------------|------------|
| FEDS25MTBS | Fire perimeter time series | 375m |
| Fire Radiative Power (FRP) | Fire intensity from VIIRS (day/night) | 375m |
| USGS 3DEP | Elevation | 1m |
| LANDFIRE | Canopy Bulk Density (CBD), Canopy Cover (CC) | 30m |
| HRRR | Weather: humidity (r2), wind (u10, v10) | 3km |
| Global Building Atlas | Building heights | 3m |
| ESA WorldCover | Land cover classification | 10m |
| Tree Canopy LAI | Leaf Area Index | 10m |
| NAIP/Sentinel-2 | Satellite imagery (RGB) | 1m/10m |
| Hillshade | Terrain visualization (from elevation) | 1m |
| Global WUI | Wildland-Urban Interface classification | 10m |

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
# Process a fire event
python main.py CA3859812261820171009

# With verbose output
python main.py CA3859812261820171009 -v
```

## Usage

```bash
python main.py <event_id> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --resolution` | Spatial resolution (meters) | 30 |
| `-b, --buffer` | Buffer around fire bounds (meters) | 20 |
| `-c, --crs` | Target coordinate reference system | EPSG:5070 |
| `-o, --output_dir` | Output directory | output |
| `-t, --interpolation` | Intermediate frames between timesteps | 0 |
| `--herbie_cache_dir` | HRRR cache directory | datasets/herbie |
| `-v, --verbose` | Enable verbose logging | False |

### Examples

```bash
# High resolution processing
python main.py CA3859812261820171009 -r 10 -v

# With temporal interpolation (3 intermediate frames)
python main.py CA3859812261820171009 -t 3

# Custom output directory
python main.py CA3859812261820171009 -o ./my_output
```

## Output

Data is saved as `.npy` files in `output/<event_id>/`:

```
output/CA3859812261820171009/
├── task_info.npy         # Processing configuration
├── burn_perimeters.npy   # Fire perimeter time series
├── frp_day.npy           # Daytime Fire Radiative Power (MW)
├── frp_night.npy         # Nighttime Fire Radiative Power (MW)
├── elevation.npy         # Terrain elevation
├── cbd.npy               # Canopy Bulk Density
├── cc.npy                # Canopy Cover
├── r2.npy                # Relative humidity
├── u10.npy               # Wind U component
├── v10.npy               # Wind V component
├── building_height.npy   # Building heights
├── landcover.npy         # Land cover classes
├── lai.npy               # Leaf Area Index
├── satellite.npy         # RGB satellite imagery (NAIP/Sentinel-2)
├── hillshade.npy         # Terrain hillshade visualization
└── wui.npy               # Wildland-Urban Interface classification
```

### Visualizing Data

```bash
# Plot all layers overview (saves to file)
python plot_data.py CA3859812261820171009

# Plot and display interactively
python plot_data.py CA3859812261820171009 --show

# Plot time series frames (e.g., burn perimeters)
python plot_data.py CA3859812261820171009 -t burn_perimeters

# Plot time series and display interactively
python plot_data.py CA3859812261820171009 -t burn_perimeters --show
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

The Fire Event Data Suite (FEDS) provides half-daily fire perimeter time series derived from VIIRS satellite observations using an object-based tracking system.

### Data Source

- **Publication**: Balch, J.K. et al. (2022). California wildfire spread derived using VIIRS satellite observations and an object-based tracking system. *Scientific Data*. https://doi.org/10.1038/s41597-022-01343-0
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
