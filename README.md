# FireDataLoader

A tool for downloading and processing wildfire-related geospatial data from multiple sources for machine learning and analysis.

## Data Sources

| Dataset | Description | Resolution |
|---------|-------------|------------|
| FEDS25MTBS | Fire perimeter time series | 375m |
| USGS 3DEP | Elevation | 1m |
| LANDFIRE | Canopy Bulk Density (CBD), Canopy Cover (CC) | 30m |
| HRRR | Weather: humidity (r2), wind (u10, v10) | 3km |
| Global Building Atlas | Building heights | 3m |
| ESA WorldCover | Land cover classification | 10m |
| Tree Canopy LAI | Leaf Area Index | 10m |
| NAIP/Sentinel-2 | Satellite imagery (RGB) | 1m/10m |
| Hillshade | Terrain visualization (from elevation) | 1m |

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
└── hillshade.npy         # Terrain hillshade visualization
```

### Visualizing Data

```bash
# Plot all layers overview
python plot_data.py CA3859812261820171009

# Plot time series frames (e.g., burn perimeters)
python plot_data.py CA3859812261820171009 -t burn_perimeters
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

## License

See [LICENSE](LICENSE).
