# FireDataForge

A tool for downloading and processing wildfire-related geospatial data from multiple sources for machine learning and analysis.

Software DOI: [10.5281/zenodo.20744740](https://doi.org/10.5281/zenodo.20744740).

Preprint: Xia, Z., Chen, L., Liu, Y., & Huang, H. (2026). *FireDataForge: A
Unified Framework for Multi-Source Wildfire Data Retrieval and Integration*.
arXiv [doi:10.48550/arXiv.2606.21198](https://doi.org/10.48550/arXiv.2606.21198).
If you use FireDataForge, please cite it — see [Citation](#citation).

## Data Sources

| Dataset | Description | Native Spatial Resolution | Temporal Resolution | Feature Name(s) |
|---------|-------------|--------------------|---------------------|-----------------|
| FEDS-MTBS | Fire perimeter polygons and active firelines tracked from clusters of satellite active-fire detections | 375 m | 12-hourly | `burn_perimeter`, `fireline` |
| VIIRS Active Fire | Fire Radiative Power (MW) reported at detected hotspot pixels, split by day vs. night overpass | 375 m | ~2 overpasses/day | `frp_daytime`, `frp_nighttime` |
| FEDS × VIIRS Active Fire *(derived)* | Maximum FRP from nearby hotspots painted onto each FEDS fireline segment | 375 m | 12-hourly | `fireline_max_frp` |
| 3DEP | Bare-earth elevation, plus a colored hill-shade RGB visualization derived from it | 1 m | Static | `elevation`, `terrain_rgb` |
| LANDFIRE | Canopy fuel layers: canopy bulk density and percent canopy cover | 30 m | Static | `canopy_bulk_density`, `canopy_cover` |
| NIFC IFPH (recent burns) | Most-recent burn year per pixel, rasterized from NIFC InteragencyFirePerimeterHistory perimeters that intersect the AOI in the prior N years (default 5; same-year fires only included if contained before the current event's `t_start`) | task grid (default 30 m) | Updated ~monthly | `recent_burn` |
| HRRR | Near-surface weather forecast fields: 2 m relative humidity and 10 m wind components | 3 km | Hourly | `r2`, `u10`, `v10` |
| Global Building Atlas | Per-building height estimates rasterized to a regular grid | 3 m | Static | `building_height` |
| WorldCover | Global land cover classification (11 classes) | 10 m | Static | `landcover` |
| Global LAI | Leaf Area Index retrieved from Sentinel-2 surface reflectance | 10 m | Single global snapshot | `lai` |
| Sentinel-2 Cloudless Mosaic | Cloud-free RGB composite built from Sentinel-2 surface-reflectance scenes | 10 m | Annual composite | `sentinel2_rgb` |
| Global WUI | Wildland–Urban Interface classes from a buildings × wildland-vegetation overlay | 10 m | Static | `wui` |

## Installation

Requires Python 3.12+ (tested on 3.14) and [uv](https://github.com/astral-sh/uv).
FireDataForge is run from source (it has no installed console script), so get the
code and create the environment with:

```bash
git clone https://github.com/xiazeyu/FireDataForge.git
cd FireDataForge
uv sync
```

`uv sync` creates a local `.venv` with every dependency. The `python main.py …`
and `python plot.py …` commands throughout this README assume that environment
is active — either activate it once per shell:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

…or prefix each command with `uv run` (e.g. `uv run python main.py <event_id>`)
to run it in the project environment without activating.

## Prerequisites

### First-run setup wizard

The **first time you run** FireDataForge (or anytime via `python main.py --setup`),
a short interactive wizard configures the optional credentials and saves them to
a local `.env` file. Everything is optional — skip any step and the
wizard tells you exactly which output features become unavailable.

```bash
python main.py --setup
```

<details>
<summary><strong>Wizard walkthrough</strong> — what it does and how to use it</summary>

Settings are read from `.env`, but values set in the real environment always take
precedence, so you can override per-run, e.g. `FIRMS_MAP_KEY=... python main.py ...`.

| Credential / dependency | Stored as | Unlocks | If unavailable |
|------------|-----------|---------|------------|
| Earth Engine auth + project | `earthengine` creds + project from `earthengine set_project` (optional `EARTHENGINE_PROJECT` override in `.env`) | `elevation`, `terrain_rgb`, `canopy_bulk_density`, `canopy_cover`, `building_height`, `landcover`, `lai`, `sentinel2_rgb` | those layers are **skipped**; all others still run |
| NASA FIRMS MAP_KEY | `FIRMS_MAP_KEY` | `frp_daytime`, `frp_nighttime` for any year (streamed from the FIRMS Area API) | pre-2025 events fall back to the bundled FEDS firepix archive; otherwise FRP is skipped |
| FEDS-MTBS archive (Zenodo) | optional full archive under `datasets/FEDS25MTBS/` | `burn_perimeter`, `fireline`, `fireline_max_frp`, perimeter-masked FRP, and the tightest fire window | auto-streamed per fire from [Zenodo](https://doi.org/10.5281/zenodo.20187962) into `cache/` at run time; set `FIREDATAFORGE_LAZY_FETCH=0` to disable, then those layers skip and FRP comes from FIRMS unmasked |
| *(none)* | — | `wui`, `recent_burn`, `hrrr` | no credential needed; fetched from public services at run time (skipped fail-soft if a service is unreachable; `hrrr` only covers events on/after 2014-09-30) |

The pipeline is **fail-soft**: a missing dependency, an unauthenticated service, or
a server under maintenance only disables the layers that need it — every other layer
is still produced, a warning is logged, and the reason is written to the per-event
`task_summary.json`.

For the FEDS-MTBS and Global WUI datasets the wizard also asks how you want the
data on disk: **full download** (the whole archive is fetched and unpacked into
`datasets/`, trading disk for bandwidth on repeated runs) or **on-the-fly** (only
the bits each fire needs are fetched into `cache/` as you go). Either way nothing
needs to be pre-staged by hand.

> **Two on-disk buckets** (both optional):
> - `datasets/` — **user-managed**. Full archives you deliberately download (via
>   the wizard or by hand). FireDataForge only writes here on an explicit
>   user-requested download.
> - `cache/` — **software-managed**. Everything fetched on the fly (per-fire
>   GeoPackages, FIRMS slices, HRRR GRIBs, WUI tiles, the MTBS fire list) lands
>   here. Readers check `datasets/` first, then `cache/`. Delete `cache/` anytime
>   to reclaim space; it is transparently re-fetched. Delete `datasets/` too to
>   reclaim more (then everything streams on demand again).

The wizard's final step lets you choose where each fire's **name + acreage** come
from — the FEDS-MTBS fire list (Zenodo, FEDS-aligned acreage, 2012–2024, ~280 MB),
a built-once **offline MTBS fire list** (`python main.py --build-firelist`,
mtbs.gov, more recent), or live on-the-fly lookups — since the GeoPackage already
supplies everything else. See the
[resolution table](#how-an-event-id-is-resolved) for the tradeoffs.

</details>

<details>
<summary><strong>Manual setup</strong> — configure credentials yourself instead of using the wizard</summary>

**Google Earth Engine**

```bash
# Authenticate (one-time, opens browser)
earthengine authenticate

# Set your project ID
earthengine set_project <YOUR-PROJECT-ID>
```

To get a project ID:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project with [Earth Engine enabled](https://developers.google.com/earth-engine/guides/access)
3. Copy the project ID

**NASA FIRMS API Key** (for VIIRS active fire)

Active-fire data is streamed on demand from the NASA
FIRMS [Area API](https://firms.modaps.eosdis.nasa.gov/api/area/) — only the
points inside each event's bounding box and date window are downloaded. Get a free MAP_KEY and either let the
wizard save it, export it, or add it to `.env`:

```bash
# Request a key (instant, free): https://firms.modaps.eosdis.nasa.gov/api/map_key/
export FIRMS_MAP_KEY=your-map-key-here
# ...or in .env:  FIRMS_MAP_KEY=your-map-key-here
```

If no FIRMS key is configured, the pipeline falls back to bundled archive CSVs in
`datasets/FIRMS/` (see [FIRMS / VIIRS Active Fire Dataset](#firms--viirs-active-fire-dataset)).
Pre-2025 events use FEDS firepix and need no key.

**FEDS-MTBS archive** (for `burn_perimeter`, `fireline`, `fireline_max_frp`)

No credential needed — each fire is streamed on demand. To pre-stage the examples
or download the full archive, see [FEDS-MTBS Dataset](#feds-mtbs-dataset).

</details>

## Quick Start

```bash
# Download the example fires from Zenodo
python main.py --fetch-examples

# Process a single fire event given its MTBS Event ID
python main.py CA3432611848120191010
```

The argument is an **MTBS Event ID** — FireDataForge's canonical event key. Any
valid ID works (unrecognized IDs are resolved live from mtbs.gov, no archive
needed); see [Looking up an MTBS Event ID](#looking-up-an-mtbs-event-id) and
[Event ID Format](#event-id-format) for the character scheme.
`CA3432611848120191010` is the 2019 Saddleridge fire, one of the bundled
[example events](#example--evaluation-fire-events).

## Usage

### Single Event

```bash
python main.py <event_id> [options]
```

### Batch Processing

```bash
# From comma-separated list  
python main.py --batch CA123,CA456,CA789 [options]

# From a file (one event ID per line)
python main.py --batch events.txt [options]
```

<details>
<summary><strong>Options</strong></summary>

| Option | Description | Default |
|--------|-------------|---------|
| `--batch` | Batch mode: file path or comma-separated event IDs | - |
| `--setup` | Run the interactive credential wizard and exit | - |
| `--build-firelist` | Download the full MTBS archive to the offline cache and exit | - |
| `--fetch-examples` | Download the example fires from the Zenodo reproducibility artifact ([doi:10.5281/zenodo.20743743](https://doi.org/10.5281/zenodo.20743743); into `datasets/FEDS25MTBS/` + `events.txt`) and exit | - |
| `-w, --workers` | Events processed in parallel in batch mode | 1 |
| `--layer-workers` | Concurrent layer downloads within a single event | 5 |
| `-r, --resolution` | Spatial resolution (meters) | 30 |
| `-b, --buffer` | Buffer around fire bounds (meters) | 100 |
| `-c, --crs` | Target coordinate reference system (must be a projected/metric CRS — see note below) | EPSG:5070 |
| `-o, --output_dir` | Output directory | output |
| `-t, --interpolation` | Intermediate frames between timesteps | 0 |
| `--cache_dir` | Root directory for all on-the-fly downloads (HRRR, FIRMS, FEDS, firepix, WUI, fire list); each caches under its own fixed subfolder | cache |
| `--only` | Only process specific feature(s), comma-separated | all |
| `-v, --verbose` | Enable verbose logging | False |

#### Available Features for `--only`

| Feature | Description |
|---------|-------------|
| `burn_perimeter` | Fire perimeter time series from FEDS |
| `fireline` | Active fireline derived from consecutive perimeter differences |
| `fireline_max_frp` | Per-pixel maximum FRP along the fireline |
| `frp_daytime` | Daytime Fire Radiative Power (NASA FIRMS / FEDS firepix) |
| `frp_nighttime` | Nighttime Fire Radiative Power (NASA FIRMS / FEDS firepix) |
| `elevation` | USGS 3DEP elevation |
| `canopy_bulk_density`, `canopy_cover` | LANDFIRE canopy fuel layers (alias: `landfire`) |
| `recent_burn` | Most-recent burn year per pixel from NIFC InteragencyFirePerimeterHistory (default 5-yr lookback; lookback recorded in the layer's `note`) |
| `building_height` | Global Building Atlas heights |
| `landcover` | ESA WorldCover classification |
| `lai` | Leaf Area Index |
| `sentinel2_rgb` | Sentinel-2 cloudless RGB mosaic |
| `terrain_rgb` | Colored shaded-relief terrain (Google-Maps style, RGB) |
| `wui` | Wildland-Urban Interface classification |
| `r2`, `u10`, `v10` | HRRR weather: relative humidity + 10 m wind components (alias: `hrrr`) |

Each name above is an output file stem. The convenience aliases `landfire` and
`hrrr` select all of their respective layers.

</details>

> **Target CRS must be projected/metric.** `--resolution` and `--buffer` are in
> **meters**, and the grid snaps the projected bounds to whole multiples of the
> resolution. A geographic CRS in degrees such as `EPSG:4326` is therefore **not
> supported** and is **rejected up front** with a clear error, since meter-valued
> settings applied in degrees would produce a nonsensical grid. Use a projected,
> meter-based CRS: the default `EPSG:5070` (CONUS Albers), a UTM zone, or similar.

<details>
<summary><strong>Examples</strong></summary>

```bash
# High resolution processing
python main.py CA3432611848120191010 -r 10 -v

# With temporal interpolation (3 intermediate frames)
python main.py CA3432611848120191010 -t 3

# Custom output directory
python main.py CA3432611848120191010 -o ./my_output

# Batch process from file with 4 workers
python main.py --batch events.txt -w 4 -o results/

# Batch process specific events
python main.py --batch CA123,CA456,CA789 --workers 3

# Process only a single feature (for quick debugging)
python main.py CA3432611848120191010 --only frp_daytime

# Process multiple specific features
python main.py CA3432611848120191010 --only frp_daytime,frp_nighttime,elevation

# Regenerate only weather data
python main.py CA3432611848120191010 --only hrrr
```

The batch summary is saved to `output/batch_summary.json`.

</details>

## Output

Data is saved as `.npy` files in `output/<event_id>/`:

```
output/CA3432611848120191010/
├── task_summary.json     # Per-layer outcome (ok / skipped / failed) + reasons + metadata
├── task_info.npy         # Processing configuration
├── coordinates.npy       # Pixel-center x/y coordinates + CRS for the grid
├── burn_perimeter.npy    # Fire perimeter time series
├── fireline.npy          # Active fireline time series (perimeter differences)
├── fireline_max_frp.npy  # Max FRP painted onto each fireline segment
├── frp_daytime.npy       # Daytime Fire Radiative Power (MW)
├── frp_nighttime.npy     # Nighttime Fire Radiative Power (MW)
├── elevation.npy         # Terrain elevation
├── canopy_bulk_density.npy  # Canopy Bulk Density
├── canopy_cover.npy      # Canopy Cover
├── recent_burn.npy       # Most-recent burn year per pixel (NIFC IFPH, NaN = unburned)
├── r2.npy                # Relative humidity
├── u10.npy               # Wind U component
├── v10.npy               # Wind V component
├── building_height.npy   # Building heights
├── landcover.npy         # Land cover classes
├── lai.npy               # Leaf Area Index
├── sentinel2_rgb.npy     # RGB Sentinel-2 cloudless mosaic
├── terrain_rgb.npy       # Colored shaded-relief terrain RGB (H, W, 3)
└── wui.npy               # Wildland-Urban Interface classification
```

Only the layers that were successfully produced are written; any that were skipped
or failed are omitted from the directory but **always recorded** in
`task_summary.json` with the reason.

<details>
<summary><strong>Task summary</strong> (<code>task_summary.json</code>) — per-layer outcome schema</summary>

Every event writes a `task_summary.json` describing the run and the outcome of each
layer, so a partial run is self-documenting:

```json
{
  "event_id": "CA3432611848120191010",
  "name": "SADDLERIDGE", "year": 2019, "status": "partial",
  "crs": "EPSG:5070", "resolution_m": 30, "shape": [275, 377],
  "t_start": "2019-10-01T12:00:00", "t_end": "2019-10-16T12:00:00",
  "t_end_estimated": true,
  "has_feds_archive": false, "earth_engine": true, "firms_key": true,
  "notes": ["t_end is an estimate (t_start + 15 days): no FEDS perimeter ..."],
  "layers": {
    "elevation":      {"status": "ok",      "files": ["elevation.npy"]},
    "frp_daytime":    {"status": "ok",      "files": ["frp_daytime.npy"], "n_frames": 7},
    "burn_perimeter": {"status": "skipped", "reason": "no local FEDS archive"},
    "wui":            {"status": "failed",  "reason": "..."}
  },
  "counts": {"ok": 2, "skipped": 1, "failed": 1}
}
```

`status` is `"ok"` when all layers succeed, `"partial"` when any failed, or
`"error"` if the event itself could not be resolved. Batch runs additionally write
an aggregated `output/batch_summary.json`.

</details>

<details>
<summary><strong>Grid coordinates</strong> (<code>coordinates.npy</code>) — georeferencing the saved arrays</summary>

Every event directory also contains `coordinates.npy`, which stores the
pixel-center coordinates of the common output grid together with the CRS.
Almost all raster layers (`elevation.npy`, `frp_*.npy`, `wui.npy`, ...) are
sampled on this exact grid, so this file is the single source of truth for
georeferencing the arrays — useful for wrapping outputs into `xarray`
DataArrays or re-projecting them when preparing publication figures.

> **Weather layers are the exception.** `r2.npy`, `u10.npy`, and `v10.npy`
> (HRRR) share the same `bounds` and `crs` but sit on a coarser ~500 m grid
> (HRRR is ~3 km natively; resampling an hourly series to 30 m would bloat the
> files for no added detail). Each records its grid size in
> `current_resolution`; reconstruct their grid from the shared `bounds` and the
> array's own shape, not from `coordinates.npy`.

```python
from main import load_numpy

coords = load_numpy('output/CA3432611848120191010/coordinates.npy')
x, y = coords.data                  # 1-D arrays, shape (width,) and (height,)
geo = coords.georeference           # typed GeoReference (see schemas.py)
crs = geo.crs                       # e.g. 'EPSG:5070'
crs_wkt = geo.crs_wkt               # full WKT2 (works without an EPSG db)
crs_proj4 = geo.crs_proj4           # legacy PROJ string
bounds = geo.bounds                 # (minx, miny, maxx, maxy)
height, width = geo.shape
a, b, c, d, e, f = geo.transform    # affine: (col, row) -> (x, y)
```

For standard EPSG codes, `geo.crs` alone is enough for pyproj /
rasterio / cartopy. The extra `crs_wkt` and `crs_proj4` fields are
included so the file is fully self-describing — useful for archival,
custom CRSes (e.g. EQUI7, HRRR Lambert), or environments without a PROJ
database.

`y` is ordered top-to-bottom (north → south) to match the row order of the
saved rasters.

</details>

<details>
<summary><strong>Array shapes &amp; dtypes</strong> — what each <code>.npy</code> holds</summary>

Every `.npy` (except `task_info.npy`) loads via `load_numpy` into a `DataLayer`
whose `.data` is **always a list of frames** (`schemas.py`). The frame rank and
the number of frames follow the layer's type:

| Layer kind | Layers | `len(data)` | Frame shape | dtype |
|------------|--------|-------------|-------------|-------|
| Static raster (integer) | `elevation`, `canopy_bulk_density`, `canopy_cover` | 1 (`timestamps == [t_start]`) | `(H, W)` | int16 |
| Static raster (float) | `lai`, `building_height`, `recent_burn` | 1 | `(H, W)` | float (32/64) |
| Static categorical | `landcover`, `wui` | 1 | `(H, W)` | int (`landcover` int16, `wui` uint8) |
| Time-varying mask | `burn_perimeter`, `fireline` | `T` frames (`len(data) == len(timestamps)`) | `(H, W)` | bool |
| Time-varying raster | `fireline_max_frp`, `frp_daytime`, `frp_nighttime` | `T` frames | `(H, W)` | float |
| Time-varying raster | `r2`, `u10`, `v10` (HRRR) | `T` frames | `(h, w)` *(coarser grid)* | float |
| RGB visualization | `sentinel2_rgb`, `terrain_rgb` | 1 | `(H, W, 3)` | uint8 |
| Coordinates | `coordinates` | 2 | `[x: (W,), y: (H,)]` | float |

`(H, W)` is the common grid in `coordinates.npy` for every layer **except** the
HRRR fields (`r2`/`u10`/`v10`), which sit on a coarser `(h, w)` grid recorded in
their `current_resolution` — derive it from the shared `bounds` and the array's
own shape. For time-varying layers `data[i]` is the frame observed at
`timestamps[i]` (the pairing the temporal cursor relies on; see
[temporal alignment](#nodata-missing-data--temporal-alignment)). Per-pixel nodata
sentinels are in the same section's table. Each layer also carries, inside its
`DataLayer`, `native_resolution` (the source's true resolution, in **meters**),
`current_resolution` (the grid it is sampled on, in **meters**), `unit`,
`source`, and — for categorical layers — `categories`.

</details>

### Visualizing Data

```bash
# Plot everything (default): the combined overview grid, one PNG per channel,
# and one time-series figure per multi-frame layer
python plot.py CA3432611848120191010
```

<details>
<summary><strong>Other plotting options</strong></summary>

```bash
# Only the combined overview grid
python plot.py CA3432611848120191010 --mode overview

# Only one PNG per channel
python plot.py CA3432611848120191010 --mode channels

# Only the time-series figures (one per multi-frame layer)
python plot.py CA3432611848120191010 --mode timeseries

# Overview grid + per-channel PNGs, but skip time series
python plot.py CA3432611848120191010 --mode both

# Also write the overview as PDF (PNG only by default)
python plot.py CA3432611848120191010 --pdf

# Plot and display interactively
python plot.py CA3432611848120191010 --show

# Plot the time series for ONLY one layer (e.g., burn perimeters)
python plot.py CA3432611848120191010 -t burn_perimeter

# Batch plot multiple events
python plot.py --batch events.txt

# Batch plot from comma-separated list
python plot.py --batch CA123,CA456,CA789
```

</details>

### Loading Data

```python
from main import load_numpy

# Load a data file
data = load_numpy('output/CA3432611848120191010/elevation.npy')
print(data.name)        # 'elevation'
print(data.data[0].shape)  # (height, width)
print(data.unit)        # 'm'
```

### Data contract (`schemas.py`)

The output format is a small, stable API defined in [`schemas.py`](schemas.py) —
the typed dataclasses every layer deserializes into. `load_numpy` returns a
`DataLayer`, the universal envelope wrapping a layer's frames, timestamps, unit,
categories, and (for the grid) a `GeoReference`; `FireEvent`, `ProcessingTask`,
and `ProcessingArgs` describe the event and run configuration, and
`SCHEMA_VERSION` tags the envelope version.

`schemas.py` is **standard-library only** (no `rasterio` / `earthengine` / other
geo dependencies), so a third-party application can import — or simply vendor —
this single file to read, type-check, and validate FireDataForge outputs without
installing the rest of the pipeline.

### Python API

```python
from firedataforge import forge_event, get_fire_info, get_task_info, ProcessingArgs

# One call: resolve, retrieve, harmonize, and write every available layer.
summary = forge_event("CA3432611848120191010", ProcessingArgs(resolution=30))
print(summary["counts"])          # {'ok': ..., 'skipped': ..., 'failed': ...}

# Or step through the resolution stages.
fire = get_fire_info("CA3432611848120191010")
print(f"{fire.name}: {fire.acres_burned} acres")
task = get_task_info(fire, resolution=30)
print(f"Grid: {task.shape}, CRS: {task.crs}")
```

### Downstream examples & validation

- `examples/ml_dataloader.py` — a PyTorch `Dataset` that stacks the static layers
  of each event into a `(C, H, W)` tensor (`python examples/ml_dataloader.py [output]`).
- `examples/fire_spread_demo.py` — a self-contained NumPy fire-spread automaton that
  consumes the harmonized terrain/fuel/wind layers and shows a 3-frame progression
  with matplotlib (`python examples/fire_spread_demo.py [output/<event_id>]`).

  Both are written as notebook-style `# %%` cell scripts — run them top-to-bottom, or
  open them in a Jupyter/VS Code interactive window.
- `validation/` — quantitative checks (reprojection round-trip error, FRP
  conservation, categorical overall accuracy, continuous RMSE, sub-pixel
  registration); see [`validation/README.md`](validation/README.md).

<details>
<summary>Repository layout</summary>

The implementation lives in the `firedataforge/` package; `main.py` is a thin CLI
entry point that re-exports the public API.

```
main.py                  CLI entry point + backward-compatible import surface
schemas.py               public data contract — stdlib-only output dataclasses
plot.py                  visualization of saved layers
firedataforge/
├── constants.py         paths + source-API constants
├── config.py            credentials, first-run wizard, dataset discovery
├── events.py            fire-list resolution + ProcessingTask (grid + time window)
├── io.py                .npy + coordinate persistence
├── pipeline.py          fail-soft per-event / batch orchestration + task summaries
├── cli.py               argument parsing
├── examples.py          --fetch-examples: download the example bundle from Zenodo
├── remote_archive.py    on-the-fly range-fetch of FEDS fires/firepix from Zenodo
├── progress.py          download + zip-extraction progress helpers
└── sources/             one module per source family
    ├── mtbs.py          MTBS Event-ID → metadata
    ├── feds.py          perimeter / fireline / fireline_max_frp
    ├── frp.py           VIIRS FRP (FIRMS / firepix)
    ├── gee.py           Earth Engine layers (3DEP, LANDFIRE, GBA, WorldCover, LAI, S2, terrain)
    ├── weather.py       HRRR
    ├── wui.py           Global WUI
    └── nifc.py          recent burns
```
</details>

## Resampling and Harmonization

Every source is reprojected to the target CRS and resampled to the target grid
with a single, **direction-independent** method chosen by the data *type* — the
same method is used whether the native source is finer or coarser than the target
grid (there is no conditional on the scale ratio). Categorical layers keep
nearest-neighbour deliberately: it introduces no mixed/invented classes and
preserves the expected class proportions (majority-vote aggregation would erode
thin minority features such as narrow WUI strips).

| Data type | Layers | Method | Why |
|-----------|--------|--------|-----|
| Continuous raster | `elevation`, `canopy_bulk_density`, `canopy_cover`, `lai`, `r2`, `u10`, `v10` | Bilinear | Smooth fields; bilinear avoids blocky artifacts. When **upsampling** a coarse source (e.g. 3 km HRRR → 30 m) it interpolates but adds **no real detail**; when **downsampling** a fine source (e.g. 1 m DEM → 30 m) it point-samples and discards sub-pixel variance (quantified in [`validation/`](validation/README.md)). |
| Categorical raster | `landcover`, `wui` | Nearest neighbour | No mixed/invented classes; preserves class proportions. At heavy downsampling, sub-grid features smaller than a pixel may be dropped rather than blended. |
| RGB visualization | `sentinel2_rgb`, `terrain_rgb` | Nearest / bilinear | Display layers; resampling is cosmetic, not analytic. |
| Vector polygon | `burn_perimeter`, `fireline` | Rasterization (`all_touched`) | Any pixel the polygon touches is burned in, so thin firelines survive at coarse grids. |
| Vector polygon (weighted) | `building_height` | Area-weighted mean | Larger building footprints contribute proportionally more to a pixel's mean height. |
| Point mass | `frp_daytime`, `frp_nighttime` | Mass-preserving Gaussian splat (below) | Conserves total radiative power across the regrid. |
| Point max | `fireline_max_frp` | Per-segment nearby max | Keeps the observed peak FRP (MW) intensity rather than a spread share. |

> **Choosing the target resolution.** A fine target grid does not create
> information that a coarse source lacks — upsampling 3 km HRRR weather to 30 m
> yields smooth but not genuinely fine fields, so treat such layers as their
> native resolution despite the grid spacing. Conversely a coarse target grid
> discards real local variation in fine sources (terrain, fuels, buildings, WUI).
> Each layer records its true `native_resolution` (and, where it differs, its
> `current_resolution`) — both in **meters** — in the saved metadata so the scale
> mismatch is never hidden.

### Mass-preserving Gaussian splat (VIIRS FRP)

Each VIIRS active-fire detection reports a Fire Radiative Power (FRP, MW) at a
point. Rather than dropping the whole value into one grid cell, the detection's
FRP is spread over a Gaussian footprint the size of the VIIRS sensor pixel, so
the rasterized field reflects the sensor's true spatial uncertainty while
**conserving the total radiative power**.

For a detection with value `F` at grid position `(pₓ, p_y)` (pixel units):

- **Footprint width.** `σ = (source_resolution / target_resolution) / 2` pixels,
  with `source_resolution = 375 m` (VIIRS). At a 30 m grid, `σ = 375/30/2 = 6.25 px`
  (= 187.5 m), giving a full-width-at-half-maximum `FWHM = 2√(2 ln 2)·σ ≈ 2.355 σ ≈ 441 m`.
- **Kernel extent.** Weights are evaluated over a square window of radius
  `⌈3σ⌉` pixels (covering ±3σ, ≈ 99.7 % of the Gaussian mass).
- **Weights.** A pixel whose center is at distance `d` (pixels) from the
  detection gets `w = exp(−d² / (2σ²))`.
- **Normalization (conservation).** The in-grid weights are normalized to sum to
  one and the deposit is `F · w / Σw`. Summing the rasterized footprint therefore
  recovers `F` exactly for any detection whose footprint lies inside the grid;
  the only loss is the fraction of a footprint that falls off the grid edge. Each
  pixel value is thus a *share* of the detection's FRP (≪ the observed MW), not
  the observed value itself.

The `validation/` suite re-splats each event's detections and reports the
relative conservation error (typically far below 1 %); see
[`validation/README.md`](validation/README.md).

## Nodata, missing data & temporal alignment

FireDataForge degrades gracefully at three levels — a whole layer, an individual
timestep, and an individual pixel — and records every gap, so a partial cube is
never silently mistaken for a complete one.

**Layer level — fail-soft.** If a source is unauthenticated, unreachable, or under
maintenance, only the layers that depend on it are dropped; every other layer is
still produced. A dropped layer is **omitted from the event directory** and recorded
in `task_summary.json` with `status: "skipped"`/`"failed"` and a human-readable
`reason` (see the [Output](#output) section's task-summary schema).

**Pixel level — per-layer nodata sentinels.** There is **no universal nodata
value**; the sentinel follows the data *type*, so check this table (and each layer's
`unit`) before masking:

| Layer(s) | dtype | "No data / absent" value |
|----------|-------|--------------------------|
| `elevation`, `canopy_bulk_density`, `canopy_cover` (int16), `lai`, `building_height` (float) | int16 / float | `0` — Earth Engine masks (out-of-coverage, water) are filled with `0` (`unmask(0)`); these layers carry **no NaN**, so where the distinction matters treat `0` as "no data" rather than a measured zero |
| `landcover`, `wui` | int | `0` — outside the source's coverage; defined classes are ≥ 10 (WorldCover) / 1–8 (WUI) |
| `recent_burn` | float | `NaN` = pixel never burned in the lookback window (stated explicitly in the layer `unit`) |
| `frp_daytime`, `frp_nighttime`, `fireline_max_frp` | float | `0` = no active-fire detection at/near the pixel (FRP is a conserved *share* of MW, so `0` means "no fire", not "missing") |
| `burn_perimeter`, `fireline` | bool | `False` (`0`) = outside the perimeter / not on the fireline at that timestep; `True` (`1`) = inside |
| `r2`, `u10`, `v10` (HRRR) | float | dense over CONUS — no per-pixel nodata in practice |
| `sentinel2_rgb`, `terrain_rgb` | uint8 | display layers; `0` where the source mosaic had no pixel |

**Timestep level — the per-source datetime cursor.** Time-varying layers
(`burn_perimeter`, `fireline`, `fireline_max_frp`, `frp_*`, and the HRRR fields)
store **only the frames actually observed**, held in `DataLayer.data` paired
one-to-one with `DataLayer.timestamps` (`schemas.py`). Sources observe at different,
irregular cadences — FEDS perimeters/firelines every 12 h, VIIRS at ~2 overpasses/day,
HRRR hourly — and **each keeps its own timestamp vector**, so cadences are never
resampled onto a shared clock. A consumer reads each layer through a *datetime
cursor*: at simulation time `t`, take the frame at the most recent
`timestamps[i] <= t` for that layer, advancing each source independently — the same
"most-recent observation ≤ t" rule the pipeline itself uses to mask FRP to the live
perimeter (`firedataforge/sources/frp.py`). Consequences:

- **A skipped or missing overpass is a no-op:** the previous frame stays current
  until the next real observation. The pipeline never forward-fills synthetic values
  or fabricates a frame for a cadence it did not observe. (Pass `--interpolation N`
  to *explicitly* synthesize `N` SDF-interpolated perimeter frames between timesteps.)
- **The cursor never exposes a not-yet-valid observation:** it cannot return a frame
  whose timestamp is after `t`, so temporal alignment is correct by construction —
  there is nothing to "synchronize" after the fact.
- **Out-of-window / empty sources are excluded upstream:** every frame is bounded to
  the event's active-burning window (`t_start`–`t_end`); a source with zero valid
  observations in that window degrades to the layer-level fail-soft case above rather
  than emitting empty frames.

Beyond gaps, every `.npy` envelope is self-describing: it carries `source`
(provenance/attribution), `unit`, `native_resolution` (the source's resolution in
**meters**), `timestamps`, and — for categorical layers — a `categories` map,
while the grid CRS/transform lives in `coordinates.npy`. See
[Data contract (`schemas.py`)](#data-contract-schemaspy).

## Available Fire Events

Any **MTBS Event ID** (Monitoring Trends in Burn Severity) is a valid input, and
you do **not** need the FEDS-MTBS archive to choose or resolve one: unrecognized
IDs are looked up live from [mtbs.gov](https://www.mtbs.gov/) at run time, and
`python main.py --build-firelist` caches the full MTBS list offline for browsing
and network-free resolution.

If you have the full FEDS-MTBS archive, the summary
`fireslist_FEDS25MTBS_2012-2024.geojson` (the `Event_ID` column) is a ready-made
list of all 7,739 FEDS-covered events (2012–2024).

See [Event ID Format](#event-id-format) for the `{STATE}{LAT}{LON}{DATE}` scheme.

### Example & evaluation fire events

`python main.py --fetch-examples` stages the eight fires below and writes their IDs
to `events.txt` at the repo root (ready for `--batch events.txt`); they are the
events used for the benchmark and the
[quantitative validation](validation/README.md). The bundle — these eight fires
plus the benchmark and validation reference outputs — is archived on Zenodo as a
citable reproducibility artifact
([doi:10.5281/zenodo.20743743](https://doi.org/10.5281/zenodo.20743743)).
They span 2013–2025 and ~9.7k–189k
acres across **5 California** (chaparral / coastal, wind-driven) and **3 Colorado**
(montane conifer / grassland, terrain-driven) fires, chosen to exercise the pipeline
across diverse fuels, scales, and spread regimes.

| Fire | Year | State | MTBS Event ID | Burned area (ac) | Active-fire window (UTC) | Grid (H×W) |
|------|------|-------|---------------|------------------|--------------------------|------------|
| Black Forest | 2013 | CO | `CO3901210474920130611` | 11,885 | 2013-06-11 → 06-13 | 369 × 462 |
| Tubbs | 2017 | CA | `CA3859812261820171009` | 36,981 | 2017-10-09 → 10-15 | 848 × 681 |
| Spring Creek | 2018 | CO | `CO3749610529120180627` | 107,108 | 2018-06-28 → 07-10 | 1194 × 944 |
| Camp | 2018 | CA | `CA3982012144020181108` | 153,687 | 2018-11-08 → 11-20 | 1397 × 1448 |
| Saddleridge | 2019 | CA | `CA3432611848120191010` | 9,654 | 2019-10-01 → 10-12 | 361 × 388 |
| East Troublesome | 2020 | CO | `CO4020310623920201014` | 188,924 | 2020-10-14 → 10-24 | 1016 × 1602 |
| Palisades | 2025 | CA | `CA3406811855120250107` | 23,448 † | 2025-01-07 → 01-11 | 572 × 716 |
| Eaton | 2025 | CA | `CA3419211810520250108` | 14,021 † | 2025-01-08 → 01-10 | 396 × 519 |

Burned area is the MTBS `BurnBndAc`. The two 2025 fires (Palisades, Eaton) are not
yet in the MTBS final record, so the value marked **†** is the MTBS *Provisional
Initial Assessment* acreage instead (the pre-final figure FireDataForge resolves
these events to; see `PROVISIONAL_IA` in `firedataforge/sources/mtbs.py`). It is
not directly comparable to the FEDS VIIRS perimeter area in each fire's
GeoPackage, which runs larger (~31.8k ac Palisades, ~18.6k ac Eaton) because the
375 m active-fire perimeter over-bounds the refined MTBS burn boundary.
The **active-fire window** is each event's perimeter-growth
period taken from the FEDS progression (observation-derived, not the estimated
fallback); the exact `t_start`/`t_end` (including the time of day) are in each
event's `task_summary.json`. **Grid** is the output raster size on the default
EPSG:5070 30 m grid (100 m buffer); it scales with `-r`/`-b`.

### Looking up an MTBS Event ID

If you know a fire by name, year, or location rather than by ID:

1. Open the [MTBS Data Explorer](https://www.mtbs.gov/viewer/) (interactive map)
   or the [Direct Download / search page](https://www.mtbs.gov/direct-download).
2. Filter by **fire name**, **year**, and **state/region** (or click the fire on
   the map).
3. Read the **Fire ID** (a.k.a. Event ID) field of the matching record — e.g.
   `CA3432611848120191010` — and pass it to `python main.py <Event_ID>`.

Querying by the exact MTBS Event ID (rather than a name or bounding box) is what
makes event matching unambiguous and reproducible: one ID always resolves to the
same fire. To browse offline, run `python main.py --build-firelist` once and
search the resulting `cache/mtbs_firelist.csv`.

## FEDS-MTBS Dataset

FEDS-MTBS derives fire perimeters and firelines every 12 hours from VIIRS active-fire hotspots via object-based tracking, constrained to MTBS burn records.

> **Note:** The FEDS-MTBS archive is **optional**. FireDataForge streams each
> requested fire on demand, a few hundred KB per fire with no full download — run
> `python main.py --fetch-examples` to stage the eight example fires, or grab the
> full archive for heavy offline use. Without it the FEDS layers — `burn_perimeter`,
> `fireline`, `fireline_max_frp` — are skipped and FRP comes from NASA FIRMS
> unmasked; every other layer is unaffected.

<details>
<summary>Dataset details — source, event-ID resolution, setup, and ID format</summary>

### Data Source

- **Publication**: Chen, Y. et al. (2022). California wildfire spread derived using VIIRS satellite observations and an object-based tracking system. *Scientific Data*. https://doi.org/10.1038/s41597-022-01343-0
- **Dataset (used here)**: FEDS-MTBS, the MTBS-constrained extension from the UCI–UBC–NASA fire-tracking group, published on Zenodo (DOI [10.5281/zenodo.20187962](https://doi.org/10.5281/zenodo.20187962)).
- **Temporal Coverage**: 2012–2024 fire seasons (7,739 fires; FireDataForge also ships two 2025 example fires)
- **Resolution**: 375 m (VIIRS native resolution)

### How an Event ID is resolved

The overall priority is **gpkg › FEDS-MTBS fire list › MTBS fire list › MTBS
online**. The GeoPackage ranks first but has no fire *name*, so the name/acreage
fall to the first fire-list/online source that has the event, while the gpkg (when
present) supplies the bounds and active-burning window. Concretely, fire metadata
(name, year, acres, bounds, start/end) is resolved in order, first hit wins:

1. **FEDS-MTBS fire list** (Zenodo) in `datasets/FEDS25MTBS/` — the bundled
   `fireslist_examples.csv` (FEDS perimeter bbox plus both `tst` and `ted` for the
   eight demo fires; **preferred when present**) or the released
   `fireslist_FEDS25MTBS_2012-2024.geojson` GeoPackage (MTBS final-perimeter bbox
   + `Ig_Date` for all 7,739 fires).
2. **MTBS fire list** (`cache/mtbs_firelist.csv`, from `mtbs.gov`) — a *different*
   source: grown from prior live lookups, or pre-built with
   `python main.py --build-firelist`; covers all ~30k MTBS fires; carries the MTBS
   burn-boundary bbox.
3. **Live MTBS service** (`mtbs.gov`) + a small Provisional IA supplement for very
   recent fires — the resolved record is appended to the offline MTBS list.
4. **FEDS GeoPackage only** — if none of the above has the event but a local (or
   fetchable) `<event_id>.gpkg` exists, the run still proceeds with the Event ID as
   the name and the gpkg's bounds/window.

Whenever a local GeoPackage exists, its perimeter time series and extent take
precedence: `t_end` comes from the progression (tightest), then the example list's
`ted`, then an estimate of `t_start + 15 days` (flagged `t_end_estimated: true` in
`task_summary.json`); bounds come from the perimeter extent, then the fire-list
bbox, then the MTBS bbox. So the gpkg alone supplies the window and bounds — the
fire list mainly adds the display **name** and **acreage** for offline use.

Because the name/acreage are all the fire list adds, the source is your choice
(the setup wizard's step 5 stages it, or pick by what you place in
`datasets/FEDS25MTBS/`):

| Source | Acreage | Coverage | Reliability / cost |
|--------|---------|----------|--------------------|
| **FEDS-MTBS fire list** (`fireslist_FEDS25MTBS_2012-2024.geojson`, Zenodo) | aligned to FEDS | 2012–2024 only | offline, most reliable; ~280 MB |
| **MTBS fire list** (`cache/mtbs_firelist.csv`, `--build-firelist`) | MTBS (not FEDS-aligned) | all ~30k MTBS fires, more recent | offline once built (~30 s) |
| **On-the-fly** (live `mtbs.gov`) | MTBS (not FEDS-aligned) | most up-to-date | per-event network, least reliable |

If none is available, the run still proceeds with the **Event ID** as the name.

### Data Setup

Three ways to get the data, in increasing weight:

1. **Nothing** (default, on-the-fly) — when you forge a fire whose GeoPackage
   isn't local, FireDataForge range-fetches just that fire (and, on first use of a
   year, that year's firepix CSV) out of the Zenodo archive and caches it under
   `cache/FEDS25MTBS/<year>/<event_id>.gpkg` (the same layout as the archive).
   Needs network at run time; set `FIREDATAFORGE_LAZY_FETCH=0` to turn this off for
   a fully offline/deterministic run.
2. **Examples** — `python main.py --fetch-examples` downloads `examples.zip` and
   unzips it at the repo root, pulling the eight demo fires up front into
   `datasets/FEDS25MTBS/` (with their firepix, the offline `fireslist_examples.csv`,
   and a top-level `events.txt`).
3. **Full archive** — let the wizard download it (or call
   `download_full_feds_archive()`): it pulls `FEDS25MTBS.zip` (GeoPackages +
   firepix) into `datasets/FEDS25MTBS/` so all fires resolve offline. The fire
   **name/acreage** list is a separate, optional choice (it is *not* in the zip):
   stage `fireslist_FEDS25MTBS_2012-2024.geojson` via the wizard's step 5 or
   `download_feds_firelist()`, or skip it and let names come from `mtbs.gov` /
   the Event ID — see [How an Event ID is resolved](#how-an-event-id-is-resolved).

The loader checks the user archive (`datasets/FEDS25MTBS/`) first, then the cache
(`cache/FEDS25MTBS/`), and searches recursively for `<event_id>.gpkg`:

```
datasets/FEDS25MTBS/
├── fireslist_FEDS25MTBS_2012-2024.geojson  # Optional: MTBS perimeters + metadata (7,739 fires)
├── fireslist_examples.csv   # Example fire metadata (Event_ID, Year, tst, ted, bbox)
├── firepix/                 # Per-fire VIIRS active-fire CSVs (pre-2025 FRP source)
├── 2012/
│   └── CA3245811923420120801.gpkg
├── ...
└── 2024/
    └── ...
```

Each `.gpkg` file contains the `perimeter` and `fireline` layers with fire
boundary polygons at each timestep.

### Event ID Format

FEDS-MTBS reuses the **MTBS Event ID** as its canonical event key, so the
same identifier addresses both the MTBS burn record and the FEDS-MTBS
perimeter time series. The ID follows the pattern `{STATE}{LAT}{LON}{DATE}`:
- **STATE**: 2-letter state code (e.g., `CA`)
- **LAT**: Latitude × 1000 (5 digits, e.g., `34326` for 34.326°)
- **LON**: Longitude × 1000, unsigned (6 digits, e.g., `118481` for 118.481°)
- **DATE**: Fire start date as YYYYMMDD (e.g., `20191010`)

Example: `CA3432611848120191010` = California fire at (34.326°N, 118.481°W) starting October 10, 2019

</details>

## FIRMS / VIIRS Active Fire Dataset

Fire Radiative Power comes from NASA FIRMS VIIRS (Collection 2) active-fire
detections. When the FEDS archive is available, pre-2025 fires use the bundled FEDS
firepix CSVs and FRP is masked to the perimeter at each timestep; otherwise FRP is
streamed from FIRMS for any year and left unmasked (`perimeter_masked: false` in the
layer metadata). The FIRMS Area API archive (S-NPP) covers the full FEDS period.

<details>
<summary>Dataset details — source, streaming, and offline fallback</summary>

### Data Source

- **Provider**: NASA FIRMS (Fire Information for Resource Management System)
- **Data Access**: [Area API](https://firms.modaps.eosdis.nasa.gov/api/area/) — streamed per event
- **Sources queried**: VIIRS S-NPP and NOAA-20, each with `*_SP` (standard processing / archive) and `*_NRT` (near real-time)
- **Resolution**: 375 m

Both VIIRS platforms are queried and merged because a single platform can have
gaps — e.g. S-NPP VIIRS had a multi-day outage during the July 2024 Park Fire
(zero detections on its peak days) that NOAA-20 captured in full.

### Streaming (default, recommended)

Set the `FIRMS_MAP_KEY` environment variable (see
[Prerequisites](#prerequisites)). For each
event the pipeline requests only the bounding box and date window from the Area
API (in ≤5-day chunks, the API's per-request limit), merges all VIIRS
platform/processing sources, de-duplicates, and caches the small result under
`cache/FIRMS/<event_id>.csv` (typically a few hundred KB per event).
**No multi-GB archive download is needed.**

### Fallback: bundled archive CSVs

If `FIRMS_MAP_KEY` is not set, the pipeline reads full-archive VIIRS CSVs placed
directly in `datasets/FIRMS/` (e.g. `fire_archive_SV-C2_*.csv`,
`fire_nrt_SV-C2_*.csv`), filtering them by the event's bounds and time range.
These large files are only required for this offline fallback and can be deleted
once streaming is configured.

</details>

## ESA WorldCover Dataset

The ESA WorldCover dataset provides global land cover classification at 10m resolution based on Sentinel-1 and Sentinel-2 data.

<details>
<summary>Dataset details — source and land-cover class table</summary>

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

</details>

## Global WUI Dataset

The Global Wildland-Urban Interface (WUI) dataset maps where buildings and wildland vegetation meet or intermingle at 10m resolution globally.

<details>
<summary>Dataset details — source, WUI class table, and tile setup</summary>

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

**No manual download required.** The Global WUI data uses the EQUI7 tiling grid,
and the pipeline automatically determines which tiles a fire event needs. Any
tile not already present locally is streamed directly out of the remote continent
archive (North America) using HTTP byte-range requests — roughly **32 KB per
tile** — instead of downloading the full ~3.8 GB archive. The loader checks the
user archive (`datasets/GlobalWUI/`) first, then the cache; streamed tiles are
cached under `cache/GlobalWUI/` so repeated runs reuse them:

```
cache/GlobalWUI/
├── X0065_Y0040/
│   └── WUI.tif      # streamed + cached on first use
├── X0066_Y0040/
│   └── WUI.tif
└── ...
```

If you already have the full archive extracted into `datasets/GlobalWUI/` (or let
the wizard download it), those local tiles are used as-is and nothing is streamed.
The remote source is the SILVIS Lab geoserver (`NA.zip`); to pre-seed tiles for
offline use yourself, extract the relevant `X..._Y...` directories under
`datasets/GlobalWUI/`.

</details>

### Data Provenance & Citations

Every output layer and the underlying source dataset, with primary citation and the
exact product version / Earth Engine asset ID or service endpoint used. All Earth
Engine layers are accessed through Google Earth Engine
([Gorelick et al. 2017](https://doi.org/10.1016/j.rse.2017.06.031)).

| Output feature(s) | Source & primary citation | Version / asset ID or endpoint |
|---|---|---|
| `burn_perimeter`, `fireline`, `fireline_max_frp` | FEDS — Chen et al. 2022, *Sci. Data*, [doi:10.1038/s41597-022-01343-0](https://doi.org/10.1038/s41597-022-01343-0) | FEDS-MTBS extension (2012–2024), Zenodo [doi:10.5281/zenodo.20187962](https://doi.org/10.5281/zenodo.20187962); FEDS *Sci. Data* DOI is the algorithm reference |
| `frp_daytime`, `frp_nighttime` | NASA FIRMS — Davies et al. 2009, [doi:10.1109/TGRS.2008.2002076](https://doi.org/10.1109/TGRS.2008.2002076); VIIRS 375 m product, Schroeder et al. 2014, [doi:10.1016/j.rse.2013.12.008](https://doi.org/10.1016/j.rse.2013.12.008) | VIIRS S-NPP + NOAA-20, Collection 2; FIRMS Area API |
| *(event keys / MTBS constraint)* | MTBS — Eidenshink et al. 2007, [doi:10.4996/fireecology.0301003](https://doi.org/10.4996/fireecology.0301003) | mtbs.gov burn records |
| `elevation`, `terrain_rgb` | USGS 3DEP — U.S. Geological Survey 2015 | GEE `USGS/3DEP/1m` |
| `canopy_bulk_density`, `canopy_cover` | LANDFIRE 2.4.0 — USGS/USDA 2025 | GEE `projects/sat-io/open-datasets/landfire/FUEL/{CBD,CC}` |
| `recent_burn` | NIFC InteragencyFirePerimeterHistory *(no DOI)* | ArcGIS FeatureServer `services3.arcgis.com/T4QMspbfLg3qTGWY` |
| `r2`, `u10`, `v10` | NOAA HRRR — Dowell et al. 2022, [doi:10.1175/WAF-D-21-0151.1](https://doi.org/10.1175/WAF-D-21-0151.1); retrieved via Herbie, Blaylock 2026, [doi:10.5281/zenodo.18902673](https://doi.org/10.5281/zenodo.18902673) | HRRR v4, AWS NODD archive |
| `building_height` | GlobalBuildingAtlas — Zhu et al. 2025, *ESSD*, [doi:10.5194/essd-17-6647-2025](https://doi.org/10.5194/essd-17-6647-2025) | GEE `projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS`; dataset [doi:10.14459/2025mp1782307](https://doi.org/10.14459/2025mp1782307) |
| `landcover` | ESA WorldCover — Zanaga et al. 2022, [doi:10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221) | GEE `ESA/WorldCover/v200` (2021) |
| `lai` | Sentinel-2 LAI — Mukherjee & Chakraborty 2026, [doi:10.21203/rs.3.rs-8970245/v1](https://doi.org/10.21203/rs.3.rs-8970245/v1) | GEE `projects/tc-global-urban/assets/LAI_Grid_30deg_*` (2020) |
| `sentinel2_rgb` | Sentinel-2 — Drusch et al. 2012, [doi:10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026) | GEE `COPERNICUS/S2_SR_HARMONIZED` |
| `wui` | Global WUI — Schug et al. 2023, *Nature*, [doi:10.1038/s41586-023-06320-0](https://doi.org/10.1038/s41586-023-06320-0) | SILVIS Lab GlobalWUI (EQUI7, ca. 2020) |

## Citation

If you use FireDataForge in your research, please cite the preprint:

```bibtex
@misc{xia2026firedataforgeunifiedframeworkmultisource,
  title={FireDataForge: A Unified Framework for Multi-Source Wildfire Data Retrieval and Integration}, 
  author={Zeyu Xia and Lexie Chen and Ye Liu and Huilin Huang},
  year={2026},
  eprint={2606.21198},
  archivePrefix={arXiv},
  primaryClass={cs.CE},
  url={https://arxiv.org/abs/2606.21198}, 
}
```

The software itself ([doi:10.5281/zenodo.20744740](https://doi.org/10.5281/zenodo.20744740))
and the reproducibility bundle ([doi:10.5281/zenodo.20743743](https://doi.org/10.5281/zenodo.20743743))
have their own archival DOIs (see [`CITATION.cff`](CITATION.cff)).

## Acknowledgements & Data Attribution

This work is supported by the University of Virginia's Environmental Institute.
PNNL is operated by DOE by the Battelle Memorial Institute under contract
DE-AC05-76RL01830.

This product contains modified Copernicus Sentinel data (2017–2025), used to derive the `sentinel2_rgb` surface-reflectance imagery layer.

## License

See [LICENSE](LICENSE).
