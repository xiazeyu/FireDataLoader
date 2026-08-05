"""FireDataForge: a unified framework for multi-source wildfire data retrieval and integration.

The public API is re-exported here so consumers can ``from firedataforge import
forge_event`` (or ``import firedataforge as fdf``) without reaching into submodules.
See ``firedataforge.cli`` for the command-line entry point.
"""

from firedataforge.config import (
    ensure_ca_bundle, ensure_setup, feds_available, gee_ready, load_env,
    run_setup_wizard,
)
from firedataforge.constants import DEFAULT_FIRE_WINDOW_DAYS
from firedataforge.events import (
    build_firelist, get_fire_info, get_task_info, read_feds_firelist,
    validate_projected_crs,
)
from firedataforge.examples import fetch_examples
from firedataforge.geotiff import export_event_dir, export_layer, export_tree
from firedataforge.io import (
    DATA_EXT, convert_legacy_dir, layer_files, load_numpy, resolve_path,
    save_coordinates, save_numpy,
)
from firedataforge.pipeline import (
    forge_event, parse_batch_input, process_batch, process_single_fire,
)
from firedataforge.sources.feds import find_event_gpkg
from schemas import (
    DataLayer, FireEvent, GeoReference, ProcessingArgs, ProcessingTask,
)

# Load persisted .env on import (the real environment still takes precedence),
# then make sure HTTPS calls have a usable CA bundle (some HPC Pythons ship none).
load_env()
ensure_ca_bundle()

__all__ = [
    "forge_event", "process_single_fire", "process_batch", "parse_batch_input",
    "get_fire_info", "get_task_info", "validate_projected_crs",
    "find_event_gpkg", "read_feds_firelist",
    "build_firelist", "fetch_examples", "load_numpy", "save_numpy", "save_coordinates",
    "layer_files", "resolve_path", "convert_legacy_dir", "DATA_EXT",
    "export_layer", "export_event_dir", "export_tree",
    "gee_ready", "feds_available", "run_setup_wizard", "ensure_setup", "load_env",
    "ensure_ca_bundle",
    "FireEvent", "ProcessingTask", "ProcessingArgs", "DataLayer", "GeoReference",
    "DEFAULT_FIRE_WINDOW_DAYS",
]
