"""FireDataForge command-line entry point and backward-compatible import surface.

    python main.py <event_id> [options]      # one event
    python main.py --batch events.txt        # many events
    python main.py --setup                   # credential wizard
    python main.py --build-firelist          # cache the full MTBS archive offline
    python main.py --fetch-examples          # download the example fires from Zenodo
"""

from firedataforge import *  # noqa: F401,F403  (re-export the public API)
from firedataforge import (  # noqa: F401  (explicit names for `import main` users)
    DATA_EXT, build_firelist, convert_legacy_dir, feds_available, fetch_examples,
    find_event_gpkg, forge_event, gee_ready, get_fire_info, get_task_info,
    layer_files, load_numpy, parse_batch_input, process_batch, process_single_fire,
    resolve_path, save_coordinates, save_numpy,
)
from firedataforge.cli import main

if __name__ == "__main__":
    main()
