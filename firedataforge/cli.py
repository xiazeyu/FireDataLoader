"""Command-line interface for FireDataForge."""

import argparse
import logging

from firedataforge.config import (
    ensure_setup, is_first_run, is_interactive, load_env, run_setup_wizard,
)
from firedataforge.constants import CACHE_DIR
from firedataforge.events import build_firelist, validate_projected_crs
from firedataforge.examples import fetch_examples
from firedataforge.pipeline import (
    AVAILABLE_LAYERS, LAYER_ALIASES, parse_batch_input, process_batch,
    process_single_fire,
)
from schemas import ProcessingArgs

log = logging.getLogger(__name__)


def main() -> None:
    """Command-line entry point: resolve an MTBS Event ID (or a batch) and forge it."""
    parser = argparse.ArgumentParser(
        description="FireDataForge -- unified multi-source wildfire data retrieval and integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # event_id and --batch are mutually exclusive and both optional, so --setup /
    # --build-firelist can run on their own.
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "event_id", type=str, nargs="?",
        help="MTBS Event ID to process (e.g. CA3432611848120191010)")
    input_group.add_argument(
        "--batch", type=str,
        help="A file of Event IDs (one per line) or a comma-separated list")

    parser.add_argument("--setup", action="store_true",
                        help="Run the interactive credential wizard and exit")
    parser.add_argument("--build-firelist", dest="build_firelist", action="store_true",
                        help="Download the full MTBS archive to the offline fire-list cache and exit")
    parser.add_argument("--fetch-examples", dest="fetch_examples", action="store_true",
                        help="Download examples.zip from Zenodo and unzip it at the repo root "
                             "(into datasets/FEDS25MTBS/ + events.txt), then exit")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Events processed in parallel in batch mode (1-4 recommended)")
    parser.add_argument("--layer-workers", dest="layer_workers", type=int, default=5,
                        help="Concurrent layer downloads within a single event")
    parser.add_argument("--resolution", "-r", type=int, default=30,
                        help="Target spatial resolution in meters")
    parser.add_argument("--buffer", "-b", type=int, default=600,
                        help="Margin around the fire's extent in meters")
    parser.add_argument("--aoi-mode", dest="aoi_mode", type=str, default="tight",
                        choices=["tight", "bbox"],
                        help="AOI extent: 'tight' uses the fire's true projected "
                             "perimeter; 'bbox' uses the legacy lon/lat envelope")
    parser.add_argument("--crs", "-c", type=str, default="EPSG:5070",
                        help="Target coordinate reference system")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--interpolation", "-t", type=int, default=0,
                        help="Intermediate frames to interpolate between perimeter timesteps")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR,
                        help="Root directory for all on-the-fly downloads "
                             "(HRRR, FIRMS, FEDS, firepix, WUI, fire list); each "
                             "caches under its own fixed subfolder of this root")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose (DEBUG) logging")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated subset of layers to process. Available: "
                             + ", ".join(AVAILABLE_LAYERS))

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Stand-alone maintenance commands.
    load_env()
    if args.setup:
        run_setup_wizard()
        return
    if args.build_firelist:
        build_firelist()
        return
    if args.fetch_examples:
        try:
            fetch_examples()
        except Exception as exc:
            log.error(str(exc))
            raise SystemExit(1)
        log.info(
            "Example FEDS-MTBS fires ready in datasets/FEDS25MTBS/ "
            "(event IDs listed in events.txt)")
        return

    # On the first interactive run with no event, configure and stop.
    ran_wizard = is_first_run() and is_interactive()
    ensure_setup()
    if ran_wizard and not (args.event_id or args.batch):
        return

    if not args.event_id and not args.batch:
        parser.error("an event_id or --batch is required")

    # Fail fast on an unusable target CRS, before any network resolution: the
    # grid is built in metres, so a geographic or non-metric CRS cannot work.
    try:
        validate_projected_crs(args.crs)
    except ValueError as exc:
        parser.error(str(exc))

    only_features = (
        [f.strip() for f in args.only.split(",") if f.strip()] if args.only else None
    )
    # Validate --only against the known layer names (+ the landfire/hrrr aliases)
    # so a typo fails loudly instead of silently producing an empty output dir.
    if only_features is not None:
        valid_only = set(AVAILABLE_LAYERS) | set(LAYER_ALIASES.values())
        unknown = [f for f in only_features if f not in valid_only]
        if unknown:
            only_features = [f for f in only_features if f in valid_only]
            log.warning(
                "Ignoring unknown --only layer(s): %s. Valid names: %s",
                ", ".join(unknown), ", ".join(AVAILABLE_LAYERS + ["landfire", "hrrr"]),
            )
        if not only_features:
            parser.error(
                "--only contained no recognized layer names. Valid names: "
                + ", ".join(AVAILABLE_LAYERS + ["landfire", "hrrr"]))

    processing_args = ProcessingArgs(
        resolution=args.resolution,
        buffer=args.buffer,
        aoi_mode=args.aoi_mode,
        crs=args.crs,
        output_dir=args.output_dir,
        interpolation=args.interpolation,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        only=only_features,
        layer_workers=args.layer_workers,
    )

    # Earth Engine is initialized lazily and per-event (fail-soft); no global init
    # here, so a GEE outage never blocks the FEDS/FIRMS/HRRR layers.
    if args.batch:
        event_ids = parse_batch_input(args.batch)
        if not event_ids:
            log.error("No valid event IDs found in batch input")
            return
        process_batch(event_ids, processing_args, max_workers=args.workers)
    else:
        process_single_fire(args.event_id, processing_args)
