"""Convert saved FireDataForge layers to GeoTIFF for QGIS/ArcGIS.

    python to_geotiff.py output/                       # every event in a tree
    python to_geotiff.py output/CA3432611848120191010  # one event
    python to_geotiff.py output/ -o gis_export         # write elsewhere

Each raster layer becomes ``<event_id>/geotiff/<name>.tif``, with one band per
frame (band descriptions are the observation timestamps). The ``.npz`` files are
left untouched and remain the canonical output -- GeoTIFF carries the pixels,
CRS, and transform, but not the full ``DataLayer`` envelope (``note``,
``categories``, provenance).

Legacy 0.1 ``.npy`` layers are read too; convert them first with
``python main.py --convert-legacy output/`` to drop the pickle dependency.
"""

import argparse
import logging
import sys

from firedataforge.geotiff import export_tree

log = logging.getLogger("to_geotiff")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FireDataForge .npz layers to GeoTIFF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path", type=str,
        help="An output tree (output/) or a single event directory")
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None,
        help="Base directory to write under (default: beside the .npz files)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose (DEBUG) logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        import rasterio  # noqa: F401
    except ImportError:
        log.error("rasterio is required for GeoTIFF export: uv sync (or pip install rasterio)")
        raise SystemExit(1)

    results = export_tree(args.path, args.output_dir)
    if not results:
        log.error(f"No event directories found under {args.path} "
                  "(an event directory contains a task_info layer)")
        raise SystemExit(1)

    total = sum(results.values())
    for event_id, count in results.items():
        print(f"{event_id}: {count} layer(s)")
    print(f"\nDone — {total} GeoTIFF(s) across {len(results)} event(s)")


if __name__ == "__main__":
    sys.exit(main())
