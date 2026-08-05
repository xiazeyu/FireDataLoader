"""Fail-soft per-event and batch orchestration + task summaries."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Literal, Optional

from tqdm import tqdm

from firedataforge.config import _firms_map_key, gee_ready
from firedataforge.constants import DEFAULT_FIRE_WINDOW_DAYS, FIRELIST_CACHE_NAME
from firedataforge.events import get_fire_info, get_task_info
from firedataforge.io import save_coordinates, save_numpy
from firedataforge.sources.feds import (
    find_event_gpkg, interpolate_burn_perimeter, process_feds25mtbs,
    process_fireline, process_fireline_max_frp,
)
from firedataforge.sources.frp import process_fireline_frp_points, process_frp
from firedataforge.sources.gee import (
    download_building_height, download_eca, download_landfire,
    download_sentinel2_rgb, download_tc, download_terrain_rgb, download_usgs,
)
from firedataforge.sources.nifc import download_nifc_perimeters
from firedataforge.sources.weather import download_hrrr, write_data_gap_log
from firedataforge.sources.wui import download_globalwui
from schemas import DataLayer, ProcessingArgs

log = logging.getLogger(__name__)


GEE_LAYERS = {"elevation", "landfire", "building_height", "landcover",
              "lai", "sentinel2_rgb", "terrain_rgb"}

# Layers that cannot be produced at all without the local FEDS archive.
FEDS_ONLY_LAYERS = {"fireline"}  # burn_perimeter / fireline_max_frp handled separately

# Per-layer retry budget for transient network/service failures.
LAYER_MAX_ATTEMPTS = 3

# Some builders emit several output layers whose file names differ from the
# builder key (e.g. ``landfire`` writes canopy_bulk_density + canopy_cover,
# ``hrrr`` writes r2 + u10 + v10). Map those output names back to their builder
# key so ``--only canopy_cover`` / ``--only r2`` select the right builder, not
# only the builder key. Builder keys still work (they pass through unchanged).
LAYER_ALIASES = {
    "canopy_bulk_density": "landfire",
    "canopy_cover": "landfire",
    "r2": "hrrr",
    "u10": "hrrr",
    "v10": "hrrr",
}


class LayerUnavailable(Exception):
    """Raised by a layer builder when a required dependency is missing.

    Lets the task summary distinguish an *expected* skip (an absent dataset or
    credential) from an *unexpected* failure (a bug or a transient service error).
    """


def _normalize_layers(result: Any) -> list[DataLayer]:
    """Coerce a builder result (``DataLayer`` | list | ``None``) into a list."""
    if result is None:
        return []
    if isinstance(result, list):
        return [layer for layer in result if layer is not None]
    return [result]


def _write_task_summary(output_dir: str, event_id: str, summary: dict) -> str:
    """Write ``<output_dir>/<event_id>/task_summary.json`` and return its path."""
    event_dir = os.path.join(output_dir, event_id)
    os.makedirs(event_dir, exist_ok=True)
    path = os.path.join(event_dir, "task_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return path


def process_single_fire(
    event_id: str, args: ProcessingArgs, show_progress: bool = True
) -> dict[str, Any]:
    """Retrieve and harmonize every available layer for one fire event (fail-soft).

    Each layer is produced independently: a missing dependency (no FEDS archive,
    no Earth Engine, no FIRMS key) or a failed/maintenance-down service is recorded
    and skipped without affecting the other layers. Outputs are written under
    ``args.output_dir/<event_id>/`` as one ``.npy`` per layer plus
    ``coordinates.npy``, ``task_info.npy``, and a ``task_summary.json`` recording
    each layer's status (``ok`` / ``skipped`` / ``failed``) and reason.

    Args:
        event_id: MTBS Event ID to process.
        args: Processing configuration.
        show_progress: Show a per-layer progress bar (auto-hidden on a non-TTY).
            Disabled by :func:`process_batch`, which shows its own per-event bar.

    Returns:
        The task-summary dict (also persisted to ``task_summary.json``).
    """
    summary: dict[str, Any] = {
        "event_id": event_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "notes": [],
        "layers": {},
    }
    only = {LAYER_ALIASES.get(k, k) for k in args.only} if args.only else None

    def selected(key: str) -> bool:
        return only is None or key in only

    # --- Resolve the event + processing task (the one fatal step) ---
    try:
        fire_info = get_fire_info(
            event_id,
            firelist_cache=os.path.join(args.cache_dir, FIRELIST_CACHE_NAME),
            cache_dir=args.cache_dir)
        task_info = get_task_info(fire_info, resolution=args.resolution,
                                  buffer=args.buffer, crs=args.crs,
                                  cache_dir=args.cache_dir,
                                  aoi_mode=args.aoi_mode)
    except Exception as exc:
        log.error(f"[{event_id}] cannot resolve event: {exc}")
        summary["status"] = "error"
        summary["error"] = str(exc)
        _write_task_summary(args.output_dir, event_id, summary)
        return summary

    summary.update({
        "name": task_info.name,
        "year": task_info.year,
        "crs": task_info.crs,
        "resolution_m": task_info.resolution,
        "buffer_m": task_info.buffer,
        "aoi_mode_requested": args.aoi_mode,
        "aoi_mode_effective": task_info.aoi_mode,
        "shape": list(task_info.shape),
        "bounds": list(task_info.bounds),
        "t_start": task_info.t_start.isoformat(),
        "t_end": task_info.t_end.isoformat(),
        "t_end_estimated": task_info.t_end_estimated,
    })
    if task_info.t_end_estimated:
        summary["notes"].append(
            f"t_end is an estimate (t_start + {DEFAULT_FIRE_WINDOW_DAYS} days): no "
            "FEDS perimeter and no fire-list end date were available.")
    if task_info.aoi_mode != args.aoi_mode:
        summary["notes"].append(
            f"aoi_mode fell back to '{task_info.aoi_mode}' from '{args.aoi_mode}': "
            "this event has no FEDS perimeter geometry, so its lon/lat bounding box "
            "is the only extent available.")

    # Persist the grid descriptors shared by every layer.
    save_numpy(task_info, DataLayer(name="task_info", data=[asdict(task_info)]),
               args.output_dir)
    save_coordinates(task_info, args.output_dir)
    log.info(f"[{event_id}] {task_info.name} ({task_info.year}) | "
             f"{task_info.t_start.date()} -> {task_info.t_end.date()} | "
             f"{task_info.resolution} m | {task_info.shape} | {task_info.crs} | "
             f"aoi={task_info.aoi_mode} +{task_info.buffer} m")

    # --- Probe optional dependencies once ---
    # gee_ready() is non-interactive: it initializes Earth Engine if credentials
    # already exist and returns False otherwise, never opening a browser. Auth is
    # handled separately by the setup wizard (`--setup`).
    has_feds = find_event_gpkg(event_id, task_info.year, cache_dir=args.cache_dir) is not None
    gee_ok = gee_ready()
    if not gee_ok:
        log.warning(f"[{event_id}] Earth Engine unavailable; GEE layers will be skipped")
    summary.update({"has_feds_archive": has_feds, "earth_engine": gee_ok,
                    "firms_key": _firms_map_key() is not None})

    # --- Per-layer bookkeeping ---
    def record(key: str, status: str, reason: Optional[str] = None,
               layers: Any = None) -> None:
        entry: dict[str, Any] = {"status": status}
        if reason:
            entry["reason"] = reason
        layers = _normalize_layers(layers)
        if layers:
            entry["files"] = [f"{layer.name}.npy" for layer in layers]
            if layers[0].timestamps is not None:
                entry["n_frames"] = len(layers[0].data)
        summary["layers"][key] = entry
        if status == "failed":
            log.warning(f"[{event_id}] layer '{key}' FAILED: {reason}")
        elif status == "skipped" and reason and "deselect" not in reason:
            log.warning(f"[{event_id}] layer '{key}' skipped: {reason}")

    def save_and_record(key: str, layers: Any) -> None:
        layers = _normalize_layers(layers)
        if not layers or all(len(layer.data) == 0 for layer in layers):
            record(key, "skipped", reason="no data available")
            return
        for layer in layers:
            save_numpy(task_info, layer, args.output_dir)
        record(key, "ok", layers=layers)

    # --- Phase 1: FEDS perimeter (local; feeds masking + fireline_max_frp) ---
    perimeter: Optional[DataLayer] = None
    if has_feds and (selected("burn_perimeter") or selected("fireline_max_frp")
                     or selected("frp_daytime") or selected("frp_nighttime")):
        try:
            perimeter = process_feds25mtbs(task_info, cache_dir=args.cache_dir)
            if args.interpolation > 0:
                perimeter = interpolate_burn_perimeter(
                    perimeter, multiplier=args.interpolation)
        except Exception as exc:
            record("burn_perimeter", "failed", reason=repr(exc))
            perimeter = None
        else:
            if selected("burn_perimeter"):
                save_and_record("burn_perimeter", perimeter)
    elif selected("burn_perimeter") and not has_feds:
        record("burn_perimeter", "skipped", reason="no local FEDS archive")

    # --- Phase 2: independent layers, in parallel, fail-soft ---
    def frp_builder(time_of_day: Literal['all', 'day', 'night']) -> DataLayer:
        return process_frp(task_info,
                           perimeter_data=perimeter if has_feds else None,
                           time_interval_hours=24, time_of_day=time_of_day,
                           cache_dir=args.cache_dir)

    builders: dict[str, Callable[[], Any]] = {
        "fireline": lambda: process_fireline(task_info, cache_dir=args.cache_dir),
        "frp_daytime": lambda: frp_builder("day"),
        "frp_nighttime": lambda: frp_builder("night"),
        "elevation": lambda: download_usgs(task_info),
        "landfire": lambda: download_landfire(task_info),
        "building_height": lambda: download_building_height(task_info),
        "landcover": lambda: download_eca(task_info),
        "lai": lambda: download_tc(task_info),
        "sentinel2_rgb": lambda: download_sentinel2_rgb(task_info),
        "terrain_rgb": lambda: download_terrain_rgb(task_info),
        "wui": lambda: download_globalwui(task_info, cache_dir=args.cache_dir),
        "recent_burn": lambda: download_nifc_perimeters(task_info),
        "hrrr": lambda: download_hrrr(task_info, cache_dir=args.cache_dir),
    }

    jobs: list[str] = []
    for key in builders:
        if not selected(key):
            continue
        if key in FEDS_ONLY_LAYERS and not has_feds:
            record(key, "skipped", reason="no local FEDS archive")
            continue
        if key in GEE_LAYERS and not gee_ok:
            record(key, "skipped", reason="Earth Engine not authenticated")
            continue
        jobs.append(key)

    built: dict[str, list[DataLayer]] = {}

    def run(key: str) -> tuple[str, str, Optional[str], list[DataLayer]]:
        last: Optional[Exception] = None
        for attempt in range(LAYER_MAX_ATTEMPTS):
            try:
                return key, "ok", None, _normalize_layers(builders[key]())
            except LayerUnavailable as exc:
                return key, "skipped", str(exc), []
            except Exception as exc:  # transient failure -> retry, then give up
                last = exc
                if attempt + 1 < LAYER_MAX_ATTEMPTS:
                    log.info(f"[{event_id}] '{key}' attempt {attempt + 1} failed "
                             f"({exc}); retrying")
        return key, "failed", repr(last), []

    if jobs:
        log.info(f"[{event_id}] retrieving {len(jobs)} layer(s): {sorted(jobs)}")
        with ThreadPoolExecutor(max_workers=max(1, args.layer_workers)) as pool:
            futures = [pool.submit(run, key) for key in jobs]
            for future in tqdm(
                as_completed(futures), total=len(jobs),
                desc=f"[{event_id}] layers", unit=" layer",
                disable=None if show_progress else True, leave=False,
            ):
                key, status, reason, layers = future.result()
                built[key] = layers
                if status == "ok":
                    save_and_record(key, layers)
                    if key == "hrrr" and layers and layers[0].note.get("data_gaps"):
                        write_data_gap_log(task_info, layers[0].note["data_gaps"],
                                           args.output_dir)
                else:
                    record(key, status, reason=reason)

    # --- Phase 3: fireline_max_frp (depends on fireline + daytime FRP points) ---
    if selected("fireline_max_frp"):
        if not has_feds:
            record("fireline_max_frp", "skipped", reason="no local FEDS archive")
        else:
            fireline = next(iter(built.get("fireline") or []), None)
            try:
                if fireline is None:
                    fireline = process_fireline(task_info, cache_dir=args.cache_dir)
                # Per-pixel max raw FRP (true MW), not the mass-preserving splat, so
                # the per-segment max stays an observed radiative intensity.
                frp_points = process_fireline_frp_points(
                    task_info, perimeter if has_feds else None,
                    cache_dir=args.cache_dir)
                layer = process_fireline_max_frp(task_info, fireline, frp_points)
                save_and_record("fireline_max_frp", layer)
            except Exception as exc:
                record("fireline_max_frp", "failed", reason=repr(exc))

    # --- Finalize ---
    def by_status(status: str) -> list[str]:
        return [k for k, v in summary["layers"].items() if v["status"] == status]
    summary["counts"] = {s: len(by_status(s)) for s in ("ok", "skipped", "failed")}
    if summary["counts"]["failed"]:
        summary["status"] = "partial"
    path = _write_task_summary(args.output_dir, event_id, summary)
    log.info(f"[{event_id}] done: {summary['counts']['ok']} ok, "
             f"{summary['counts']['skipped']} skipped, "
             f"{summary['counts']['failed']} failed -> {path}")
    return summary


# Public, descriptively named alias for the single-event entry point.
forge_event = process_single_fire


# =============================================================================
# Batch Processing
# =============================================================================

def parse_batch_input(batch_input: str) -> list[str]:
    """Parse batch input which can be a file path or comma-separated event IDs.
    
    Args:
        batch_input: Either a path to a file containing event IDs (one per line)
                     or a comma-separated string of event IDs.
    
    Returns:
        List of event IDs to process.
    """
    # Check if it's a file
    if os.path.isfile(batch_input):
        log.info(f"Reading event IDs from file: {batch_input}")
        with open(batch_input, 'r') as f:
            event_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        log.info(f"Found {len(event_ids)} event IDs in file")
        return event_ids
    
    # Otherwise treat as comma-separated
    event_ids = [eid.strip() for eid in batch_input.split(',') if eid.strip()]
    log.info(f"Parsed {len(event_ids)} event IDs from input")
    return event_ids


def process_batch(
    event_ids: list[str],
    args: ProcessingArgs,
    max_workers: int = 2,
) -> dict[str, dict[str, Any]]:
    """Process multiple fire events concurrently, each independent and fail-soft.

    Events share the in-process FIRMS/firepix caches. A failure in one event never
    affects the others. Writes a ``batch_summary.json`` aggregating each event's
    per-layer outcome.

    Args:
        event_ids: Event IDs to process.
        args: Processing configuration (each event also parallelizes its layers,
            so keep ``max_workers`` moderate, e.g. 1-4).
        max_workers: Number of events processed in parallel.

    Returns:
        Mapping of event ID to its task-summary dict.
    """
    log.info(f"Batch: {len(event_ids)} event(s) across {max_workers} worker(s)")
    all_results: dict[str, dict[str, Any]] = {}

    def worker(event_id: str) -> tuple[str, dict[str, Any]]:
        try:
            # Per-event layer bars would collide with the outer events bar below,
            # so suppress them; this bar tracks whole-event progress instead.
            return event_id, process_single_fire(event_id, args, show_progress=False)
        except Exception as exc:  # defensive: process_single_fire is already fail-soft
            log.error(f"[{event_id}] fatal error: {exc}")
            return event_id, {"event_id": event_id, "status": "error", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, eid) for eid in event_ids]
        for future in tqdm(as_completed(futures), total=len(event_ids), desc="events"):
            event_id, result = future.result()
            all_results[event_id] = result

    tally = {"ok": 0, "partial": 0, "error": 0}
    for result in all_results.values():
        tally[result.get("status", "ok") if result.get("status") in tally else "ok"] += 1

    summary = {
        "total": len(event_ids),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **tally,
        "events": {
            eid: {
                "status": r.get("status"),
                "counts": r.get("counts"),
                "layers": {k: v.get("status")
                           for k, v in (r.get("layers") or {}).items()},
            }
            for eid, r in all_results.items()
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    log.info("=" * 60)
    log.info(f"Batch complete: {tally['ok']} ok, {tally['partial']} partial, "
             f"{tally['error']} error -> {summary_path}")
    log.info("=" * 60)
    return all_results


# User-facing layer names accepted by ``--only`` (output file stems). Builder
# keys that emit multiple layers are expanded to those names; ``landfire``,
# ``hrrr`` and ``recent_burn`` also remain valid via LAYER_ALIASES.
AVAILABLE_LAYERS = [
    "burn_perimeter", "fireline", "fireline_max_frp", "frp_daytime", "frp_nighttime",
    "elevation", "terrain_rgb", "canopy_bulk_density", "canopy_cover", "recent_burn",
    "building_height", "landcover", "lai", "sentinel2_rgb", "wui", "r2", "u10", "v10",
]
