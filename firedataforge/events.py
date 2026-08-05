"""Event resolution: fire-list reading/building, MTBS lookup order, and the
ProcessingTask (grid + active-burning window)."""

import csv
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any, Optional

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import box
from tqdm import tqdm

from firedataforge.config import find_feds_firelist, metadata_priority
from firedataforge.constants import (
    CACHE_DIR, DEFAULT_FIRE_WINDOW_DAYS,
    DEFAULT_FIRELIST_CACHE, FEDS_DIR,
)
from firedataforge.sources.feds import (
    find_event_gpkg, get_fire_progression_dates, get_perimeter_bounds,
    get_perimeter_hull_wkt,
    index_event_gpkgs, read_perimeter_gdf,
)
from firedataforge.sources.mtbs import PROVISIONAL_IA, iter_all_events, query_mtbs
from schemas import FireEvent, ProcessingTask

log = logging.getLogger(__name__)


FIRELIST_FIELDS = ["Event_ID", "Year", "Ig_Date", "Ted", "Incid_Name", "Fire_Type",
                   "Asmnt_Type", "BurnBndAc", "IrwinID", "Map_ID",
                   "lon0", "lat0", "lon1", "lat1"]

# Parsed fire lists, memoized by (path -> (mtime, records)) to avoid re-reading
# the CSV for every event in a batch.
_firelist_memo: dict[str, tuple[float, dict[str, dict]]] = {}


def _maybe_dt(value: Any) -> Optional[datetime]:
    """Parse a value to ``datetime``, returning ``None`` for blanks/NaN/garbage."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value)
        return None if pd.isna(ts) else ts.to_pydatetime()
    except (ValueError, TypeError):
        return None


def _read_firelist_csv(path: Optional[str]) -> dict[str, dict]:
    """Read any fire-list CSV into ``{event_id: record}``, keyed by Event ID.

    Handles both the bundled example list (``...,tst,ted,lon0,lon1,lat0,lat1``)
    and the self-built cache (``...,Ig_Date,Ted,...,lon0,lat0,lon1,lat1``). Columns
    are read by name, so column order does not matter. ``t_start`` comes from
    ``Ig_Date``/``tst``; ``t_end`` from ``Ted``/``ted`` when present. The released
    FEDS-MTBS GeoPackage fire list is read separately by :func:`_read_firelist_geo`.
    """
    if not path or not os.path.exists(path):
        return {}
    mtime = os.path.getmtime(path)
    cached = _firelist_memo.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    has = set(df.columns)
    records: dict[str, dict] = {}
    for _, row in df.iterrows():
        event_id = str(row["Event_ID"])
        name = row.get("Incid_Name")
        acres = row.get("BurnBndAc")
        ig = row.get("Ig_Date") if "Ig_Date" in has else row.get("tst")
        ted = row.get("Ted") if "Ted" in has else (row.get("ted") if "ted" in has else None)
        try:
            bounds = (float(row["lon0"]), float(row["lat0"]),
                      float(row["lon1"]), float(row["lat1"]))
        except (KeyError, ValueError, TypeError):
            bounds = None
        records[event_id] = {
            "event_id": event_id,
            "name": event_id if pd.isna(name) else str(name),
            "year": int(row["Year"]),
            "acres": 0 if pd.isna(acres) else int(acres),
            "t_start": _maybe_dt(ig),
            "t_end": _maybe_dt(ted),
            "bounds": bounds,
        }
    _firelist_memo[path] = (mtime, records)
    return records


def _year_from_event_id(event_id: str) -> Optional[int]:
    """Recover the fire year from an MTBS Event ID (``...YYYYMMDD``)."""
    try:
        year = int(event_id[-8:-4])
    except (ValueError, IndexError):
        return None
    return year if 1900 <= year <= 2100 else None


def _read_firelist_geo(path: str) -> dict[str, dict]:
    """Read the released FEDS-MTBS fire list (a GeoPackage of MTBS final perimeters).

    The published ``fireslist_FEDS25MTBS_2012-2024.geojson`` is a GeoPackage whose
    attributes are the MTBS metadata (``Event_ID``, ``Incid_Name``, ``BurnBndAc``,
    ``Ig_Date``) and whose geometry is each fire's final MTBS perimeter. Unlike the
    example CSV it carries no FEDS ``tst``/``ted`` and no explicit bbox columns, so:

    * ``bounds`` is each feature's perimeter envelope, read with
      :func:`pyogrio.read_bounds` (no geometry load -- fast and low-memory). The
      list is in EPSG:4269 (NAD83), whose <2 m offset from EPSG:4326 is negligible
      next to the AOI buffer, so the values are used directly as lon/lat.
    * ``t_start`` is ``Ig_Date`` and ``year`` is derived from it (or the Event ID).
    * ``t_end`` is left ``None`` -- it is supplied by the per-fire GeoPackage
      progression (or the fallback window) in :func:`_resolve_fire_event`.
    """
    mtime = os.path.getmtime(path)
    cached = _firelist_memo.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    import warnings

    import pyogrio

    # The released list is named ``.geojson`` but is actually a GeoPackage, which
    # makes GDAL emit a benign "non conformant file extension" RuntimeWarning. Read
    # every attribute field (no geometry -- fast) so a renamed optional column never
    # breaks the read; only ``Event_ID`` is required.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non conformant file extension.*")
        attrs = pyogrio.read_dataframe(path, read_geometry=False, fid_as_index=True)
        fids, bounds = pyogrio.read_bounds(path)  # bounds[(xmin,ymin,xmax,ymax), n]
    if "Event_ID" not in attrs.columns:
        log.warning(f"{path} has no Event_ID column; ignoring this fire list")
        _firelist_memo[path] = (mtime, {})
        return {}
    bbox_by_fid = {
        int(f): (float(bounds[0, i]), float(bounds[1, i]),
                 float(bounds[2, i]), float(bounds[3, i]))
        for i, f in enumerate(fids)
    }

    records: dict[str, dict] = {}
    for fid, row in attrs.iterrows():
        event_id = str(row["Event_ID"])
        ig = _maybe_dt(row.get("Ig_Date"))
        year = (ig.year if ig else None) or _year_from_event_id(event_id)
        if year is None:
            continue  # cannot place the fire in time -> skip
        name = row.get("Incid_Name")
        acres = row.get("BurnBndAc")
        records[event_id] = {
            "event_id": event_id,
            "name": event_id if (name is None or pd.isna(name)) else str(name),
            "year": int(year),
            "acres": 0 if (acres is None or pd.isna(acres)) else int(acres),
            "t_start": ig,
            "t_end": None,
            "bounds": bbox_by_fid.get(int(fid)),
        }
    _firelist_memo[path] = (mtime, records)
    return records


def _read_firelist(path: Optional[str]) -> dict[str, dict]:
    """Read a fire list into ``{event_id: record}``, dispatching on file type.

    GeoPackage/GeoJSON lists (the released FEDS-MTBS summary) go through
    :func:`_read_firelist_geo`; everything else is parsed as CSV.
    """
    if not path or not os.path.exists(path):
        return {}
    if path.lower().endswith((".geojson", ".gpkg")):
        return _read_firelist_geo(path)
    return _read_firelist_csv(path)


def read_feds_firelist(feds_dir: str = FEDS_DIR) -> dict[str, dict]:
    """Read the local FEDS-MTBS fire list, or ``{}`` if none is present.

    Reads whichever list :func:`find_feds_firelist` locates in ``feds_dir`` -- the
    bundled ``fireslist_examples.csv`` (preferred when present) or the released
    ``fireslist_FEDS25MTBS_2012-2024.geojson`` GeoPackage.
    """
    return _read_firelist(find_feds_firelist(feds_dir))


def _read_fire_cache(path: Optional[str] = DEFAULT_FIRELIST_CACHE) -> dict[str, dict]:
    """Read the self-built offline MTBS fire-list cache, or ``{}`` if absent."""
    return _read_firelist_csv(path)


def _resolve_fire_event(
    event_id: str, record: dict, cache_dir: str = CACHE_DIR
) -> FireEvent:
    """Build a :class:`FireEvent`, applying the bounds and ``t_end`` priorities.

    bounds: ``FEDS GeoPackage perimeter > fire-list bbox > MTBS bbox``.
    t_end:  ``FEDS GeoPackage progression > record's explicit end date > fallback window``.

    A local FEDS GeoPackage, when present, wins outright: its perimeter extent
    gives the bounds and its progression gives the active-burning window
    (``t_start``/``t_end``) -- the growth period, which is tighter than any
    fire-list end date (FEDS keeps tracking the scar long after it stops growing)
    and keeps the hourly HRRR / FRP series from spanning that post-growth tail.
    Without a GeoPackage, an explicit end date from the record is used when present
    (the example CSV's ``ted``); otherwise -- including the released FEDS-MTBS fire
    list and the MTBS service, neither of which carries an end date -- ``t_end``
    falls back to ``t_start`` + :data:`DEFAULT_FIRE_WINDOW_DAYS`, a blind estimate
    that :func:`get_task_info` flags so downstream consumers know it is approximate.
    The released fire list's ``bounds`` is the MTBS final-perimeter envelope.
    """
    year = int(record["year"])
    t_start = record.get("t_start") or datetime(year, 1, 1)
    t_end = record.get("t_end")
    bounds = record["bounds"]
    aoi_wkt = None

    gpkg_path = find_event_gpkg(event_id, year, cache_dir=cache_dir)
    if gpkg_path is not None:
        # A local FEDS perimeter exists -> bound the window to the actual
        # perimeter-growth period. This is tighter than any fire-list end date,
        # which marks when FEDS stopped *tracking* the scar (often months after it
        # stopped growing) and would otherwise stretch the hourly HRRR / FRP series
        # across a long post-growth tail. The perimeter extent likewise supersedes
        # whatever bbox the metadata source carried. Read the perimeter layer once
        # and share it across both resolvers (resolving the path once, too).
        perimeter_gdf = read_perimeter_gdf(gpkg_path)
        t_start, t_end = get_fire_progression_dates(
            event_id, year, gpkg_path=gpkg_path, perimeter_gdf=perimeter_gdf)
        bounds = get_perimeter_bounds(
            event_id, year, gpkg_path=gpkg_path, perimeter_gdf=perimeter_gdf) or bounds
        aoi_wkt = get_perimeter_hull_wkt(
            event_id, year, gpkg_path=gpkg_path, perimeter_gdf=perimeter_gdf)
    elif t_end is None:
        # No local GeoPackage and no explicit end date -> blind fallback window.
        t_end = t_start + timedelta(days=DEFAULT_FIRE_WINDOW_DAYS)
        log.warning(
            f"No end date for {event_id} (no FEDS perimeter GeoPackage and no "
            f"fire-list end date); estimating t_end = t_start + "
            f"{DEFAULT_FIRE_WINDOW_DAYS} days ({t_start.date()} -> "
            f"{t_end.date()}). FEDS-derived perimeter/fireline layers will be skipped."
        )

    return FireEvent(
        event_id=event_id,
        name=str(record["name"]),
        year=year,
        acres_burned=int(record["acres"]),
        t_start=t_start,
        t_end=t_end,
        bounds=bounds,
        aoi_wkt=aoi_wkt,
    )


def _firelist_row(record: dict) -> dict:
    """Flatten a metadata record into a fire-list cache CSV row."""
    minx, miny, maxx, maxy = record["bounds"]
    ig = record.get("t_start")
    ted = record.get("t_end")
    return {
        "Event_ID": record["event_id"],
        "Year": record["year"],
        "Ig_Date": ig.strftime("%Y-%m-%d") if ig else "",
        "Ted": ted.strftime("%Y-%m-%d %H:%M:%S") if ted else "",
        "Incid_Name": record.get("name", ""),
        "Fire_Type": record.get("fire_type", ""),
        "Asmnt_Type": record.get("asmnt_type", ""),
        "BurnBndAc": record.get("acres", 0),
        "IrwinID": record.get("irwinid", ""),
        "Map_ID": record.get("map_id", ""),
        "lon0": minx, "lat0": miny, "lon1": maxx, "lat1": maxy,
    }


def _append_to_firelist_cache(
    event_id: str, record: dict, path: str = DEFAULT_FIRELIST_CACHE
) -> None:
    """Append a single resolved event to the offline fire-list cache.

    Creates the file (with header) on first write and skips events already
    present, so the cache grows incrementally as new IDs are resolved live.
    """
    if record.get("bounds") is None:
        return
    if event_id in _read_firelist_csv(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIRELIST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(_firelist_row(record))
    _firelist_memo.pop(path, None)  # invalidate memo


def build_firelist(
    out_path: str = DEFAULT_FIRELIST_CACHE,
    year_min: Optional[int] = None,
) -> str:
    """Download the full MTBS fire list (+ Provisional IA) to ``out_path``.

    Pages through the entire MTBS Burned Area Boundaries archive (~30k fires,
    ~30 s) and writes the offline cache used to resolve any event ID without a
    network round-trip. End dates (``Ted``) are filled from local FEDS
    GeoPackages where present, otherwise left blank.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    seen: set[str] = set()
    count = 0
    # The per-event "Fire progression dates" INFO log (from the FEDS module) is
    # useful for a single event but floods the console across the ~30k fires
    # scanned here; quiet that module to WARNING for the bulk build and restore
    # it afterward, so single-event runs still report their window.
    feds_log = logging.getLogger("firedataforge.sources.feds")
    prev_level = feds_log.level
    feds_log.setLevel(logging.WARNING)
    # This scans the whole MTBS archive (~30k fires) and only checks which gpkgs
    # exist locally; suppress on-demand fetching so it never tries to download
    # thousands of fires from Zenodo.
    from firedataforge.remote_archive import lazy_enabled, set_lazy_enabled
    prev_lazy = lazy_enabled()
    set_lazy_enabled(False)
    # find_event_gpkg() rescans the whole archive on every call; doing that per
    # fire across ~30k events walks the tree tens of thousands of times (minutes
    # of pure directory stat'ing). Index every local GeoPackage once up front and
    # resolve each event with an O(1) lookup instead.
    gpkg_index = index_event_gpkgs()
    try:
        with open(out_path, "w", newline="") as fh, tqdm(
            desc="MTBS fire list", unit=" fires", disable=None, leave=False
        ) as bar:
            writer = csv.DictWriter(fh, fieldnames=FIRELIST_FIELDS)
            writer.writeheader()
            for record in iter_all_events(year_min=year_min):
                event_id = record["event_id"]
                if event_id in seen:
                    continue
                seen.add(event_id)
                gpkg = gpkg_index.get(event_id)
                if gpkg and record.get("t_end") is None:
                    try:
                        _, record["t_end"] = get_fire_progression_dates(
                            event_id, int(record["year"]), gpkg_path=gpkg)
                    except Exception:
                        pass
                writer.writerow(_firelist_row(record))
                count += 1
                bar.update(1)
                # The live bar already shows the running count; on a non-TTY (where
                # the bar is auto-disabled) fall back to the periodic INFO log.
                if bar.disable and count % 2000 == 0:
                    log.info(f"  {count} events fetched...")
            for event_id, meta in PROVISIONAL_IA.items():
                if event_id in seen or (year_min and meta["year"] < year_min):
                    continue
                writer.writerow(_firelist_row({"event_id": event_id, **meta}))
                seen.add(event_id)
                count += 1
                bar.update(1)
    finally:
        feds_log.setLevel(prev_level)
        set_lazy_enabled(prev_lazy)
    log.info(f"Wrote {count} events to {out_path}")
    _firelist_memo.pop(out_path, None)
    return out_path


def get_fire_info(
    event_id: str,
    firelist_cache: Optional[str] = DEFAULT_FIRELIST_CACHE,
    use_mtbs_api: bool = True,
    update_cache: bool = True,
    cache_dir: str = CACHE_DIR,
) -> FireEvent:
    """Resolve fire-event metadata for an MTBS Event ID.

    Overall source priority is ``FEDS GeoPackage > FEDS-MTBS fire list > MTBS fire
    list > MTBS online``. The GeoPackage ranks first but carries no fire *name*, so
    the name/acreage come from the first fire-list/online source that has the event
    (FEDS-MTBS fire list, then the offline MTBS list, then the live service), while
    the GeoPackage -- when present -- supplies the bounds and the active-burning
    window, refined in :func:`_resolve_fire_event`.

    The name/acreage chain (sources 1-3 below) is what the setup wizard's step
    [5/5] configures: :func:`~firedataforge.config.metadata_priority` reads the
    ticked sources, in the order they were ticked, from
    ``FIREDATAFORGE_METADATA_PRIORITY`` in ``.env``. Untick a source and it is
    skipped entirely; the live lookup is always kept as the last resort. With no
    setting the documented default order below applies. The GeoPackage-driven
    refinements are unaffected:

    * **bounds:** ``FEDS GeoPackage perimeter > fire-list bbox > MTBS bbox``.
    * **t_end:** ``FEDS GeoPackage progression > example-list ted > fallback window``.

    A local FEDS GeoPackage, when present, takes precedence for both the bounds
    and the window: its perimeter extent supersedes any bbox and its progression
    gives the perimeter-growth window (tighter than any fire-list end date).
    Otherwise the fire list supplies the bbox and, for the example CSV, an explicit
    end date; the released GeoPackage list, the offline cache, and the live MTBS
    service carry a burn-boundary bbox but no end date, so ``t_end`` falls back to
    the estimated window. If none of the lists has the event but a GeoPackage does,
    the run still proceeds with the Event ID as the name (source 4 below).

    1. **FEDS-MTBS fire list** -- shipped with the local FEDS archive (from Zenodo):
       the bundled ``fireslist_examples.csv`` (FEDS perimeter bbox plus both ``tst``
       and ``ted``; preferred when present) or the released
       ``fireslist_FEDS25MTBS_2012-2024.geojson`` GeoPackage (MTBS final-perimeter
       bbox + ``Ig_Date``); no network needed.
    2. **Offline MTBS cache** -- the self-built ``mtbs_firelist.csv`` grown from
       prior live lookups (and optionally pre-built with ``--build-firelist``);
       carries the MTBS burn-boundary bbox.
    3. **Live MTBS service** (+ Provisional IA supplement) -- queried for the one
       event and appended to the offline cache for next time.
    4. **FEDS GeoPackage only** -- if none of the above has the event but a local
       (or fetchable) ``<event_id>.gpkg`` exists, the run proceeds with the Event
       ID as the name and the GeoPackage's bounds/window.

    Args:
        event_id: MTBS Event ID (e.g. ``CA3432611848120191010``).
        firelist_cache: Path to the offline cache (read + grown). ``None`` disables it.
        use_mtbs_api: Set ``False`` to skip the network entirely.
        update_cache: Append newly resolved events to ``firelist_cache``.
        cache_dir: Cache root for an on-demand FEDS GeoPackage fetch (under its
            fixed ``FEDS25MTBS`` subfolder) when resolving the bounds/window.

    Returns:
        FireEvent with name, year, acres, start/end, and lon/lat bounds.

    Raises:
        RuntimeError: If the event cannot be resolved from any source.
    """
    # Sources 1-3, in the order configured by the setup wizard's step [5/5].
    for source in metadata_priority():
        if source == "feds":
            # 1. Local FEDS-MTBS fire list (released GeoPackage or example CSV).
            record = read_feds_firelist().get(event_id)
            if record is not None and record.get("bounds") is not None:
                log.info(f"Resolved {event_id} from the FEDS25MTBS fire list")
                return _resolve_fire_event(event_id, record, cache_dir=cache_dir)

        elif source == "mtbs":
            # 2. Self-built offline MTBS cache (mtbs_firelist.csv).
            record = _read_fire_cache(firelist_cache).get(event_id)
            if record is not None and record.get("bounds") is not None:
                log.info(
                    f"Resolved {event_id} from the offline fire-list cache "
                    f"({firelist_cache})")
                return _resolve_fire_event(event_id, record, cache_dir=cache_dir)

        elif source == "live" and use_mtbs_api:
            # 3. Live MTBS service (+ Provisional IA); cache it for next time.
            record = query_mtbs(event_id)
            if record is not None:
                log.info(
                    f"Resolved {event_id} from the live MTBS / Provisional IA service")
                if update_cache and firelist_cache:
                    try:
                        _append_to_firelist_cache(event_id, record, firelist_cache)
                    except Exception as exc:  # pragma: no cover - best effort
                        log.warning(
                            f"Could not append {event_id} to {firelist_cache}: {exc}")
                return _resolve_fire_event(event_id, record, cache_dir=cache_dir)

    # 4. No fire-list/online metadata, but a local (or fetchable) FEDS GeoPackage
    #    can still drive the run: it supplies the bounds and the active-burning
    #    window. The gpkg carries no fire name, so the name defaults to the Event
    #    ID and the year is taken from the ID.
    year = _year_from_event_id(event_id)
    if year is not None and find_event_gpkg(event_id, year, cache_dir=cache_dir) is not None:
        log.info(
            f"Resolved {event_id} from its FEDS GeoPackage (no fire-list metadata; "
            f"name defaults to the Event ID)")
        record = {
            "event_id": event_id, "name": event_id, "year": year,
            "acres": 0, "t_start": None, "t_end": None, "bounds": None,
        }
        return _resolve_fire_event(event_id, record, cache_dir=cache_dir)

    # Not found anywhere.
    raise RuntimeError(
        f"Cannot resolve {event_id}: none of the enabled metadata sources "
        f"({' > '.join(metadata_priority())}) has it -- FEDS-MTBS fire list, offline "
        f"MTBS cache ({firelist_cache}), live MTBS service -- and no FEDS GeoPackage "
        f"is available. Re-run `python main.py --setup` to enable more sources, build "
        f"the offline list with `python main.py --build-firelist`, or verify the "
        f"Event ID at https://www.mtbs.gov/."
    )


def validate_projected_crs(crs: str) -> None:
    """Raise ``ValueError`` unless ``crs`` is a projected, metre-based CRS.

    ``resolution`` and ``buffer`` are in **metres** and the grid is built by
    snapping the projected bounds to whole multiples of the resolution, so a
    geographic CRS whose units are degrees (e.g. ``EPSG:4326``) or a projected
    CRS in non-metric units (e.g. US-survey-foot State Plane) produces a
    nonsensical grid and downstream out-of-coverage errors. Validating here lets
    a bad ``--crs`` fail fast with a clear message instead of a cryptic failure
    deep inside a layer builder.

    Args:
        crs: Target coordinate reference system (any pyproj-parseable form).

    Raises:
        ValueError: If ``crs`` is unparseable, geographic/non-projected, or
            projected but not in metres.
    """
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        crs_obj = CRS.from_user_input(crs)
    except CRSError as exc:
        raise ValueError(f"Unrecognized target CRS {crs!r}: {exc}") from exc

    hint = ("Use a projected, metre-based CRS -- the default EPSG:5070 "
            "(CONUS Albers), a UTM zone, or similar.")
    if not crs_obj.is_projected:
        kind = "geographic (degree-based)" if crs_obj.is_geographic else "not projected"
        raise ValueError(
            f"Target CRS {crs!r} is {kind}; FireDataForge needs a projected, "
            f"metre-based CRS because --resolution and --buffer are in metres "
            f"(the grid snaps the projected bounds to whole multiples of the "
            f"resolution). {hint}"
        )
    units = {(ax.unit_name or "").lower() for ax in crs_obj.axis_info}
    if not units & {"metre", "meter"}:
        listed = ", ".join(sorted(u for u in units if u)) or "unknown"
        raise ValueError(
            f"Target CRS {crs!r} is projected but its axis units are {listed}, "
            f"not metres; --resolution and --buffer are in metres. {hint}"
        )


def get_task_info(
    fire_info: FireEvent,
    resolution: int = 30,
    buffer: int = 600,
    crs: str = "EPSG:5070",
    cache_dir: str = CACHE_DIR,
    aoi_mode: str = "tight",
) -> ProcessingTask:
    """Create a processing task configuration from fire event information.

    Projects the fire's extent into the target CRS, adds a buffer, and calculates
    the output grid dimensions. The active-burning window (``t_start``, ``t_end``)
    and the extent are resolved upstream in :func:`get_fire_info` (which applies
    the GeoPackage > FEDS-MTBS fire list > MTBS fire list > MTBS online priority),
    so this function consumes them as-is and only re-derives whether ``t_end`` is
    an estimate.

    Args:
        fire_info: Fire event information.
        resolution: Target spatial resolution in meters.
        buffer: Margin in meters to clear around the fire's extent.
        crs: Target coordinate reference system.
        cache_dir: Cache root for an on-demand FEDS GeoPackage fetch (under its
            fixed ``FEDS25MTBS`` subfolder) used to flag whether ``t_end`` is an
            estimate.
        aoi_mode: ``"tight"`` or ``"bbox"``.

    Returns:
        ProcessingTask object defining the processing parameters.

    Raises:
        ValueError: If ``crs`` is not a projected, metre-based CRS (see
            :func:`validate_projected_crs`), or ``aoi_mode`` is not a known mode.
    """
    validate_projected_crs(crs)
    if aoi_mode not in ("tight", "bbox"):
        raise ValueError(
            f"Unknown aoi_mode {aoi_mode!r}; expected 'tight' (grid built from the "
            f"fire's true projected perimeter) or 'bbox' (legacy lon/lat envelope).")

    t_start, t_end = fire_info.t_start, fire_info.t_end

    # t_end is observation-derived in every case except the blind fallback window,
    # which get_fire_info sets to exactly t_start + DEFAULT_FIRE_WINDOW_DAYS when
    # no FEDS perimeter GeoPackage and no fire-list 'ted' exist. Recognise that
    # case (no GeoPackage + that exact span) so consumers know t_end is approximate
    # and FEDS-derived perimeter/fireline layers will be skipped.
    t_end_estimated = (
        find_event_gpkg(fire_info.event_id, fire_info.year, cache_dir=cache_dir) is None
        and t_end == t_start + timedelta(days=DEFAULT_FIRE_WINDOW_DAYS)
    )

    use_tight = aoi_mode == "tight" and fire_info.aoi_wkt is not None
    if use_tight:
        # Project the perimeter's own geometry, then take its extent. Expanding by
        # ``buffer`` arithmetically (rather than via a geometric buffer) is exact,
        # so every side clears the fire by exactly that margin before snapping.
        aoi = gpd.GeoSeries([shapely_wkt.loads(fire_info.aoi_wkt)],
                            crs=fire_info.crs).to_crs(crs)
        p_minx, p_miny, p_maxx, p_maxy = aoi.total_bounds
        t_minx, t_miny = p_minx - buffer, p_miny - buffer
        t_maxx, t_maxy = p_maxx + buffer, p_maxy + buffer
    else:
        if aoi_mode == "tight":
            log.info(
                f"No perimeter geometry for {fire_info.event_id}; building the grid "
                f"from its lon/lat bounding box instead (aoi_mode=bbox). The box is "
                f"all that is known about this fire's extent.")
        # LEGACY PATH
        minx, miny, maxx, maxy = fire_info.bounds
        bbox_poly = box(minx, miny, maxx, maxy)
        bounds_gs = gpd.GeoSeries([bbox_poly], crs=fire_info.crs)
        bounds_proj = bounds_gs.to_crs(crs)

        bounds_proj = bounds_proj.buffer(buffer)

        t_minx, t_miny, t_maxx, t_maxy = bounds_proj.total_bounds

    target_bounds = (
        math.floor(t_minx / resolution) * resolution,
        math.floor(t_miny / resolution) * resolution,
        math.ceil(t_maxx / resolution) * resolution,
        math.ceil(t_maxy / resolution) * resolution
    )

    width = max(1, int((target_bounds[2] - target_bounds[0]) / resolution))
    height = max(1, int((target_bounds[3] - target_bounds[1]) / resolution))

    return ProcessingTask(
        event_id=fire_info.event_id,
        name=fire_info.name,
        year=fire_info.year,
        t_start=t_start,
        t_end=t_end,
        resolution=resolution,
        bounds=target_bounds,
        shape=(height, width),
        crs=crs,
        t_end_estimated=t_end_estimated,
        buffer=buffer,
        aoi_mode="tight" if use_tight else "bbox",
    )
