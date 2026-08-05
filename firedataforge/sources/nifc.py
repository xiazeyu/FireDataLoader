"""NIFC InteragencyFirePerimeterHistory: most-recent burn year per pixel."""

import json
import logging

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_origin

from firedataforge.schemas import DataLayer, ProcessingTask

log = logging.getLogger(__name__)


NIFC_IFPH_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query"
)

# The All-Years view lags badly for the current decade: as of mid-2026 it holds
# ~3,000 perimeters/yr through 2019 but only ~30-150/yr for 2020-2024, while the
# per-decade read-only layer (same schema) carries the missing fires. Lookback
# windows that touch the 2020s must therefore sweep this layer too.
NIFC_IFPH_2020S_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "InterAgencyFirePerimeterHistory_2020s_Read_Only/FeatureServer/0/query"
)


def _query_ifph(base_url: str, where: str, bbox_4326: str) -> list:
    """Page through one IFPH FeatureServer query, returning all GeoJSON features.

    The service caps each response at maxRecordCount (2000) and flags
    truncation via properties.exceededTransferLimit in GeoJSON output, so page
    through with resultOffset until a short/untruncated page comes back.
    Without this, a heavily-burned AOI would silently lose every perimeter past
    the cap (cf. the paged query in sources/mtbs.py).
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    page_size = 2000
    feats: list = []
    offset = 0
    while True:
        params = {
            "where": where,
            "geometry": bbox_4326,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "FIRE_YEAR_INT,INCIDENT,GIS_ACRES,SOURCE,DATE_CUR",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "returnGeometry": "true",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        log.debug(f"NIFC IFPH query URL (offset={offset}): {url}")

        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"NIFC IFPH query failed: {e}") from e

        page = payload.get("features", []) or []
        feats.extend(page)
        # GeoJSON nests the flag under "properties" (f=json puts it top-level).
        truncated = bool((payload.get("properties") or {}).get("exceededTransferLimit"))
        # Stop on an empty page (guards against an infinite loop) or a short page
        # the service did not flag as truncated.
        if not page or (len(page) < page_size and not truncated):
            break
        offset += len(page)
    return feats


def download_nifc_perimeters(
    task_info: ProcessingTask,
    lookback_years: int = 5,
) -> DataLayer:
    """Download NIFC InteragencyFirePerimeterHistory perimeters that intersect
    the AOI and burned within the `lookback_years` years before the current
    fire's start, then rasterize to a most-recent-burn-year raster.

    Fills the recent-fires gap left by LANDFIRE Annual Disturbance (1–2 yr
    MTBS-fed lag) and HDIST (truncated ~2022). NIFC IFPH consolidates final
    perimeters from CALFIRE/USFS/BLM/NPS and updates roughly monthly, so any
    fire that ended before ``task_info.t_start`` is captured here even if
    LANDFIRE has not yet ingested it.

    Queries the All-Years view and, when the lookback window touches the 2020s,
    also the ``_2020s_Read_Only`` decade layer -- the All-Years view is nearly
    empty for 2020+ (see :data:`NIFC_IFPH_2020S_URL`), so without the decade
    sweep a post-2019 event would see no recent burns at all. Rows returned by
    both services (or duplicated between agency and WFIGS copies) dedupe by
    their attribute tuple; genuinely distinct copies of the same incident are
    kept, which is harmless -- rasterization keeps one year per pixel.

    Args:
        task_info: Task configuration with bounds, CRS, and event time range.
        lookback_years: Number of years before ``task_info.year`` to include.
            Same-year fires are also included if their ``DATE_CUR`` is before
            ``task_info.t_start`` (so a Dec-2024 burn gets picked up when the
            simulated fire starts Jan-2025, but a concurrent fire does not).

    Returns:
        DataLayer named ``recent_burn`` containing one float32 raster of shape
        ``task_info.shape``. Pixel values are the most-recent calendar year (e.g.
        ``2024.0``) that the pixel was inside a fire perimeter; pixels never burned
        in the window are ``NaN``. The lookback length is recorded in ``note``.
    """
    from shapely.geometry import shape

    log.info(
        f"Downloading NIFC IFPH recent-burn perimeters for event_id: "
        f"{task_info.event_id} (lookback={lookback_years} yr)"
    )

    year = task_info.year
    year_min = year - lookback_years
    # DATE_CUR is a string field in NIFC IFPH (YYYYMMDD), so the SQL
    # comparison must also be string-typed.
    tstart_yyyymmdd = task_info.t_start.strftime("%Y%m%d")
    layer_name = "recent_burn"

    # AOI bbox in EPSG:4326 (NIFC service expects WGS84).
    transformer = Transformer.from_crs(task_info.crs, "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = task_info.bounds
    lons, lats = transformer.transform(
        [minx, maxx, minx, maxx],
        [miny, miny, maxy, maxy],
    )
    bbox_4326 = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"

    # Prior-year fires unconditionally; same-year fires only if contained
    # before this event's t_start (so we don't lump in concurrent fires).
    where = (
        f"(FIRE_YEAR_INT >= {year_min} AND FIRE_YEAR_INT < {year}) OR "
        f"(FIRE_YEAR_INT = {year} AND DATE_CUR IS NOT NULL "
        f"AND DATE_CUR < '{tstart_yyyymmdd}')"
    )
    urls = [NIFC_IFPH_URL]
    if year >= 2020 and year_min <= 2029:
        urls.append(NIFC_IFPH_2020S_URL)

    # The same underlying record can come back from both services; dedupe on the
    # requested attribute tuple so it is counted once. Distinct copies of one
    # incident (e.g. a CALFIRE and a WFIGS perimeter with different DATE_CUR)
    # are intentionally kept.
    feats: list = []
    seen_rows: set = set()
    for base_url in urls:
        for feat in _query_ifph(base_url, where, bbox_4326):
            props = feat.get("properties") or {}
            key = (
                props.get("FIRE_YEAR_INT"),
                (props.get("INCIDENT") or "").strip().upper(),
                props.get("SOURCE"),
                props.get("DATE_CUR"),
                props.get("GIS_ACRES"),
            )
            if key in seen_rows:
                continue
            seen_rows.add(key)
            feats.append(feat)

    log.info(
        f"NIFC IFPH returned {len(feats)} unique perimeter(s) intersecting AOI "
        f"for years {year_min}–{year} across {len(urls)} service(s)"
    )

    # Output grid (matches process_feds25mtbs convention).
    t_minx, _, _, t_maxy = task_info.bounds
    transform = from_origin(t_minx, t_maxy, task_info.resolution, task_info.resolution)
    out_arr = np.full(task_info.shape, np.nan, dtype=np.float32)

    records: list[dict] = []
    for feat in feats:
        geom = feat.get("geometry")
        props = feat.get("properties") or {}
        yr_val = props.get("FIRE_YEAR_INT")
        if geom is None or yr_val is None:
            continue
        try:
            records.append({
                "geometry": shape(geom),
                "FIRE_YEAR_INT": int(yr_val),
                "INCIDENT": (props.get("INCIDENT") or "").strip(),
                "DATE_CUR": props.get("DATE_CUR"),
                "GIS_ACRES": props.get("GIS_ACRES"),
            })
        except Exception as e:
            log.warning(f"Skipping NIFC feature with bad geometry: {e}")

    incidents_summary: dict[int, list[str]] = {}
    if records:
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(task_info.crs)

        # Sort ascending by year so later (more-recent) burns overwrite earlier
        # burns under rasterio's default 'replace' merge — the final pixel value
        # is the most-recent burn year.
        gdf = gdf.sort_values("FIRE_YEAR_INT", ascending=True).reset_index(drop=True)

        shapes = [
            (g, float(y)) for g, y in zip(gdf.geometry, gdf["FIRE_YEAR_INT"])
        ]
        raster = features.rasterize(
            shapes=shapes,
            out_shape=task_info.shape,
            transform=transform,
            fill=np.nan,
            dtype=np.float32,
            all_touched=True,
        )
        out_arr = raster.astype(np.float32)

        for yr, name in zip(gdf["FIRE_YEAR_INT"], gdf["INCIDENT"]):
            incidents_summary.setdefault(int(yr), []).append(name or "(unnamed)")
        for yr in sorted(incidents_summary):
            names = ", ".join(sorted(set(incidents_summary[yr])))
            log.info(f"  {yr}: {names}")

    n_burned = int(np.sum(~np.isnan(out_arr)))
    log.info(
        f"✓ {layer_name}: {n_burned}/{out_arr.size} px burned in prior "
        f"{lookback_years} yr (most-recent year per pixel)"
    )

    return DataLayer(
        name=layer_name,
        data=[out_arr],
        timestamps=[task_info.t_start],
        source=(
            "NIFC "
            + " + ".join(u.split("/services/")[1].split("/")[0] for u in urls)
            + " (services3.arcgis.com/T4QMspbfLg3qTGWY)"
        ),
        native_resolution=task_info.resolution,
        unit="calendar year (NaN = unburned)",
        note={
            "lookback_years": lookback_years,
            "year_window": [year_min, year],
            "t_start": task_info.t_start.isoformat(),
            "n_features": len(feats),
            "services": [u.split("/services/")[1].split("/")[0] for u in urls],
            "incidents_by_year": {
                str(y): sorted(set(n)) for y, n in incidents_summary.items()
            },
            "description": (
                "Most-recent burn year per pixel from NIFC IFPH; NaN where "
                "no perimeter overlaps the pixel in the lookback window. "
                "Replaces the LANDFIRE Annual Disturbance + HDIST role for "
                "the missing-recent-fires gap."
            ),
        },
    )
