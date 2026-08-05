# Quantitative validation

Scripts backing the "Quantitative Validation" experiments. They operate on event output directories produced by the
pipeline (`output/<event_id>/`).

## Run

```bash
# Validate the events listed in events.txt (results -> output/validation_metrics.csv)
python validation/run_validation.py --events events.txt --output-dir output

# Or pass event IDs directly
python validation/run_validation.py CA3432611848120191010 CO4020310623920201014
```

## Metrics

Each event contributes one CSV row; the column names below are `<metric>.<field>`.

| Metric (CSV key) | What it checks |
|------------------|----------------|
| `reprojection` | Pixel-center displacement after target-CRS → EPSG:4326 → target-CRS round trip (meters). Catches CRS / axis-order regressions. |
| `frp_conservation` | Reloads the event's VIIRS fire points, re-splats them unmasked, and compares the rasterized total against two references: the summed point FRP (`rel_error`, i.e. the loss at the grid edge) and the expected in-grid Gaussian mass (`rel_error_vs_expected`, which must be ~0). The second is the real assertion — see the note below. |
| `categorical_landcover` | Overall accuracy between the pipeline `landcover` and a majority-aggregated native ESA WorldCover (10 m) reference on the same grid — the fraction of pixels where nearest-neighbour resampling agrees with the dominant native class (its complement is the disagreement rate). |
| `categorical_wui` | Overall accuracy between the pipeline `wui` and a majority-aggregated native GlobalWUI (10 m) reference. |
| `continuous_elevation` | RMSE / MAE of `elevation` vs. the native 3DEP DEM mean-aggregated to the target grid — the residual is sub-pixel terrain variance discarded by bilinear resampling. |
| `registration_elevation` | Sub-pixel alignment shift (meters) between `elevation` and its native reference, by phase correlation. A near-zero shift is direct evidence of correct co-registration. |

The categorical and continuous metrics build a native-resolution reference
and aggregate it independently of the pipeline's
resampling — majority for categorical, mean for continuous — so the comparison
measures resampling fidelity rather than being tautological. `landcover` / `elevation`
re-fetch from Earth Engine; `wui` re-reads the
native GlobalWUI tiles. Large events are fetched in tiles to stay under Earth Engine's
per-request reprojection cap. When a backend is unavailable the metric returns `{}`
and the runner still writes the remaining metrics.

`registration_elevation` phase-correlates the pipeline DEM against the area-aggregated
native reference. Across the bundled events the shift is consistently sub-pixel
(≲ 0.7 px, ≲ 20 m on the 30 m grid); part of it is the bilinear-vs-mean aggregation
difference rather than true mis-registration, so read it as a sub-pixel upper bound,
corroborated by the low elevation RMSE (a 0.7 px offset over sloped terrain would
inflate that RMSE far beyond the ~4.5 m observed).

## Results across the 8 evaluation events

Running the full suite over `events.txt`
(`python validation/run_validation.py --events events.txt --output-dir output`)
writes `output/validation_metrics.csv`; that file is also archived, alongside the
benchmark outputs, in the FireDataForge reproducibility artifact on Zenodo
([doi:10.5281/zenodo.20743743](https://doi.org/10.5281/zenodo.20743743)).
The per-event results:

| Event | MTBS Event ID | Grid (H×W) | Reproj. RMSE (m) | FRP cons. err. | Land cover OA | WUI OA | Elevation RMSE / MAE (m) | Registration (px) |
|-------|---------------|-----------|------------------|----------------|---------------|--------|--------------------------|-------------------|
| Palisades 2025 | `CA3406811855120250107` | 572×716 | 2.8×10⁻⁹ | 0.006% | 0.945 | 0.998 | 4.71 / 3.07 | 0.67 |
| Eaton 2025 | `CA3419211810520250108` | 396×519 | 2.9×10⁻⁹ | 0.029% | 0.932 | 0.999 | 7.64 / 5.36 | 0.66 |
| Saddleridge 2019 | `CA3432611848120191010` | 361×388 | 2.9×10⁻⁹ | <0.001% | 0.922 | 0.998 | 4.64 / 2.83 | 0.28 |
| Tubbs 2017 | `CA3859812261820171009` | 848×681 | 3.0×10⁻⁹ | <0.001% | 0.950 | 0.997 | 3.88 / 2.60 | 0.49 |
| Camp 2018 | `CA3982012144020181108` | 1397×1448 | 3.0×10⁻⁹ | <0.001% | 0.947 | 0.998 | 4.21 / 2.63 | 0.23 |
| Spring Creek 2018 | `CO3749610529120180627` | 1194×944 | 2.9×10⁻⁹ | <0.001% | 0.969 | 0.999 | 3.52 / 2.36 | 0.37 |
| Black Forest 2013 | `CO3901210474920130611` | 369×462 | 3.0×10⁻⁹ | <0.001% | 0.938 | 0.998 | 0.97 / 0.73 | 0.49 |
| East Troublesome 2020 | `CO4020310623920201014` | 1016×1602 | 3.0×10⁻⁹ | <0.001% | 0.949 | 0.999 | 4.58 / 3.11 | 0.21 |
| **Mean (n=8)** | — | — | **2.9×10⁻⁹** | **0.004%** | **0.944** | **0.998** | **4.27 / 2.84** | **0.43** |

Round-trip reprojection RMSE is ≈3×10⁻⁹ m (sub-nanometer) for every event — far below
any physically meaningful displacement, confirming the CRS/axis-order handling is exact
(the per-pixel maximum displacement stays ≤1.2×10⁻⁸ m across all events).
FRP conservation holds to ≤0.03% against the expected in-grid mass. Note that
`rel_error` (measured against the summed point FRP) is *expected* to be non-zero
whenever a footprint is clipped by the grid edge, which is common: the AOI margin is
routinely smaller than the 570 m splat kernel radius. `rel_error` alone therefore
cannot distinguish a correct edge loss from a broken operator — an implementation
that renormalized by the in-grid weights would drive it to ~0 by piling the off-grid
mass back onto the boundary pixels. `rel_error_vs_expected` is the assertion that
actually constrains the splat. Land-cover overall accuracy averages 0.944 and
WUI overall accuracy 0.998 against the independently majority-aggregated native references
(a 5.6% and 0.2% mean per-pixel disagreement rate, respectively); elevation RMSE averages
4.27 m (well under the 30 m grid spacing); and registration stays sub-pixel (≤0.67 px,
≤20 m) on every event.

Building height is intentionally omitted: the pipeline rasterizes vector building
footprints with area-weighted aggregation, so there is no independent native raster to
aggregate against — any reference recomputed from the same footprints would be
tautological.

> Temporal alignment is correct by construction — the consumption-time per-source
> cursor never exposes a not-yet-valid observation — so no quantitative temporal
> metric is included.
