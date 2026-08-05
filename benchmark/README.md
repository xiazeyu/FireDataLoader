# Runtime benchmark

`run_benchmark.py` measures end-to-end wall-clock time for forging a complete,
harmonized data cube per fire event.

## What it measures

For every MTBS event listed in `../events.txt` (written by
`python main.py --fetch-examples`), the script times
a full from-scratch forge (`main.py <event> --workers 1`, so the wall time is
attributable to that single event) under two cache regimes:

| Regime | Caches | What hits the network |
| ------ | ------ | --------------------- |
| **cold** | run-generated caches (HRRR GRIB, GlobalWUI tiles, FIRMS slices, MTBS fire-list) are wiped immediately before each rep | every byte is fetched over the network |
| **warm** | caches left populated by the preceding cold run | HRRR / WUI / fire-list / FIRMS slices come from local disk; Earth Engine and NIFC (which have no on-disk cache) still hit the network |

Each event is run `COLD_REPS` times cold (caches re-wiped each rep) then
`WARM_REPS` times warm (both default to 3). Reported timings depend on your
machine, your network, and upstream-service latency, so absolute numbers will
vary between runs.

## Prerequisites

The benchmark runs the full pipeline, so it needs the same setup as a normal
forge — see the [project README](../README.md) for details:

- Dependencies installed (`uv sync`).
- Credentials configured (NASA FIRMS map key, Earth Engine auth). Missing
  credentials don't crash the run — affected layers fail-soft and are skipped —
  but the timing then won't reflect a complete cube.
- Network access (a cold run fetches every source; a warm run still reaches the
  uncached Earth Engine and NIFC).
- The `FEDS25MTBS` archive under `../datasets/` is optional and is **not**
  wiped by cold runs; without it, perimeter-dependent layers degrade gracefully.

## Running

From the repository root, using either recommended entry point:

```bash
# via uv
uv run python benchmark/run_benchmark.py

# or, with the project environment activated
python benchmark/run_benchmark.py
```

> [!WARNING]
> A **cold** run deletes the software-managed cache listed in `CACHES` (the whole
> `cache/` directory — HRRR GRIBs, Global WUI tiles, FIRMS slices, the MTBS fire
> list, any range-fetched fires) before each repetition. It is re-downloaded
> automatically on the next run, but the first forge afterwards will be slower
> while it repopulates. The user-managed `datasets/` bucket (e.g. the downloaded
> `FEDS25MTBS` dataset) is never touched.

## Configuration

Edit the constants at the top of `run_benchmark.py`:

| Constant | Default | Meaning |
| -------- | ------- | ------- |
| `COLD_REPS` | `3` | cold repetitions per event |
| `WARM_REPS` | `3` | warm repetitions per event |
| `CACHES` | `[<repo>/cache]` | filesystem paths wiped before each cold rep — the whole software-managed `cache/` directory (HRRR GRIBs, Global WUI tiles, FIRMS slices, the MTBS fire list, range-fetched fires) |

To benchmark a different set of events, edit `../events.txt`
(one event ID per line).

## Output

`results.jsonl` is rewritten fresh on each invocation, with one JSON record
appended per run:

```json
{"event": "CA3432611848120191010", "regime": "cold", "rep": 1,
 "wall_s": 118.4, "returncode": 0, "shape": [...], "resolution_m": 30,
 "perimeter_frames": 11, "hrrr_frames": 264, "counts": {...},
 "t_start": "...", "t_end": "..."}
```

`wall_s` and `returncode` are always present; the grid/metadata fields
(`shape`, `resolution_m`, `perimeter_frames`, `hrrr_frames`, `counts`,
`t_start`, `t_end`) are copied from each event's `task_summary.json` when the
run produced one. Per-run stdout/stderr is captured under `logs/`, and the heavy
`.npy` cube outputs written under `out/` are deleted after each run once the
summary has been recorded.

`results.jsonl`, `logs/`, and `out/` are generated artifacts and are
git-ignored — only this README and the script are committed.

> **Timings published before the tight-AOI default are not comparable.** The
> reference results (and the Zenodo reproducibility artifact) were produced with
> the legacy AOI and a 100 m buffer. Grid sizes, and therefore wall-clock, now
> differ: large fires shrink, small fires grow with the wider default buffer. Add
> `--aoi-mode bbox --buffer 100` to reproduce the published grids and timings.
