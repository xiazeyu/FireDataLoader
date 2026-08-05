"""FireDataForge runtime benchmark.

For each MTBS event in events.txt, time a full from-scratch forge under two
cache regimes:

  cold : run-generated caches (HRRR GRIB, GlobalWUI tiles, FIRMS slices, MTBS
         fire-list) are wiped immediately before the run -> every byte is fetched
         over the network. Repeated COLD_REPS times (caches re-wiped each rep).
  warm : caches left populated by a prior run on the same event -> HRRR/WUI/
         fire-list/FIRMS slices come from local disk; GEE and NIFC (which have no
         on-disk cache) still hit the network. Repeated WARM_REPS times.

Each event is run with --workers 1 (one event at a time) so the wall-clock time
is attributable to that single event. Results are appended to results.jsonl and
summarized to results_summary.json.
"""
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmark", "out")
RESULTS = os.path.join(ROOT, "benchmark", "results.jsonl")
LOGDIR = os.path.join(ROOT, "benchmark", "logs")

# All software-fetched data (HRRR GRIB, GlobalWUI tiles, FIRMS slices, MTBS
# fire-list, any range-fetched fires) lives under cache/; wipe it for a cold run.
# The user-managed datasets/ bucket (e.g. datasets/FEDS25MTBS) is kept.
CACHES = [os.path.join(ROOT, "cache")]

COLD_REPS = 3
WARM_REPS = 3


def wipe_caches():
    for p in CACHES:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)


def run_event(event, regime, rep):
    """Run one event once; return a result dict with wall time + grid metadata."""
    out_dir = os.path.join(OUT, f"{event}_{regime}_{rep}")
    shutil.rmtree(out_dir, ignore_errors=True)
    log = os.path.join(LOGDIR, f"{event}_{regime}_{rep}.log")
    # Re-use the interpreter that launched this script.
    cmd = [sys.executable, os.path.join(ROOT, "main.py"),
           event, "-o", out_dir, "--workers", "1"]
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - t0

    rec = {"event": event, "regime": regime, "rep": rep,
           "wall_s": round(wall, 3), "returncode": rc}
    summ_path = os.path.join(out_dir, event, "task_summary.json")
    if os.path.isfile(summ_path):
        s = json.load(open(summ_path))
        lyr = s.get("layers", {})
        rec.update({
            "shape": s.get("shape"),
            "resolution_m": s.get("resolution_m"),
            "perimeter_frames": (lyr.get("burn_perimeter") or {}).get("n_frames"),
            "hrrr_frames": (lyr.get("hrrr") or {}).get("n_frames"),
            "counts": s.get("counts"),
            "t_start": s.get("t_start"),
            "t_end": s.get("t_end"),
        })
    # Summary already captured in `rec`; drop the heavy .npz outputs on disk.
    shutil.rmtree(out_dir, ignore_errors=True)
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{regime} rep{rep}] {event}: {wall:6.1f}s rc={rc} "
          f"shape={rec.get('shape')} frames={rec.get('perimeter_frames')} "
          f"hrrr={rec.get('hrrr_frames')} counts={rec.get('counts')}", flush=True)
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    events_path = os.path.join(ROOT, "events.txt")
    if not os.path.isfile(events_path):
        sys.exit(
            "events.txt not found at the repo root. The benchmark runs on the "
            "example fires; stage them first with:\n\n"
            "    python main.py --fetch-examples\n\n"
            "(or create events.txt yourself, one MTBS Event ID per line)."
        )
    events = [line.strip() for line in open(events_path)
              if line.strip() and not line.startswith("#")]
    open(RESULTS, "w").close()  # fresh results file
    print(f"Benchmarking {len(events)} events: {COLD_REPS} cold + {WARM_REPS} warm reps each")
    for event in events:
        # COLD: wipe caches before every rep so each is a true from-scratch run.
        for rep in range(1, COLD_REPS + 1):
            wipe_caches()
            run_event(event, "cold", rep)
        # WARM: caches now populated by the last cold rep; keep them.
        for rep in range(1, WARM_REPS + 1):
            run_event(event, "warm", rep)
    print("DONE")


if __name__ == "__main__":
    main()
