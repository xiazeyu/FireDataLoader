"""Test ``coordinates.npy`` produced by ``save_coordinates``.

Verifies:

1. The saved coordinate arrays are consistent with ``task_info`` (shape,
   bounds, monotonicity, pixel size).
2. The grid can be wrapped into an ``xarray.DataArray`` with the saved CRS
   and re-projected/plotted on a Cartopy ``Orthographic(-10, 45)`` map.

Run::

    python test_coordinates.py CA3859812261820171009 --layer elevation

If ``coordinates.npy`` is missing for the event (e.g. generated before this
feature existed), the script regenerates it from ``task_info.npy``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from main import load_numpy, save_coordinates
from schemas import TaskInfo


def ensure_coordinates(event_dir: str) -> str:
    """Make sure ``coordinates.npy`` exists in ``event_dir``; create if not."""
    coord_path = os.path.join(event_dir, "coordinates.npy")
    if os.path.exists(coord_path):
        return coord_path

    task_path = os.path.join(event_dir, "task_info.npy")
    if not os.path.exists(task_path):
        raise FileNotFoundError(
            f"Neither coordinates.npy nor task_info.npy found in {event_dir}"
        )

    task_obj = load_numpy(task_path)
    raw = task_obj.data[0]
    task_info = raw if isinstance(raw, TaskInfo) else TaskInfo(**raw)

    output_dir = os.path.dirname(event_dir.rstrip(os.sep))
    save_coordinates(task_info, output_dir)
    print(f"[setup] Generated {coord_path} from task_info.npy")
    return coord_path


def check_coordinates(event_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Validate coordinates.npy against task_info.npy. Returns (x, y, note)."""
    coords = load_numpy(os.path.join(event_dir, "coordinates.npy"))
    task = load_numpy(os.path.join(event_dir, "task_info.npy"))
    raw = task.data[0]
    task_info = raw if isinstance(raw, TaskInfo) else TaskInfo(**raw)

    x, y = coords.data
    note = coords.note
    minx, miny, maxx, maxy = task_info.bounds
    height, width = task_info.shape

    assert x.shape == (width,), f"x shape {x.shape} != ({width},)"
    assert y.shape == (height,), f"y shape {y.shape} != ({height},)"
    assert np.all(np.diff(x) > 0), "x must be strictly increasing"
    assert np.all(np.diff(y) < 0), "y must be strictly decreasing (top->bottom)"

    px = (maxx - minx) / width
    py = (maxy - miny) / height
    assert np.allclose(x[0], minx + 0.5 * px), "x[0] not pixel-centered"
    assert np.allclose(y[0], maxy - 0.5 * py), "y[0] not pixel-centered"
    assert np.allclose(x[-1], maxx - 0.5 * px), "x[-1] not pixel-centered"
    assert np.allclose(y[-1], miny + 0.5 * py), "y[-1] not pixel-centered"

    assert note["crs"] == task_info.crs, "CRS mismatch"
    assert tuple(note["shape"]) == (height, width), "shape mismatch"

    print("[check] coordinates.npy is consistent with task_info.npy")
    print(f"        crs={note['crs']}  shape={note['shape']}  res={note['resolution']}m")
    print(f"        x: [{x[0]:.2f}, {x[-1]:.2f}]  y: [{y[0]:.2f}, {y[-1]:.2f}]")
    return x, y, note


def plot_on_orthographic(
    event_dir: str,
    layer: str,
    out_path: str,
    central_lon: float = -10.0,
    central_lat: float = 45.0,
    show: bool = False,
) -> None:
    """Wrap a layer into xarray and plot it on ccrs.Orthographic(lon, lat)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import xarray as xr
    from pyproj import CRS

    x, y, note = check_coordinates(event_dir)

    layer_path = os.path.join(event_dir, f"{layer}.npy")
    if not os.path.exists(layer_path):
        raise FileNotFoundError(f"Layer not found: {layer_path}")
    layer_obj = load_numpy(layer_path)

    arr = np.asarray(layer_obj.data[0])
    rgb = None
    if arr.ndim == 3:
        # (H, W, 3) RGB or (3, H, W).
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3]
        elif arr.shape[0] in (3, 4):
            rgb = np.transpose(arr[:3], (1, 2, 0))
        else:
            arr = arr[0]
    if rgb is not None:
        # Normalize RGB to [0, 1] for imshow.
        rgb_f = rgb.astype(float)
        if rgb_f.max() > 1.5:
            rgb_f = rgb_f / 255.0
        rgb_f = np.clip(rgb_f, 0, 1)

    da_crs = CRS.from_user_input(note["crs"])
    src_crs = ccrs.epsg(da_crs.to_epsg()) if da_crs.to_epsg() else ccrs.PlateCarree()

    proj = ccrs.Orthographic(central_longitude=central_lon,
                             central_latitude=central_lat)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.coastlines(linewidth=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)

    # Sanity check distance to projection centre.
    # Use the geographic centroid of the data extent for the message.
    import xarray as _xr  # noqa: F401
    da = xr.DataArray(
        arr if rgb is None else rgb_f,
        dims=("y", "x") if rgb is None else ("y", "x", "band"),
        coords={"y": y, "x": x},
        name=layer_obj.name,
    )
    # Lon/lat of data centre (just for the warning).
    from pyproj import Transformer
    tr = Transformer.from_crs(da_crs, "EPSG:4326", always_xy=True)
    cx, cy = float(np.mean(x)), float(np.mean(y))
    lon_c, lat_c = tr.transform(cx, cy)
    from math import acos, cos, radians, sin, degrees
    cos_d = (sin(radians(lat_c)) * sin(radians(central_lat))
             + cos(radians(lat_c)) * cos(radians(central_lat))
             * cos(radians(lon_c - central_lon)))
    cos_d = max(-1.0, min(1.0, cos_d))
    ang = degrees(acos(cos_d))
    print(f"[plot] data centre ≈ ({lon_c:.2f}, {lat_c:.2f}); "
          f"projection centre = ({central_lon}, {central_lat}); "
          f"angular separation = {ang:.1f}°")
    if ang >= 90:
        print("[plot] WARNING: data is on the far side of the globe; "
              "use --central_lon/--central_lat closer to the data, e.g. "
              f"--central_lon {lon_c:.0f} --central_lat {lat_c:.0f}.")

    # Plot in the data's native CRS — cartopy projects on the fly and
    # preserves resolution.
    if rgb is None:
        mesh = ax.pcolormesh(
            x, y, np.ma.masked_invalid(arr.astype(float)),
            transform=src_crs, shading="auto",
        )
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.6, pad=0.05)
        cbar.set_label(layer_obj.name + (f" ({layer_obj.unit})" if layer_obj.unit else ""))
    else:
        # imshow with transform places the RGB array in source CRS coords.
        extent = (float(x[0]), float(x[-1]), float(y[-1]), float(y[0]))
        # regrid_shape forces cartopy to resample to ~native resolution
        # instead of a coarse default that washes the image into one block.
        ax.imshow(
            rgb_f, origin="upper", extent=extent, transform=src_crs,
            regrid_shape=max(rgb_f.shape[:2]),
        )

    ax.set_title(
        f"{layer_obj.name} on Orthographic({central_lon}, {central_lat})"
    )

    # Zoom to data footprint with a little context.
    pad = max((x.max() - x.min()), (y.max() - y.min())) * 0.5
    ax.set_extent(
        [float(x.min()) - pad, float(x.max()) + pad,
         float(y.min()) - pad, float(y.max()) + pad],
        crs=src_crs,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_id", help="Fire event id, e.g. CA3859812261820171009")
    parser.add_argument("-o", "--output_dir", default="output")
    parser.add_argument("--layer", default="elevation",
                        help="Layer .npy to render (default: elevation)")
    parser.add_argument("--central_lon", type=float, default=-10.0,
                        help="Orthographic central longitude (default: -10)")
    parser.add_argument("--central_lat", type=float, default=45.0,
                        help="Orthographic central latitude (default: 45)")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: <event_dir>/<layer>_ortho.png)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    event_dir = os.path.join(args.output_dir, args.event_id)
    if not os.path.isdir(event_dir):
        print(f"Error: {event_dir} not found", file=sys.stderr)
        return 1

    ensure_coordinates(event_dir)
    out_path = args.out or os.path.join(event_dir, f"{args.layer}_ortho.png")
    plot_on_orthographic(
        event_dir, args.layer, out_path,
        central_lon=args.central_lon,
        central_lat=args.central_lat,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
