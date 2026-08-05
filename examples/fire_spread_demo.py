"""Toy fire-spread demo over FireDataForge layers.

A tiny cellular automaton that consumes one event's harmonized terrain, fuel, and
wind layers, ignites the first observed burn perimeter, and watches the fire crawl
downwind and uphill. The point is to show that the standardized arrays drop straight
into a simulation -- this is a teaching toy, not a calibrated fire model.

Open it as a notebook (the ``# %%`` cells) or just run it:

    python examples/fire_spread_demo.py [output/<event_id>]
"""

# %% Setup -- make `import firedataforge` work when run from anywhere in the repo.
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import firedataforge as fdf

# Any processed event under output/ works -- pass one on the command line to override.
EVENT = sys.argv[1] if len(sys.argv) > 1 else "output/CA3432611848120191010"
STEPS = 100  # how many time steps to simulate
print("event:", EVENT)


# %% Load the harmonized layers: terrain, fuel, wind, and an ignition source.
# They're already on one common grid, so they stack with no reprojection on our part.
def layer(name):
    """First frame of one layer as a float array (or None if not produced)."""
    path = fdf.resolve_path(os.path.join(EVENT, f"{name}{fdf.DATA_EXT}"))
    return np.asarray(fdf.load_numpy(path).data[0], float) if os.path.exists(path) else None


elevation = layer("elevation")
H, W = elevation.shape

# Fuel: canopy cover (%) -> 0..1, with a floor so sparse ground still carries fire.
canopy = layer("canopy_cover")
fuel = np.clip(canopy / 100.0, 0, 1) if canopy is not None else np.full((H, W), 0.5)
flammability = 0.4 + 0.6 * fuel

# Wind: collapse the 10 m u/v fields to one mean vector in (row, col) image space.
# +u blows east (+col); +v blows north, which is -row in a top-down raster.
u, v = layer("u10"), layer("v10")
wind = np.array([-np.nanmean(v) if v is not None else 0.0,
                 np.nanmean(u) if u is not None else 0.0])
wind = wind / (np.hypot(*wind) + 1e-9)

# Terrain: fire runs faster uphill, so precompute the elevation gradient.
grad_row, grad_col = np.gradient(elevation)

print(f"grid {H}x{W} | mean fuel {fuel.mean():.2f} | wind(row,col) {wind.round(2)}")


# %% Ignite and run the automaton. state: 0 = unburned, 1 = burning, 2 = burned.
# Once lit, a cell keeps burning for BURN_TIME steps -- long enough to light its
# neighbours -- then burns out. A single-step flash would just die on this sparse fuel.
BASE = 0.45  # ignition chance of a fully-fueled, downwind neighbour
BURN_TIME = 4
NEIGHBORS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]

state = np.zeros((H, W), np.uint8)
life = np.zeros((H, W), int)
ignition = layer("burn_perimeter")  # first observed perimeter = where the fire started
if ignition is not None:
    seed = ignition > 0
else:
    seed = np.zeros((H, W), bool)
    seed[H // 2 - 1:H // 2 + 1, W // 2 - 1:W // 2 + 1] = True
state[seed] = 1
life[seed] = BURN_TIME

rng = np.random.default_rng(0)
frames = [state.copy()]
for _ in range(STEPS):
    burning = state == 1
    prob = np.zeros((H, W))
    for dy, dx in NEIGHBORS:
        wind_push = max(1 + 0.7 * (dy * wind[0] + dx * wind[1]), 0)  # downwind favoured
        slope_push = np.clip(1 + 0.04 * (dy * grad_row + dx * grad_col), 0, None)  # uphill favoured
        from_burning = np.roll(np.roll(burning, dy, 0), dx, 1)
        prob = np.maximum(prob, from_burning * BASE * flammability * wind_push * slope_push)
    ignite = (state == 0) & (rng.random((H, W)) < prob)
    life[burning] -= 1
    state[(state == 1) & (life <= 0)] = 2  # burnt out
    state[ignite] = 1
    life[ignite] = BURN_TIME
    frames.append(state.copy())
    if not (state == 1).any():
        break

print(f"ran {len(frames) - 1} steps | {int((state == 2).sum())} px burned "
      f"({(state == 2).mean() * 100:.1f}% of the grid)")


# %% Show the progression: ignition -> mid-burn -> final scar.
cmap = ListedColormap(["#1b4332", "#ff4d1f", "#3a3a3a"])  # unburned, burning, burned
picks = [0, len(frames) // 2, len(frames) - 1]
titles = ["ignition", "mid-burn", "final"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, i, title in zip(axes, picks, titles):
    f = frames[i]
    ax.imshow(f, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(f"{title} (step {i}) -- {int((f == 2).sum())} px burned")
    ax.set_xticks([])
    ax.set_yticks([])
legend = [Patch(facecolor=c, label=lbl)
          for c, lbl in zip(cmap.colors, ["unburned", "burning", "burned"])]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
fig.suptitle(f"Toy fire-spread demo -- {os.path.basename(EVENT.rstrip('/'))}", fontsize=13)
fig.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()
