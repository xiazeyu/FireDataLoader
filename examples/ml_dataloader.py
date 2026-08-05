"""Load FireDataForge outputs straight into an ML pipeline.

Because every layer is harmonized onto one grid, an event's static rasters stack into
a single ``(C, H, W)`` array -- exactly the shape a CNN wants. This walks through the
stacking, then wraps it in a tiny PyTorch ``Dataset``.

Open it as a notebook (the ``# %%`` cells) or just run it:

    python examples/ml_dataloader.py [output]
"""

# %% Setup -- make `import firedataforge` work when run from anywhere in the repo.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import firedataforge as fdf

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "output"

# Static, single-frame layers that stack cleanly into a feature tensor.
CHANNELS = ["elevation", "canopy_bulk_density", "canopy_cover", "building_height",
            "landcover", "lai", "r2", "u10", "v10"]


# %% Stack one event's static layers into (C, H, W) + their names.
def load_event_stack(event_dir, channels=CHANNELS):
    """Stack the available static layers of one event into ``(C, H, W)`` + names.

    Missing layers are skipped (the pipeline is fail-soft, so not every event has
    every layer), and layers that don't share the dominant grid -- e.g. the coarse
    wind fields -- are dropped so the stack stays rectangular.
    """
    arrays, names = [], []
    for name in channels:
        path = fdf.resolve_path(os.path.join(event_dir, f"{name}{fdf.DATA_EXT}"))
        if not os.path.exists(path):
            continue
        frame = np.asarray(fdf.load_numpy(path).data[0], np.float32)
        if frame.ndim == 2:
            arrays.append(frame)
            names.append(name)
    if not arrays:
        raise FileNotFoundError(f"no stackable layers in {event_dir}")
    shape = arrays[0].shape  # the first (full-resolution) layer sets the grid
    keep = [(a, n) for a, n in zip(arrays, names) if a.shape == shape]
    arrays, names = [a for a, _ in keep], [n for _, n in keep]
    return np.stack(arrays), names


# %% Try it on the first event we can find.
events = sorted(d for d in os.listdir(OUTPUT_DIR)
                if os.path.isdir(os.path.join(OUTPUT_DIR, d)))
print(f"{len(events)} event(s) under {OUTPUT_DIR}/: {events}")

stack, names = load_event_stack(os.path.join(OUTPUT_DIR, events[0]))
print(f"\nevent      : {events[0]}")
print(f"stacked    : {stack.shape}  (C, H, W)")
print(f"channels   : {names}")
print(f"value range: {np.nanmin(stack):.2f} .. {np.nanmax(stack):.2f}")


# %% Wrap it as a PyTorch Dataset and pull a sample.
# Everything above is pure NumPy; torch is only needed here, so import it lazily.
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    print("\ninstall torch to run the Dataset part: pip install torch")
    raise SystemExit(0)


class FireEventDataset(Dataset):
    """Yields ``(features, event_id)`` for every processed event under ``output_dir``."""

    def __init__(self, output_dir, channels=CHANNELS):
        self.dirs = [os.path.join(output_dir, e) for e in sorted(os.listdir(output_dir))
                     if os.path.isdir(os.path.join(output_dir, e))]
        self.channels = channels

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        features, _ = load_event_stack(self.dirs[idx], self.channels)
        return torch.from_numpy(features), os.path.basename(self.dirs[idx])


dataset = FireEventDataset(OUTPUT_DIR)
features, event_id = dataset[0]
print(f"\nDataset of {len(dataset)} event(s)")
print(f"sample 0   : {tuple(features.shape)} {features.dtype} tensor for event {event_id}")
# To batch across events, crop them to a common (H, W) first -- grids differ per fire.
