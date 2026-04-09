"""
Plot each data channel as a separate PNG file for a fire event.

Usage:
    python plot_channels.py Eaton_2025
    python plot_channels.py Eaton_2025 -o output
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from main import load_numpy
from plot_data import (
    LANDCOVER_CLASSES,
    WUI_CLASSES,
    plot_single_layer,
    plot_landcover,
    plot_wui,
)

# Same plot config as plot_data.py
PLOT_CONFIG = {
    'elevation': {'cmap': 'terrain', 'label': 'Elevation (m)'},
    'burn_perimeters': {'cmap': 'Reds', 'label': 'Burn Perimeter'},
    'frp': {'cmap': 'hot', 'label': 'Fire Radiative Power (MW)'},
    'frp_day': {'cmap': 'hot', 'label': 'Daytime FRP (MW)'},
    'frp_night': {'cmap': 'hot', 'label': 'Nighttime FRP (MW)'},
    'cbd': {'cmap': 'YlGn', 'label': 'Canopy Bulk Density'},
    'cc': {'cmap': 'Greens', 'label': 'Canopy Cover (%)'},
    'r2': {'cmap': 'Blues', 'label': 'Relative Humidity (%)'},
    'u10': {'cmap': 'coolwarm', 'label': 'Wind U (m/s)'},
    'v10': {'cmap': 'coolwarm', 'label': 'Wind V (m/s)'},
    'building_height': {'cmap': 'plasma', 'label': 'Building Height (m)'},
    'landcover': {'cmap': 'tab20', 'label': 'Land Cover Class'},
    'lai': {'cmap': 'YlGn', 'label': 'LAI (m²/m²)'},
    'satellite': {'cmap': None, 'label': 'Satellite (RGB)'},
    'hillshade': {'cmap': 'gray', 'label': 'Hillshade'},
    'wui': {'cmap': 'tab10', 'label': 'Wildland-Urban Interface'},
    'fireline': {'cmap': 'hot', 'label': 'Fireline Intensity'},
    'fireline_max': {'cmap': 'hot', 'label': 'Fireline Max Intensity'},
}


def plot_channel(event_id: str, data_obj, name: str, event_path: str):
    """Plot a single channel and save as its own PNG."""
    config = PLOT_CONFIG.get(name, {'cmap': 'viridis', 'label': name})

    if not data_obj.data or len(data_obj.data) == 0:
        return

    # For burn_perimeters, plot first and last frame side by side
    if name == 'burn_perimeters' and len(data_obj.data) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'{event_id} — {config["label"]}', fontsize=13, fontweight='bold')

        for ax, (frame_idx, label) in zip(axes, [(0, 'First'), (-1, 'Last')]):
            frame = data_obj.data[frame_idx]
            actual_num = frame_idx + 1 if frame_idx >= 0 else len(data_obj.data)
            title = f'{label} — Frame {actual_num}/{len(data_obj.data)}'
            if data_obj.resolution:
                title += f' @ {data_obj.resolution}m'
            plot_single_layer(ax, frame, title, cmap=config['cmap'])
    else:
        plot_data = data_obj.data[0]
        if not isinstance(plot_data, np.ndarray):
            return

        title = config['label']
        if data_obj.unit:
            title = f'{name} ({data_obj.unit})'
        if data_obj.resolution:
            title += f' @ {data_obj.resolution}m'
        if len(data_obj.data) > 1:
            title += f'\n[Frame 1/{len(data_obj.data)}]'

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.suptitle(f'{event_id} — {config["label"]}', fontsize=13, fontweight='bold')

        if plot_data.ndim == 3 and plot_data.shape[2] == 3:
            im = ax.imshow(plot_data)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_visible(False)
        elif name == 'landcover':
            plot_landcover(ax, plot_data, title)
        elif name == 'wui':
            plot_wui(ax, plot_data, title)
        else:
            plot_single_layer(ax, plot_data, title, cmap=config['cmap'])

    plt.tight_layout()
    out_path = os.path.join(event_path, f'{event_id}_{name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_all_channels(event_id: str, output_dir: str = 'output'):
    event_path = os.path.join(output_dir, event_id)
    if not os.path.exists(event_path):
        print(f'Error: Directory not found: {event_path}')
        sys.exit(1)

    npy_files = sorted(f for f in os.listdir(event_path) if f.endswith('.npy') and f != 'task_info.npy')
    if not npy_files:
        print(f'Error: No .npy files found in {event_path}')
        sys.exit(1)

    print(f'Plotting {len(npy_files)} channels for event: {event_id}')

    for filename in npy_files:
        filepath = os.path.join(event_path, filename)
        data_obj = load_numpy(filepath)
        name = data_obj.name
        plot_channel(event_id, data_obj, name, event_path)

    print(f'\nDone — {len(npy_files)} channel PNGs saved to {event_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot each channel as a separate PNG')
    parser.add_argument('event_id', type=str, help='Fire event ID')
    parser.add_argument('-o', '--output_dir', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    plot_all_channels(args.event_id, args.output_dir)
