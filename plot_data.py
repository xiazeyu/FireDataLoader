"""
Plot all exported numpy data for a fire event.

Usage:
    python plot_data.py <event_id>
    python plot_data.py CA3859812261820171009
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from main import load_numpy

# ESA WorldCover land cover classes
LANDCOVER_CLASSES = {
    10: ('Tree Cover', '#006400'),           # Dark green
    20: ('Shrubland', '#ffbb22'),             # Orange/yellow
    30: ('Grassland', '#ffff4c'),             # Yellow
    40: ('Cropland', '#f096ff'),              # Pink
    50: ('Built-up', '#fa0000'),              # Red
    60: ('Bare/Sparse Vegetation', '#b4b4b4'), # Gray
    70: ('Snow and Ice', '#f0f0f0'),          # White
    80: ('Permanent Water Bodies', '#0064c8'), # Blue
    90: ('Herbaceous Wetland', '#0096a0'),    # Teal
    95: ('Mangroves', '#00cf75'),             # Green
    100: ('Moss and Lichen', '#fae6a0'),      # Beige
}


def plot_single_layer(ax, data, title, cmap='viridis', vmin=None, vmax=None):
    """Plot a single 2D data layer."""
    im = ax.imshow(data, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_landcover(ax, data, title):
    """Plot land cover data with discrete legend showing only present classes."""
    # Find unique values present in data
    unique_values = np.unique(data)
    unique_values = unique_values[unique_values != 0]  # Exclude NoData (0)
    
    # Filter to only known classes
    present_classes = [v for v in unique_values if v in LANDCOVER_CLASSES]
    
    if not present_classes:
        ax.imshow(data, cmap='tab20', interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        return
    
    # Create colormap for present classes
    colors = [LANDCOVER_CLASSES[v][1] for v in present_classes]
    cmap = ListedColormap(colors)
    
    # Create boundaries for discrete colormap
    bounds = present_classes + [present_classes[-1] + 1]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot
    im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    # Create a colorbar with class labels (same size as other colorbars for alignment)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Set ticks at center of each color segment
    tick_locs = [v + 0.5 for v in present_classes[:-1]] + [present_classes[-1]]
    tick_locs = [(present_classes[i] + present_classes[i] + 1) / 2 if i < len(present_classes) - 1 
                 else present_classes[i] for i in range(len(present_classes))]
    # Simpler: just use the class values as tick locations
    cbar.set_ticks(present_classes)
    cbar.set_ticklabels([LANDCOVER_CLASSES[v][0] for v in present_classes])
    cbar.ax.tick_params(labelsize=6)


def plot_event_data(event_id: str, output_dir: str = 'output'):
    """Load and plot all data for a fire event."""
    event_path = os.path.join(output_dir, event_id)
    
    if not os.path.exists(event_path):
        print(f"Error: Directory not found: {event_path}")
        sys.exit(1)
    
    # Find all .npy files
    npy_files = sorted([f for f in os.listdir(event_path) if f.endswith('.npy')])
    
    if not npy_files:
        print(f"Error: No .npy files found in {event_path}")
        sys.exit(1)
    
    print(f"Found {len(npy_files)} data files for event: {event_id}")
    
    # Define colormaps and settings for different data types
    plot_config = {
        'elevation': {'cmap': 'terrain', 'label': 'Elevation (m)'},
        'burn_perimeters': {'cmap': 'Reds', 'label': 'Burn Perimeter'},
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
    }
    
    # Skip task_info for plotting
    data_files = [f for f in npy_files if f != 'task_info.npy']
    n_plots = len(data_files)
    
    if n_plots == 0:
        print("No plottable data files found.")
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f'Fire Event: {event_id}', fontsize=14, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, filename in enumerate(data_files):
        filepath = os.path.join(event_path, filename)
        data_obj = load_numpy(filepath)
        
        name = data_obj.name
        config = plot_config.get(name, {'cmap': 'viridis', 'label': name})
        
        # Get the first frame (or only frame for static data)
        if data_obj.data and len(data_obj.data) > 0:
            plot_data = data_obj.data[0]
            
            # Handle non-array data (like task_info)
            if not isinstance(plot_data, np.ndarray):
                axes[idx].text(0.5, 0.5, f'{name}\n(non-array data)', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
                continue
            
            # Build title with metadata
            title = config['label']
            if data_obj.unit:
                title = f"{name} ({data_obj.unit})"
            if data_obj.resolution:
                title += f" @ {data_obj.resolution}m"
            
            # For time series data, show first frame info
            if len(data_obj.data) > 1:
                title += f"\n[Frame 1/{len(data_obj.data)}]"
            
            # Handle RGB images (satellite data) - 3D array with shape (H, W, 3)
            if plot_data.ndim == 3 and plot_data.shape[2] == 3:
                im = axes[idx].imshow(plot_data)
                axes[idx].set_title(title, fontsize=10)
                axes[idx].axis('off')
                # Add invisible colorbar to maintain alignment with other plots
                cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                cbar.ax.set_visible(False)
            # Handle landcover with discrete legend
            elif name == 'landcover':
                plot_landcover(axes[idx], plot_data, title)
            else:
                plot_single_layer(axes[idx], plot_data, title, cmap=config['cmap'])
            
            # Print stats
            print(f"  {name}: shape={plot_data.shape}, "
                  f"min={plot_data.min():.2f}, max={plot_data.max():.2f}, "
                  f"frames={len(data_obj.data)}")
        else:
            axes[idx].text(0.5, 0.5, f'{name}\n(no data)', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_fig = os.path.join(event_path, f'{event_id}_overview.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"\nSaved overview plot to: {output_fig}")
    
    plt.show()


def plot_time_series(event_id: str, layer_name: str, output_dir: str = 'output'):
    """Plot all frames of a time series layer."""
    filepath = os.path.join(output_dir, event_id, f'{layer_name}.npy')
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    data_obj = load_numpy(filepath)
    n_frames = len(data_obj.data)
    
    if n_frames <= 1:
        print(f"{layer_name} has only {n_frames} frame(s). Use plot_event_data instead.")
        return
    
    print(f"Plotting {n_frames} frames for {layer_name}")
    
    n_cols = min(4, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    fig.suptitle(f'{event_id} - {layer_name} Time Series', fontsize=12)
    
    axes = axes.flatten() if n_frames > 1 else [axes]
    
    for idx, frame in enumerate(data_obj.data):
        title = f"Frame {idx + 1}"
        if data_obj.timestamps and idx < len(data_obj.timestamps):
            title = data_obj.timestamps[idx].strftime('%Y-%m-%d %H:%M')
        
        im = axes[idx].imshow(frame, cmap='Reds', interpolation='nearest')
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis('off')
    
    for idx in range(n_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot exported fire event data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('event_id', type=str, help='Fire event ID to plot')
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default='output',
        help='Output directory containing event data'
    )
    parser.add_argument(
        '-t', '--timeseries',
        type=str,
        default=None,
        help='Plot time series for specific layer (e.g., burn_perimeters)'
    )
    
    args = parser.parse_args()
    
    if args.timeseries:
        plot_time_series(args.event_id, args.timeseries, args.output_dir)
    else:
        plot_event_data(args.event_id, args.output_dir)


if __name__ == '__main__':
    main()
