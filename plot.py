"""
Plot all exported numpy data for fire events.

Plotting modes are available (selected with --mode, default "all"):
    overview    - a single combined grid figure of every layer
    channels    - one separate PNG per data channel
    timeseries  - one multi-frame figure per time-series layer (>1 frame)
    both        - overview + channels
    all         - overview + channels + timeseries (default: plot everything)

By default only PNGs are written; pass --pdf to also save the overview as PDF.

Usage:
    Single event:
        python plot.py <event_id>
        python plot.py CA3432611848120191010
        python plot.py CA3432611848120191010 --mode channels
        python plot.py CA3432611848120191010 --pdf

    Batch processing:
        python plot.py --batch events.txt
        python plot.py --batch CA123,CA456,CA789
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
import numpy as np

from main import DATA_EXT, layer_files, load_numpy, resolve_path

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

# Global WUI (Wildland-Urban Interface) classes
WUI_CLASSES = {
    1: ('Forest/Shrub/Wetland Intermix WUI', '#d62728'),   # Red - high risk WUI
    2: ('Forest/Shrub/Wetland Interface WUI', '#ff7f0e'),  # Orange - interface WUI
    3: ('Grassland Intermix WUI', '#e377c2'),              # Pink - grassland intermix
    4: ('Grassland Interface WUI', '#f7b6d2'),             # Light pink - grassland interface
    5: ('Non-WUI: Forest/Shrub/Wetland', '#2ca02c'),       # Green - natural forest
    6: ('Non-WUI: Grassland', '#98df8a'),                  # Light green - grassland
    7: ('Non-WUI: Urban', '#7f7f7f'),                      # Gray - urban
    8: ('Non-WUI: Other', '#c7c7c7'),                      # Light gray - other
}

# Colormaps and rendering settings per layer. Shared by the overview grid and
# the per-channel plots so both render every layer identically.
PLOT_CONFIG = {
    'elevation': {'cmap': 'terrain', 'label': 'Elevation (m)'},
    'burn_perimeter': {'cmap': 'Reds', 'label': 'Burn Perimeter'},
    'frp': {'cmap': 'hot', 'label': 'Fire Radiative Power (MW)'},
    'frp_daytime': {'cmap': 'hot', 'label': 'Daytime FRP (MW)'},
    'frp_nighttime': {'cmap': 'hot', 'label': 'Nighttime FRP (MW)'},
    'canopy_bulk_density': {'cmap': 'YlGn', 'label': 'Canopy Bulk Density',
                            # Concentrated near 0 with a long tail; clip
                            # to 98th percentile so dense canopy stands out.
                            # Pixels above the cap render in magenta so
                            # the rare high-CBD areas remain identifiable.
                            'vmax_percentile': 98, 'over_color': 'magenta'},
    'canopy_cover': {'cmap': 'Greens', 'label': 'Canopy Cover (%)',
                     # Same treatment as CBD: clip to 98th percentile and
                     # highlight the densest pixels in magenta so they
                     # don't compress the rest of the dynamic range.
                     'vmax_percentile': 98, 'over_color': 'magenta'},
    'r2': {'cmap': 'RdBu', 'label': 'Relative Humidity (%)',
           # Diverging palette centered at 50%: red = dry, blue = wet.
           'diverging_center': 50.0, 'vmin': 0.0, 'vmax': 100.0},
    'u10': {'cmap': 'coolwarm', 'label': 'Wind U (m/s)'},
    'v10': {'cmap': 'coolwarm', 'label': 'Wind V (m/s)'},
    'building_height': {'cmap': 'plasma', 'label': 'Building Height (m)',
                        # Most pixels are 0 (no buildings); clip to 98th
                        # percentile of the non-zero pixels so individual
                        # buildings remain visible. Pixels above the cap
                        # render in cyan so tall outliers stay obvious.
                        'vmax_percentile': 98, 'nonzero_only': True,
                        'over_color': 'cyan'},
    'landcover': {'cmap': 'tab20', 'label': 'Land Cover Class'},
    'recent_burn': {'cmap': 'plasma', 'label': 'Most-Recent Burn Year (NIFC IFPH)'},
    'lai': {'cmap': 'YlGn', 'label': 'LAI (m²/m²)',
            # Same long-tailed distribution as CBD/canopy cover; clip to
            # 98th percentile and flag the densest pixels in magenta.
            'vmax_percentile': 98, 'over_color': 'magenta'},
    'sentinel2_rgb': {'cmap': None, 'label': 'Satellite (RGB)'},
    'terrain_rgb': {'cmap': None, 'label': 'Terrain (Colored Shaded-Relief)'},
    'wui': {'cmap': 'tab10', 'label': 'Wildland-Urban Interface'},
    'fireline': {'cmap': 'hot', 'label': 'Fireline Intensity'},
    'fireline_max_frp': {'cmap': 'hot', 'label': 'Fireline Max Intensity'},
}


def plot_single_layer(ax, data, title, cmap='viridis', vmin=None, vmax=None, norm=None,
                      over_color=None):
    """Plot a single 2D data layer.

    If ``over_color`` is given, pixels above ``vmax`` are rendered in that
    color and the colorbar grows an upward arrow (``extend='max'``) so the
    over-range pixels remain visible after clipping.
    """
    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    extend = 'neither'
    if over_color is not None:
        cmap_obj.set_over(over_color)
        extend = 'max'
    if norm is not None:
        im = ax.imshow(data, cmap=cmap_obj, interpolation='nearest', norm=norm)
    else:
        im = ax.imshow(data, cmap=cmap_obj, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend=extend)


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

    # Place each tick at the center of its color segment so the labels line up.
    tick_locs = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(present_classes))]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([LANDCOVER_CLASSES[v][0] for v in present_classes])
    cbar.ax.tick_params(labelsize=6)


def plot_wui(ax, data, title):
    """Plot WUI (Wildland-Urban Interface) data with discrete legend.
    
    Uses custom color scheme to highlight WUI risk categories:
    - Red/Orange tones for WUI areas (intermix and interface)
    - Green tones for non-WUI vegetated areas
    - Gray tones for urban and other non-WUI areas
    """
    # Find unique values present in data
    unique_values = np.unique(data)
    unique_values = unique_values[unique_values != 0]  # Exclude NoData (0)
    
    # Filter to only known WUI classes (1-8)
    present_classes = [v for v in unique_values if v in WUI_CLASSES]
    
    if not present_classes:
        ax.imshow(data, cmap='tab10', interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        return
    
    # Create colormap for present classes
    colors = [WUI_CLASSES[v][1] for v in present_classes]
    cmap = ListedColormap(colors)
    
    # Create boundaries for discrete colormap
    bounds = present_classes + [present_classes[-1] + 1]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot
    im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    # Create a colorbar with class labels
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Center each tick on its color segment so the labels line up.
    tick_locs = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(present_classes))]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([WUI_CLASSES[v][0] for v in present_classes])
    cbar.ax.tick_params(labelsize=5)  # Smaller font for longer WUI labels


def _render_layer(ax, name: str, plot_data: np.ndarray, title: str, config: dict):
    """Render a single 2D/RGB layer onto ``ax`` using its plot config.

    Shared by the overview grid and the per-channel plots so both apply the
    same colormaps, percentile clipping, diverging norms, and discrete legends.
    """
    # RGB images (sentinel2_rgb / terrain_rgb) - 3D array with shape (H, W, 3)
    if plot_data.ndim == 3 and plot_data.shape[2] == 3:
        im = ax.imshow(plot_data)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        # Invisible colorbar to keep alignment with the other panels.
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_visible(False)
    elif name == 'landcover':
        plot_landcover(ax, plot_data, title)
    elif name == 'wui':
        plot_wui(ax, plot_data, title)
    else:
        # Resolve dynamic color limits / norms from config.
        vmin = config.get('vmin')
        vmax = config.get('vmax')
        norm = None
        pct = config.get('vmax_percentile')
        if pct is not None and np.issubdtype(plot_data.dtype, np.number):
            sample = plot_data
            if config.get('nonzero_only'):
                sample = plot_data[plot_data > 0]
            # Drop NaN nodata pixels; otherwise np.percentile returns NaN and
            # collapses the panel's color scale to a degenerate range.
            if np.issubdtype(sample.dtype, np.floating):
                sample = sample[~np.isnan(sample)]
            if sample.size > 0:
                vmax = float(np.percentile(sample, pct))
                if vmin is None:
                    vmin = 0.0
        center = config.get('diverging_center')
        if center is not None:
            lo = vmin if vmin is not None else float(np.nanmin(plot_data))
            hi = vmax if vmax is not None else float(np.nanmax(plot_data))
            # TwoSlopeNorm requires lo < center < hi.
            if lo < center < hi:
                norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
                vmin = vmax = None
        plot_single_layer(
            ax, plot_data, title,
            cmap=config['cmap'], vmin=vmin, vmax=vmax, norm=norm,
            over_color=config.get('over_color'),
        )


def plot_event_data(event_id: str, output_dir: str = 'output', show: bool = False,
                    features: list[str] | None = None, pdf: bool = False):
    """Load and plot all data for a fire event as a single overview grid.

    Args:
        event_id: Fire event ID.
        output_dir: Directory containing event subfolders.
        show: Display plot interactively.
        features: Optional list of layer names to include (e.g.,
            ['elevation', 'burn_perimeter', 'landcover']). If None, all
            available layers are plotted.
        pdf: Also save the overview as a PDF (PNG is always saved).
    """
    event_path = os.path.join(output_dir, event_id)
    
    if not os.path.exists(event_path):
        print(f"Error: Directory not found: {event_path}")
        sys.exit(1)
    
    # Find all raster layer files (task_info / coordinates are already excluded)
    data_files = [os.path.basename(p) for p in layer_files(event_path)]

    if not data_files:
        print(f"Error: No layer files found in {event_path}")
        sys.exit(1)

    print(f"Found {len(data_files)} data files for event: {event_id}")

    # Optionally filter to a user-specified subset of features
    if features:
        requested = [f.strip() for f in features if f.strip()]
        available = {os.path.splitext(f)[0]: f for f in data_files}
        missing = [name for name in requested if name not in available]
        if missing:
            print(f"Warning: requested features not found for {event_id}: {', '.join(missing)}")
            print(f"  Available: {', '.join(sorted(available.keys()))}")
        data_files = [available[name] for name in requested if name in available]
        if not data_files:
            print("Error: none of the requested features are available; nothing to plot.")
            return
        print(f"Plotting {len(data_files)} selected feature(s): {', '.join(os.path.splitext(f)[0] for f in data_files)}")
    
    # burn_perimeter renders two panels (first + last timestep), but only when
    # it actually has more than one frame. Peek at its frame count so the grid
    # reserves the extra cell only when it will be filled.
    extra_plots = 0
    if any(os.path.splitext(f)[0] == 'burn_perimeter' for f in data_files):
        bp = load_numpy(os.path.join(event_path, f'burn_perimeter{DATA_EXT}'))
        if bp.data and len(bp.data) > 1:
            extra_plots = 1
    n_plots = len(data_files) + extra_plots
    
    if n_plots == 0:
        print("No plottable data files found.")
        return
    
    # Calculate grid dimensions (5 figures per row)
    n_cols = min(5, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 4.2 * n_rows),
        constrained_layout=False,
    )
    fig.suptitle(f'Fire Event: {event_id}', fontsize=14, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_plots > 1 else [axes]
    
    plot_idx = 0  # Track current plot index separately
    for filename in data_files:
        filepath = os.path.join(event_path, filename)
        data_obj = load_numpy(filepath)
        
        name = data_obj.name
        # recent_burn_<N>_yrs share one config entry keyed on the prefix.
        config_key = 'recent_burn' if name.startswith('recent_burn_') else name
        config = PLOT_CONFIG.get(config_key, {'cmap': 'viridis', 'label': name})
        
        # Get the first frame (or only frame for static data)
        if data_obj.data and len(data_obj.data) > 0:
            # For burn_perimeter, plot both first and last timestep
            if name == 'burn_perimeter' and len(data_obj.data) > 1:
                frames_to_plot = [
                    (0, data_obj.data[0], "First"),
                    (-1, data_obj.data[-1], "Last")
                ]
            else:
                frames_to_plot = [(0, data_obj.data[0], None)]
            
            for frame_idx, plot_data, frame_label in frames_to_plot:
                # Handle non-array data (like task_info)
                if not isinstance(plot_data, np.ndarray):
                    axes[plot_idx].text(0.5, 0.5, f'{name}\n(non-array data)', 
                                  ha='center', va='center', transform=axes[plot_idx].transAxes)
                    axes[plot_idx].axis('off')
                    plot_idx += 1
                    continue
                
                # Build title with metadata
                title = config['label']
                if data_obj.unit:
                    title = f"{name} ({data_obj.unit})"
                if data_obj.native_resolution:
                    title += f" @ {data_obj.native_resolution}m"
                
                # For time series data, show frame info
                if len(data_obj.data) > 1:
                    if frame_label:
                        actual_frame_num = frame_idx + 1 if frame_idx >= 0 else len(data_obj.data)
                        title += f"\n[{frame_label} - Frame {actual_frame_num}/{len(data_obj.data)}]"
                    else:
                        title += f"\n[Frame 1/{len(data_obj.data)}]"
                
                _render_layer(axes[plot_idx], name, plot_data, title, config)

                # Print stats (only once per data file)
                if frame_label is None or frame_label == "First":
                    print(f"  {name}: shape={plot_data.shape}, "
                          f"min={plot_data.min():.2f}, max={plot_data.max():.2f}, "
                          f"frames={len(data_obj.data)}")
                
                plot_idx += 1
        else:
            axes[plot_idx].text(0.5, 0.5, f'{name}\n(no data)',
                          ha='center', va='center', transform=axes[plot_idx].transAxes)
            axes[plot_idx].axis('off')
            plot_idx += 1

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    # Tight layout with reduced padding between panels
    plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5,
                     rect=(0, 0, 1, 0.97))
    
    # Save figure (PNG always; PDF only when requested)
    output_fig = os.path.join(event_path, 'overview.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"\nSaved overview plot to: {output_fig}")
    if pdf:
        output_pdf = os.path.join(event_path, 'overview.pdf')
        # Use the same DPI for the PDF so embedded raster panels (sentinel2_rgb,
        # terrain_rgb, etc.) aren't downsampled to the default 100 dpi.
        plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
        print(f"Saved overview PDF to:  {output_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_time_series(event_id: str, layer_name: str, output_dir: str = 'output',
                     show: bool = False, data_obj=None):
    """Plot all frames of a time series layer.

    If ``data_obj`` is provided it is used directly; otherwise the layer's
    ``.npz`` file is loaded from disk. Passing a preloaded object lets callers
    (e.g. plot_all_time_series) avoid re-reading the file.
    """
    if data_obj is None:
        filepath = resolve_path(
            os.path.join(output_dir, event_id, f'{layer_name}{DATA_EXT}'))

        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

        data_obj = load_numpy(filepath)
    n_frames = len(data_obj.data)
    
    if n_frames <= 1:
        print(f"{layer_name} has only {n_frames} frame(s). Use plot_event_data instead.")
        return
    
    print(f"Plotting {n_frames} frames for {layer_name}")
    
    # Choose colormap based on layer type. Fire layers get tuned overrides;
    # anything else (e.g. weather layers now plotted by default) falls back to
    # its overview PLOT_CONFIG colormap, then to 'Reds' as a last resort.
    cmap_map = {
        'burn_perimeter': 'Reds',
        'frp': 'hot',
        'frp_daytime': 'hot',
        'frp_nighttime': 'hot',
        'fireline': 'Reds',
        'fireline_max_frp': 'hot',
    }
    config_key = 'recent_burn' if layer_name.startswith('recent_burn_') else layer_name
    config_cmap = PLOT_CONFIG.get(config_key, {}).get('cmap')
    cmap = cmap_map.get(layer_name) or config_cmap or 'Reds'
    
    # Compute shared color range across all frames for a consistent scale
    all_vals = [f for f in data_obj.data if isinstance(f, np.ndarray)]
    if all_vals:
        vmin = min(float(f.min()) for f in all_vals)
        vmax = max(float(f.max()) for f in all_vals)
    else:
        vmin, vmax = None, None
    
    n_cols = min(4, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 3.5 * n_rows))
    fig.suptitle(f'{event_id} - {layer_name} Time Series', fontsize=12)
    
    axes = axes.flatten() if n_frames > 1 else [axes]
    
    images = []
    for idx, frame in enumerate(data_obj.data):
        title = f"Frame {idx + 1}"
        if data_obj.timestamps and idx < len(data_obj.timestamps):
            title = data_obj.timestamps[idx].strftime('%Y-%m-%d %H:%M')
        
        im = axes[idx].imshow(frame, cmap=cmap, interpolation='nearest',
                              vmin=vmin, vmax=vmax)
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis('off')
        images.append(im)
    
    for idx in range(n_frames, len(axes)):
        axes[idx].axis('off')
    
    # Reserve right margin for the colorbar, then add it in a dedicated axes
    fig.subplots_adjust(right=0.88)
    if images:
        unit = f" ({data_obj.unit})" if data_obj.unit else ""
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(images[0], cax=cbar_ax, label=f"{layer_name}{unit}")
    
    # Save figure
    output_fig = os.path.join(output_dir, event_id, f'{layer_name}_timeseries.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"\nSaved time series plot to: {output_fig}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_time_series(event_id: str, output_dir: str = 'output', show: bool = False):
    """Plot a time series figure for every multi-frame layer of an event.

    Scans the event's layer files and renders one time-series PNG per layer
    that has more than one frame. Single-frame (static) layers are skipped.
    """
    event_path = os.path.join(output_dir, event_id)
    if not os.path.exists(event_path):
        print(f'Error: Directory not found: {event_path}')
        sys.exit(1)

    paths = layer_files(event_path)
    if not paths:
        print(f'Error: No layer files found in {event_path}')
        sys.exit(1)

    plotted = 0
    for path in paths:
        data_obj = load_numpy(path)
        if data_obj.data and len(data_obj.data) > 1:
            plot_time_series(event_id, data_obj.name, output_dir, show,
                             data_obj=data_obj)
            plotted += 1

    if plotted == 0:
        print(f'No multi-frame time-series layers found for {event_id}')
    else:
        print(f'\nDone — {plotted} time-series PNG(s) saved to {event_path}')


def plot_channel(event_id: str, data_obj, name: str, event_path: str):
    """Plot a single channel and save it as its own PNG."""
    # recent_burn_<N>_yrs share one config entry keyed on the prefix.
    config_key = 'recent_burn' if name.startswith('recent_burn_') else name
    config = PLOT_CONFIG.get(config_key, {'cmap': 'viridis', 'label': name})

    if not data_obj.data or len(data_obj.data) == 0:
        return

    # For burn_perimeter, plot first and last frame side by side
    if name == 'burn_perimeter' and len(data_obj.data) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'{event_id} — {config["label"]}', fontsize=13, fontweight='bold')

        for ax, (frame_idx, label) in zip(axes, [(0, 'First'), (-1, 'Last')]):
            frame = data_obj.data[frame_idx]
            if not isinstance(frame, np.ndarray):
                ax.axis('off')
                continue
            actual_num = frame_idx + 1 if frame_idx >= 0 else len(data_obj.data)
            title = f'{label} — Frame {actual_num}/{len(data_obj.data)}'
            if data_obj.native_resolution:
                title += f' @ {data_obj.native_resolution}m'
            _render_layer(ax, name, frame, title, config)
    else:
        plot_data = data_obj.data[0]
        if not isinstance(plot_data, np.ndarray):
            return

        title = config['label']
        if data_obj.unit:
            title = f'{name} ({data_obj.unit})'
        if data_obj.native_resolution:
            title += f' @ {data_obj.native_resolution}m'
        if len(data_obj.data) > 1:
            title += f'\n[Frame 1/{len(data_obj.data)}]'

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.suptitle(f'{event_id} — {config["label"]}', fontsize=13, fontweight='bold')
        _render_layer(ax, name, plot_data, title, config)

    plt.tight_layout()
    out_path = os.path.join(event_path, f'{name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_all_channels(event_id: str, output_dir: str = 'output'):
    """Plot every data channel for an event as separate PNG files."""
    event_path = os.path.join(output_dir, event_id)
    if not os.path.exists(event_path):
        print(f'Error: Directory not found: {event_path}')
        sys.exit(1)

    paths = layer_files(event_path)
    if not paths:
        print(f'Error: No layer files found in {event_path}')
        sys.exit(1)

    print(f'Plotting {len(paths)} channels for event: {event_id}')

    for path in paths:
        data_obj = load_numpy(path)
        plot_channel(event_id, data_obj, data_obj.name, event_path)

    print(f'\nDone — {len(paths)} channel PNGs saved to {event_path}')


def plot_event(event_id: str, output_dir: str = 'output', show: bool = False,
               features: list[str] | None = None, pdf: bool = False,
               mode: str = 'all'):
    """Run the requested plotting mode(s) for a single event.

    Args:
        mode: 'overview' (combined grid), 'channels' (one PNG per layer),
            'timeseries' (one figure per multi-frame layer), 'both' (overview +
            channels), or 'all' (default: overview + channels + timeseries).
    """
    if mode in ('overview', 'both', 'all'):
        plot_event_data(event_id, output_dir, show, features=features, pdf=pdf)
    if mode in ('channels', 'both', 'all'):
        plot_all_channels(event_id, output_dir)
    if mode in ('timeseries', 'all'):
        plot_all_time_series(event_id, output_dir, show)


def parse_batch_input(batch_input: str, output_dir: str = 'output') -> list[str]:
    """Parse batch input which can be a file path or comma-separated event IDs.
    
    Args:
        batch_input: Either a path to a file containing event IDs (one per line)
                     or a comma-separated string of event IDs.
        output_dir: Output directory to check for existing event data.
    
    Returns:
        List of event IDs to process.
    """
    # Check if it's a file
    if os.path.isfile(batch_input):
        print(f"Reading event IDs from file: {batch_input}")
        with open(batch_input, 'r') as f:
            event_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Found {len(event_ids)} event IDs in file")
        return event_ids
    
    # Otherwise treat as comma-separated
    event_ids = [eid.strip() for eid in batch_input.split(',') if eid.strip()]
    print(f"Parsed {len(event_ids)} event IDs from input")
    return event_ids


def plot_batch(
    event_ids: list[str],
    output_dir: str = 'output',
    timeseries: str | None = None,
    show: bool = False,
    features: list[str] | None = None,
    pdf: bool = False,
    mode: str = 'all',
) -> dict[str, bool]:
    """Plot data for multiple fire events.

    Args:
        event_ids: List of event IDs to plot.
        output_dir: Output directory containing event data.
        timeseries: If specified, plot time series for only this layer.
        show: Display plots interactively.
        features: Optional subset of layer names for the overview plot.
        pdf: Also save the overview as a PDF (PNG is always saved).
        mode: Which plotting mode(s) to run ('overview', 'channels',
            'timeseries', 'both', or 'all').

    Returns:
        Dictionary mapping event IDs to success status.
    """
    print(f"\nBatch plotting {len(event_ids)} fire events...")
    print("=" * 60)

    results: dict[str, bool] = {}
    successful = 0
    failed = 0

    for i, event_id in enumerate(event_ids, 1):
        print(f"\n[{i}/{len(event_ids)}] Processing: {event_id}")

        try:
            if timeseries:
                plot_time_series(event_id, timeseries, output_dir, show)
            else:
                plot_event(event_id, output_dir, show, features=features,
                           pdf=pdf, mode=mode)
            results[event_id] = True
            successful += 1
            print(f"✓ Completed: {event_id}")
        except Exception as e:
            results[event_id] = False
            failed += 1
            print(f"✗ Failed: {event_id} - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH PLOTTING COMPLETE")
    print(f"  Total events: {len(event_ids)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Plot exported fire event data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mutually exclusive: single event_id or batch mode
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'event_id',
        type=str,
        nargs='?',
        help='Single fire event ID to plot'
    )
    input_group.add_argument(
        '--batch',
        type=str,
        help='Batch mode: file path with event IDs (one per line) or comma-separated event IDs'
    )
    
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
        help='Plot the time series for ONLY this layer (e.g., burn_perimeter) '
             'and nothing else. By default ("all" mode) time series for every '
             'multi-frame layer are already plotted.'
    )
    parser.add_argument(
        '-s', '--show',
        action='store_true',
        help='Display the plot interactively (default: only save to file)'
    )
    parser.add_argument(
        '-f', '--features',
        type=str,
        default=None,
        help='Comma-separated list of feature/layer names to include in the '
             'overview plot (e.g., "elevation,burn_perimeter,landcover,wui"). '
             'Defaults to all available layers. Ignored when --timeseries is used.'
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['overview', 'channels', 'timeseries', 'both', 'all'],
        default='all',
        help='Which plotting mode to run: "overview" (combined grid), '
             '"channels" (one PNG per layer), "timeseries" (one figure per '
             'multi-frame layer), "both" (overview + channels), or "all" '
             '(default: everything). Ignored when --timeseries is used to '
             'select a single layer.'
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Also save the overview plot as a PDF (default: PNG only)'
    )

    args = parser.parse_args()
    
    features = (
        [f.strip() for f in args.features.split(',') if f.strip()]
        if args.features else None
    )
    
    # Batch mode or single mode
    if args.batch:
        event_ids = parse_batch_input(args.batch, args.output_dir)
        if not event_ids:
            print("Error: No valid event IDs found in batch input")
            sys.exit(1)
        plot_batch(event_ids, args.output_dir, args.timeseries, args.show,
                   features=features, pdf=args.pdf, mode=args.mode)
    else:
        # Single event mode
        event_id = args.event_id
        if not event_id:
            parser.error("Either event_id or --batch is required")

        if args.timeseries:
            plot_time_series(event_id, args.timeseries, args.output_dir, args.show)
        else:
            plot_event(event_id, args.output_dir, args.show, features=features,
                       pdf=args.pdf, mode=args.mode)


if __name__ == '__main__':
    main()
