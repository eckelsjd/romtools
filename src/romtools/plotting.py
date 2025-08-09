"""Plotting utilities.

Includes:
  - plot2d - plot 2D x-y plane data for field quantities
  - plot1d - plot 1D time/slice data for field quantities or scalar metrics
  - compare2d - helper to compare two 2d sets of data
"""
from typing import Literal, Required, NotRequired, TypedDict
from pathlib import Path
import copy

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation

from romtools.utils import get_boundary, edge_normal, relative_error

__all__ = ['plot2d', 'plot1d', 'compare2d']

ANIMATE_DEFAULT = {
    'blit': True, 
    'fps': 10, 
    'frame_skip': 1, 
    'dpi': 200
}


class CellOptions(TypedDict):
    """Options for treating 2d data as cell-centered quadrilaterals.
    
    Required options are:
    :param cells: (N, 2) array of cell-center coordinates corresponding to the arrays in `data`. Order is (x,y)
    :param vertices: (M, 2) array of vertex coordinates. Order for coordinates is (x,y)
    :param connectivity: (N, 4) array specifying vertex indices for each cell
    
    Optional options are:
    :param show_mesh: Show grid lines of mesh (must have node data and connectivity)
    :param show_cell_center: Show points at each cell center (must have cell data)
    :param show_boundary_cell: Highlight boundary cells and boundary normal vectors (default False)
    :param boundary_colors: colors for boundaries, specify as dict for different values for each boundary group
    :param group_boundary: Sorts edges by index into groups (from which boundary is selected), defaults to selecting all
    """
    cells: Required[ArrayLike]
    vertices: Required[ArrayLike]
    connectivity: Required[ArrayLike]
    show_mesh: NotRequired[bool]
    show_cell_center: NotRequired[bool]
    show_boundary_cell: NotRequired[bool]
    boundary_colors: NotRequired[dict | str]
    group_boundary: NotRequired[callable]


class PcolorOptions(TypedDict):
    """Options for treating 2d data as 2d regular mesh for pcolormesh (see plt docs for details).
    
    Required options are:
    X: (Nx,) 1d array of horizontal coordinates
    Y: (Ny,) 1d array of vertical coordinates

    Optional options are:
    pcolormesh_opts: dict of extra kwargs to pass to plt.pcolormesh
    """
    X: Required[ArrayLike]
    Y: Required[ArrayLike]
    pcolormesh_opts: NotRequired[dict]


class Frame(TypedDict):
    """A single frame of simulation data. Maps several variable names `v` to numpy arrays.
    The arrays can be any shape, so long as all variables have the same shape.
    
    For example:
       (N,)    - 1d data
       (N, 2)  - 2d unstructured mesh data
       (N, M)  - 2d structured mesh data
       etc.
    """
    v: NotRequired[ArrayLike]


def _format_time_engineering(seconds: float):
    """Helper to format times in common engineering magnitudes."""
    prefixes = [
        (1e-12, "ps"),
        (1e-9, "ns"),
        (1e-6, "μs"),
        (1e-3, "ms"),
        (1.0,  "s"),
        (1e3, "ks")
    ]

    # Find the appropriate prefix and scale
    for factor, suffix in prefixes:
        scaled = seconds / factor
        if 1 <= scaled < 1000:
            return f"{scaled:.1f} {suffix}"
    
    # Fall back to scientific notation if out of normal range
    return f"{seconds:.1e} s"


def _get_scheme(scheme: Literal['white', 'dark']):
    """Return text and background colors for given scheme."""
    match scheme.lower():
        case 'white':
            text_color = 'black'
            bg_color = 'white'
        case 'dark' | 'black':
            text_color = 'white'
            bg_color = 'black'
        case _:
            text_color = 'black'
            bg_color = 'white'
    
    return text_color, bg_color


def plot2d(data: list[Frame],
           data_opts: CellOptions | PcolorOptions,
           time: ArrayLike = None,
           labels: dict = None,
           coord_labels: list[str] = None,
           cmap: str | dict = 'jet',
           norm: str | dict = 'linear',
           grid: tuple[int, int] = None,
           subplot_size_in: tuple[float, float] = (2., 2.),
           save: str | Path = None,
           animate_opts: dict = None,
           scheme: Literal['white', 'dark'] = 'white',
           exclude: list[str] = None,
           adjust: callable = None
           ):
    """Plots 2d data on the x-y plane. Only cell-center plotting is supported via the PolyCollection.
    
    :param data: List of frames of simulation data to plot. If several frames are provided, the result will be
                 animated. If a single frame is provided, the result will be a static plot. Each frame is a dict
                 with field variable names mapped to arrays of data to plot.
    :param data_opts: Options for plotting 2d data (CellOptions and PcolorOptions supported, see their docstrings for details).
    :param time: (Nt,) array of simulation time values (only used for labeling animation plot), must be same length as data
    :param labels: Labels to show on colorbars (defaults to just using variable names)
    :param coord_labels: Will show axis labels for (x,y), otherwise will hide axes (default)
    :param cmap: the name of the colormap to use, if dict then applies different cmaps to each variable (default 'jet')
    :param norm: the name of the norm to use, if dict then applies different norms to each variable (default 'linear')
    :param grid: The shape of subplots for multiple variables. By default, will make the best square grid.
    :param subplot_size_in: Tuple (W, H) of each subplot size (inches), all subplots are set to this size
    :param save: Name of file to save to (won't save if None)
    :param animate_opts: dict with animation options alternate for (blit=True, fps=10, frame_skip=1, dpi=200)
    :param scheme: Either white (default) or dark, for setting text and background colors
    :param exclude: variables to exclude from plotting, (default none)
    :param adjust: a catch-all func for applying additional changes to the plot before animating/saving,
                   callable as adjust(fig, axs)
    """
    # Perform initial checks on all arguments
    if data is None or len(data) == 0 or len(data[0]) == 0:
        # Add empty data item to just show plain mesh
        d = [{None: None}]
    else:
        d = data

    if 'cells' in data_opts:
        plot_type = 'cell'
    elif 'X' in data_opts and 'Y' in data_opts:
        plot_type = 'pcolor'
    else:
        raise TypeError(f"Somehow the passed data_opts matches neither cell nor pcolor options.")

    if plot_type == 'cell':
        for k in CellOptions.__required_keys__:
            if k not in data_opts:
                raise TypeError(f"Required cell option '{k}' missing in data_opts.")
    
    if plot_type == 'pcolor':
        for k in PcolorOptions.__required_keys__:
            if k not in data_opts:
                raise TypeError(f"Required pcolor option '{k}' missing in data_opts.")
    
    if exclude is None:
        exclude = {}
    
    all_vars = [v for v in d[0].keys() if v not in exclude]

    if labels is None:
        labels = {}
    labels = {v: labels.get(v, v) for v in all_vars}

    if grid is None:
        num_plots = len(all_vars)
        c = int(np.ceil(np.sqrt(num_plots)))
        r = int(np.ceil(num_plots / c))
        grid = (r, c)

    if animate_opts is None:
        animate_opts = dict()
    a_opts = {k: animate_opts.get(k, v) for k, v in ANIMATE_DEFAULT.items()}

    cmaps = {v: cmap.get(v, 'jet') if isinstance(cmap, dict) else cmap for v in labels}
    norms = {v: norm.get(v, 'linear') if isinstance(norm, dict) else norm for v in labels}

    hl_color = (1, 0, 0, 0.8)  # Red
    text_color, bg_color = _get_scheme(scheme)
    
    # Setup figure, axis subplots
    fig, axs = plt.subplots(*grid, layout='tight', squeeze=False, sharex='col', sharey='row', 
                            figsize=(subplot_size_in[0]*grid[1], subplot_size_in[1]*grid[0]))
    fig.patch.set_facecolor(bg_color)

    collections = []  # for animation

    if plot_type == 'cell':
        cells = data_opts['cells']
        vertices = data_opts['vertices']
        connectivity = data_opts['connectivity']
        show_mesh = data_opts.get('show_mesh', False)
        show_cell_center = data_opts.get('show_cell_center', False)
        show_boundary_cell = data_opts.get('show_boundary_cell', False)
        boundary_colors = data_opts.get('boundary_colors', None)
        group_boundary = data_opts.get('group_boundary', None)

        coords_min = np.min(vertices, axis=0)
        coords_max = np.max(vertices, axis=0)

        quads = [[vertices[i] for i in cell] for cell in connectivity]
        boundary_edges, boundary_cells = get_boundary(connectivity)
        edge_groups = group_boundary(boundary_edges, vertices) if group_boundary is not None else {}

        if boundary_colors is None:
            boundary_colors = {}
        boundary_colors = {b: boundary_colors.get(b, text_color) if isinstance(boundary_colors, dict) else boundary_colors for 
                           b in edge_groups}

        if show_mesh:
            edgecolors = []
            for cell_idx in range(len(quads)):
                if show_boundary_cell and cell_idx in boundary_cells:
                    edgecolors.append(hl_color)
                else:
                    edgecolors.append((0.5, 0.5, 0.5, 0.3))
        else:
            edgecolors='face'
    
    elif plot_type == 'pcolor':
        coords_min = [np.min(data_opts['X']), np.min(data_opts['Y'])]
        coords_max = [np.max(data_opts['X']), np.max(data_opts['Y'])]
    
    # Loop over variables and plot
    for curr_idx, v in enumerate(all_vars):
        ax = axs.flatten()[curr_idx]
        arr = d[0][v]

        axes_visible = coord_labels is not None and len(coord_labels) == 2
        ax.autoscale(enable=False)
        ax.set_xlim([coords_min[0], coords_max[0]])
        ax.set_ylim([coords_min[1], coords_max[1]])
        ax.tick_params(axis='both', which='both', top=False, bottom=axes_visible, left=axes_visible, right=False, direction='in',
                       labelleft=axes_visible, labelbottom=axes_visible, color=text_color, labelcolor=text_color)
        ax.set_facecolor(bg_color)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_visible(axes_visible)
            ax.spines[spine].set_color(text_color)
        
        if axes_visible and curr_idx // grid[1] == grid[0] - 1:  # last row on grid
            ax.set_xlabel(coord_labels[0], color=text_color)
        if axes_visible and curr_idx % grid[1] == 0:             # first column
            ax.set_ylabel(coord_labels[1], color=text_color)
        
        # Plot using poly collection elements
        if plot_type == 'cell':
            facecolors = bg_color if v is None else None
            pc = PolyCollection(quads, array=arr, cmap=cmaps[v], norm=norms[v], edgecolors=edgecolors, facecolors=facecolors)
            ax.add_collection(pc)
            collections.append(pc)

            # Outline the boundary
            for i, edge in enumerate(boundary_edges):
                x_vals, y_vals = vertices[edge, 0], vertices[edge, 1]
                c = text_color
                for b in edge_groups:
                    if i in edge_groups[b]:
                        c = boundary_colors[b]
                        break
                ax.plot(x_vals, y_vals, color=c, linewidth=1.5)
            
            # Show normal boundary vectors
            if show_boundary_cell:
                pos = np.zeros((len(boundary_edges), 2))
                vel = np.zeros((len(boundary_edges), 2))
                for i, edge in enumerate(boundary_edges):
                    p1 = vertices[edge[0]]
                    p2 = vertices[edge[1]]
                    pos[i, :] = (p1 + p2) / 2
                    vel[i, :] = edge_normal(p1, p2, cells[boundary_cells[i]])
                ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color=hl_color)

            if show_cell_center:
                ax.scatter(cells[:, 0], cells[:, 1], color=hl_color, s=10, alpha=0.5, marker='o', linewidths=0)

        elif plot_type == 'pcolor':
            pcm = ax.pcolormesh(data_opts['X'], data_opts['Y'], arr, cmap=cmaps[v], norm=norms[v], 
                                **data_opts.get('pcolormesh_opts', {}))
            collections.append(pcm)

        if v is not None:
            cb = plt.colorbar(collections[curr_idx], ax=ax)
            cb.ax.set_ylabel(labels.get(v, v), color=text_color)
            cb.ax.tick_params(labelcolor=text_color, color=text_color)
            cb.outline.set_edgecolor(text_color)
    
    if adjust is not None:
        adjust(fig, axs)
        
    # Iterate over frames to animate
    if len(d) > 1:
        # Get global (vmin, vmax) for cbars
        for curr_idx, v in enumerate(all_vars):
            vmin = []
            vmax = []
            for i in range(len(d)):
                vmin.append(np.nanmin(d[i][v]))
                vmax.append(np.nanmax(d[i][v]))
            collections[curr_idx].set_clim(np.nanmin(vmin), np.nanmax(vmax))

        num_frames = int(len(d) / a_opts['frame_skip']) + 1  # use last one to show last time step
        
        def update(idx):
            idx_use = idx * a_opts['frame_skip'] if idx < num_frames - 1 else len(d) - 1
            if time is not None:
                fig.suptitle(f"t={_format_time_engineering(time[idx_use])}", color=text_color)

            for curr_idx, v in enumerate(all_vars):
                collections[curr_idx].set_array(d[idx_use][v])  # works for cell and pcolor
                
            return collections

        ani = FuncAnimation(fig, update, frames=num_frames, blit=a_opts['blit'], interval=1/a_opts['fps'])

        if save is not None:
            print(f"Saving animation to '{save}'")
            def _progress(i, n):
                if np.mod(i, int(0.1 * n)) == 0 or i == 0 or i == n - 1:
                    print(f'Saving frame {i+1}/{n}...')
            ani.save(Path(save), writer='ffmpeg', progress_callback=_progress, dpi=a_opts['dpi'], fps=a_opts['fps'])
        else:
            plt.show()
    
    else:
        if save is not None:
            fig.savefig(Path(save), bbox_inches='tight')
    
    return fig, axs


def plot1d(data: list[Frame],
           coord: ArrayLike,
           time: ArrayLike = None,
           labels: dict = None,
           coord_label: str = None,
           show_avg: bool = False,
           share_plot: dict[list[str]] = None,
           color: str | dict = 'r',
           norm: str | dict = 'linear',
           grid: tuple[int, int] = None,
           subplot_size_in: tuple[float, float] = (2., 2.),
           save: str | Path = None,
           animate_opts: dict = None,
           animate_coord: bool = False,
           scheme: Literal['white', 'dark'] = 'white',
           exclude: list[str] = None,
           adjust: callable = None
           ):
    """Plots 1d data against the given coords.
    
    :param data: List of frames of simulation data to plot. If several frames are provided, the result will be
                 animated. If a single frame is provided, the result will be a static plot. Each frame is a dict
                 with field variable names mapped to arrays of 1d data.
    :param coord: (N,) the x-axis coordinates to plot each var array against
    :param time: (Nt,) array of simulation time values (only used for labeling animation plot), must be same length as data
    :param labels: labels to show on y-axis (or legend for shared plots), defaults to just using var names
    :param coord_label: the x-axis coordinate label, will not show x-axis if not provided (default)
    :param show_avg: includes dashed horizontal line showing the average 1d value (default False)
    :param share_plot: sets of variables to display on same subplot, dict keys give the ylabels for the subplots,
                       each variable should only be shown on exactly one subplot
    :param color:  colors for line plots, uses same for all variables if plain string (default: red),
                   ignored for variables in shared plots (uses default color cycler)
    :param norm: the name of the y-scale to use, if dict then applies different scale to each variable
    :param grid: the shape of subplots for multiple variables, By default, makes the best square grid
    :param subplot_size_in: tuple (W, H) of each subplot size (inches), all subplots set to this size
    :param save: the name of file to save to (won't save if None)
    :param animate_opts: dict with animation options alternate for (blit=True, fps=10, frame_skip=1, dpi=200)
    :paran animate_coord: Optional, only use if a single frame is provided, will add a vertical line that moves 
                          horizontally with the x-coordinate (i.e. time), uses same animation opts
    :param scheme: either white (default) or dark, for setting text and background colors
    :param exclude: list of variables to exclude from plotting (default none)
    :param adjust: a catch-all func for applying additional changes to the plot before animating/saving,
                   callable as adjust(fig, axs)
    """
    # Perform initial checks on all arguments
    if exclude is None:
        exclude = {}

    if labels is None:
        labels = {}
    labels = {v: labels.get(v, v) for v in data[0] if v not in exclude}
    
    if share_plot is None:
        share_plot = {}
    
    all_vars = set(data[0].keys()) - set(exclude)
    for s in share_plot.values():
        all_vars = all_vars - set(s)
    num_single_vars = len(all_vars)
    num_plots = num_single_vars + len(share_plot)
    all_vars = [v for v in data[0].keys() if v in all_vars]  # keep ordered individual variables first
    for s in share_plot.values():
        all_vars.extend(s)
    
    if grid is None:
        c = int(np.ceil(np.sqrt(num_plots)))
        r = int(np.ceil(num_plots / c))
        grid = (r, c)

    if animate_opts is None:
            animate_opts = dict()
    a_opts = {k: animate_opts.get(k, v) for k, v in ANIMATE_DEFAULT.items()}

    scales = {v: norm.get(v, 'linear') if isinstance(norm, dict) else norm for v in all_vars}
    colors = {v: color.get(v, 'red') if isinstance(color, dict) else color for v in all_vars}
    text_color, bg_color = _get_scheme(scheme)

    # Setup figure, axis subplots
    fig, axs = plt.subplots(*grid, layout='tight', squeeze=False, sharex='col', 
                            figsize=(subplot_size_in[0]*grid[1], subplot_size_in[1]*grid[0]))
    fig.patch.set_facecolor(bg_color)

    lines = []   # for animation of multiple frames
    vlines = {}  # for animate coord of single frame

    groups = list(share_plot.values())

    def iter_all_vars():
        plot_idx = 0   # for plots
        group_idx = 0  # for shared groups
        share_idx = 0  # for loc in each group
        for v in all_vars:
            yield v, plot_idx, group_idx

            if plot_idx < num_single_vars:
                plot_idx += 1
            else:
                share_idx += 1
                if share_idx > len(groups[group_idx]) - 1:
                    plot_idx += 1
                    group_idx += 1
                    share_idx = 0

    for v, plot_idx, group_idx in iter_all_vars():
        ax = axs.flatten()[plot_idx]
        arr = data[0].get(v)

        ax.tick_params(axis='both', which='both', direction='in', color=text_color, labelcolor=text_color)
        ax.tick_params(axis='both', which='major', width=1, pad=5, size=6)
        ax.tick_params(axis='both', which='minor', bottom=True, left=True, width=0.8, size=3)
        ax.set_facecolor(bg_color)
        ax.set_yscale(scales.get(v))
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color(text_color)
        
        if coord_label is None:
            ax.tick_params(axis='x', which='both', labelbottom=False)
        
        if coord_label is not None and plot_idx // grid[1] == grid[0]-1:  # last row of grid
            ax.set_xlabel(coord_label, color=text_color)
        
        l = ax.plot(coord, arr, '-', label=labels[v])
        lines.append(l[0])

        if animate_coord and plot_idx not in vlines:
            lvert = ax.axvline(x=coord[0], color='r', linestyle='-', linewidth=0.5)
            vlines[plot_idx] = lvert

        if plot_idx < num_single_vars:
            l[0].set_color(colors[v])
        else:
            ax.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, fancybox=True)
        
        if show_avg:
            ax.plot(coord, np.ones(len(coord))*np.mean(arr), '--', color=l[0].get_color())

        ylabel = labels[v] if plot_idx < num_single_vars else list(share_plot.keys())[group_idx]
        
        ax.set_ylabel(ylabel, color=text_color)
    
    if adjust is not None:
        adjust(fig, axs)
    
    # Iterate over frames to animate
    if len(data) > 1:
        # Get global (ymin, ymax) for ylims
        for v, plot_idx, group_idx in iter_all_vars():
            ymin, ymax = [], []
            for i in range(len(data)):
                arr = data[i][v]
                ymin.append(np.nanmin(arr))
                ymax.append(np.nanmax(arr))
            ax = axs.flatten()[plot_idx]
            ax.set_ylim([np.nanmin(ymin), np.nanmax(ymax)])
        
        num_frames = int(len(data) / a_opts['frame_skip']) + 1

        def update(idx):
            idx_use = idx * a_opts['frame_skip'] if idx < num_frames - 1 else len(data) - 1
            if time is not None:
                fig.suptitle(f"t={_format_time_engineering(time[idx_use])}", color=text_color)
            
            for curr_idx, v in enumerate(all_vars):
                lines[curr_idx].set_ydata(data[idx_use][v])
            
            return lines
        
        ani = FuncAnimation(fig, update, frames=num_frames, blit=a_opts['blit'], interval=1/a_opts['fps'])

        if save is not None:
            print(f"Saving animation to '{save}'")
            def _progress(i, n):
                if np.mod(i, int(0.1 * n)) == 0 or i == 0 or i == n - 1:
                    print(f'Saving frame {i+1}/{n}...')
            ani.save(Path(save), writer='ffmpeg', progress_callback=_progress, dpi=a_opts['dpi'], fps=a_opts['fps'])
        else:
            plt.show()
    
    # Animate single vertical line moving over x-coord on static plot
    elif animate_coord:
        num_frames = int(len(coord) / a_opts['frame_skip']) + 1

        def update(idx):
            idx_use = idx * a_opts['frame_skip'] if idx < num_frames - 1 else len(coord) - 1
            if time is not None:
                fig.suptitle(f"t={_format_time_engineering(time[idx_use])}", color=text_color)
            
            for plot_idx in vlines:
                vlines[plot_idx].set_xdata([coord[idx_use], coord[idx_use]])
            
            return list(vlines.values())

        ani = FuncAnimation(fig, update, frames=num_frames, blit=a_opts['blit'], interval=1/a_opts['fps'])

        if save is not None:
            print(f"Saving animation to '{save}'")
            def _progress(i, n):
                if np.mod(i, int(0.1 * n)) == 0 or i == 0 or i == n - 1:
                    print(f'Saving frame {i+1}/{n}...')
            ani.save(Path(save), writer='ffmpeg', progress_callback=_progress, dpi=a_opts['dpi'], fps=a_opts['fps'])
        else:
            plt.show()

    # Static figure
    else:
        if save is not None:
            fig.savefig(Path(save), bbox_inches='tight')
    
    return fig, axs


def compare2d(first: list[list[Frame]], 
              second: list[list[Frame]],
              data_opts: CellOptions | PcolorOptions,
              show_error: bool = True, 
              variable: str | list[str] = None,
              row_labels: list[str] = None,
              col_labels: list[str] = None,
              text_offset: float = 0.05,
              text_opts: dict = None,
              tol: float = 1e-6,
              error_cmap: str = None,
              **plot2d_opts
              ):
    """Helper to compare two sets of 2d data using the plot2d function.

    Each element of `first` and `second` is frame data for a single row of comparison (see plot2d). If
    this frame data has several items (i.e. multiple frames), then it will be animated like in plot2d.
    If either `first` or `second` has only one element, then this data will be repeated on each row.
    A third column will show the relative error (pointwise) between the first two columns.
    
    :param first: The frame data for the first column. One set of frames per row.
    :param second: The frame data for the second column. One set of frames per row.
    :param data_opts: Contains mesh/coord information, assumed to be the same for all rows and both columns
    :param show_error: Whether to include pointwise error in a third column.
    :param variable: The variable to plot from each frame (defaults to first). If multiple are provided in a list, 
                     then only the first item of first/second will be selected for plotting, and each row will 
                     instead contain the data for each of these variables.
    :param row_labels: Labels for rows. Rows not labeled if None
    :param col_labels: Labels for first two columns (optionally third label for error column). Columns not labeled if None.
    :param text_offset: for row labels on the left - the amount (in figure coords) to push row labels to the left
    :param text_opts: extra text options for row and column titles (bigger fontsize, bold, etc.), default none
    :param tol: the absolute value below which pointwise relative error is ignored (rel error is unstable near 0)
    :param error_cmap: different cmap to use for error (defaults to using same as variables)
    :param plot2d_opts: Rest of the options are passed to plot2d
    """
    if text_opts is None:
        text_opts = {}

    if variable is None:
        variable = next(iter(first[0][0]))
    
    num_frames = len(first[0])  # same for all
    
    opts = copy.deepcopy(plot2d_opts)
    cmap = opts.get('cmap', {})
    norm = opts.get('norm', {})
    labels = opts.get('labels', {})
    
    num_cols = 3 if show_error else 2
    num_rows = max(len(first), len(second)) if isinstance(variable, str) else len(variable)
    opts['grid'] = (num_rows, num_cols)
    combined_frames = [{} for _ in range(num_frames)]

    vmin = []
    vmax = []

    for r in range(num_rows):
        if isinstance(variable, list):  # Only use first items for each column (rows will be variables)
            frame1 = first[0]
            frame2 = second[0]
            curr_var = variable[r]
        else:
            frame1 = first[r] if len(first) > 1 else first[0]
            frame2 = second[r] if len(second) > 1 else second[0]
            curr_var = variable
        
        # Will only work on single static frames (animations update clims over time)
        vmin.append(np.nanmin([np.nanmin(frame1[0][curr_var]), np.nanmin(frame2[0][curr_var])]))
        vmax.append(np.nanmax([np.nanmax(frame1[0][curr_var]), np.nanmax(frame2[0][curr_var])]))
        
        for j in range(num_cols):
            var_key = f'{r}_{j}'

            if j < 2:
                cmap[var_key] = cmap.get(curr_var, 'jet')
                norm[var_key] = norm.get(curr_var, 'linear')
                labels[var_key] = labels.get(curr_var, curr_var)
            else:
                cmap[var_key] = cmap.get(curr_var, 'viridis') if error_cmap is None else error_cmap
                norm[var_key] = 'log'  # always log for relative error > 0
                labels[var_key] = 'Relative error'

            for i in range(num_frames):
                row_data = [frame1[i][curr_var], frame2[i][curr_var]]
                
                if show_error:
                    row_data.append(relative_error(row_data[1], row_data[0], pointwise=True, tol=tol))
                    
                combined_frames[i][var_key] = row_data[j]
    
    def _adjust(fig, axs):
        text_color, bg_color = _get_scheme(opts.get("scheme", "dark"))
        for i in range(num_rows):
            for j in range(num_cols):
                # Merge first two column color bars (won't work on animations, just for single frames)
                if j < 2:
                    axs[i, j].collections[0].set_clim(vmin[i], vmax[i])
                
                if i == 0 and col_labels is not None:
                    axs[i, j].set_title(col_labels[j] if j < len(col_labels) else 'Relative error', color=text_color, **text_opts)
                
                if j == 0 and row_labels is not None:
                    pos = axs[i, 0].get_position()
                    x_pos = pos.x0 - text_offset
                    y_pos = 0.5 * (pos.y1 + pos.y0)
                    fig.text(x_pos, y_pos, row_labels[i], va='center', ha='right', rotation=90, color=text_color, **text_opts)

    opts['cmap'] = cmap
    opts['norm'] = norm
    opts['labels'] = labels

    fig, axs = plot2d(combined_frames, data_opts, adjust=_adjust, **opts)

    return fig, axs
