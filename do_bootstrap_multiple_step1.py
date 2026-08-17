import warnings
warnings.filterwarnings("ignore")
from importlib import reload
import awkward as ak
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from scipy.optimize import curve_fit
import os
import argparse
import json
from tqdm import tqdm
import mplhep as hep
hep.style.use("CMS")

#reload(au)
#reload(tbplt)

# Constants
BRANCHES = ["row", "col", "tot_code", "cal_code", "toa_code", "mcp_volts", "mcp_seconds", "clock_seconds", "clock_volts", "chipid", "nhits"]
ITERATIONS = 10

# Columns actually read from the preselected parquet files. Only the "_sel"
# fields are used (the diagnostic maps before any cut now read them too, see
# the heatmap_allPixels_allEvents/heatmap_CAL_CODE_mode/heatmap_TOT_CODE_mean
# and plot_2d_histograms calls in main()). Excludes "event", "nhits",
# "row_global", "col_global" (folded into row/col by do_bootstrap_preselection.py
# before writing) and the raw "row"/"col"/"toa_code"/"tot_code"/"cal_code"
# (superseded by their "_sel" counterparts everywhere in this script, other
# than the dead do1HitUniform branch). Pruning them at read time cuts
# per-chunk memory noticeably since ak.from_parquet skips those columns
# entirely instead of loading and discarding them.
PRESELECTED_COLUMNS = ["row_sel", "col_sel", "toa_code_sel", "tot_code_sel", "cal_code_sel"]

# Set from --doGlobal in main(). The preselected data already has row/col
# populated according to the choice made upstream (do_bootstrap_preselection.py's
# own --doGlobal flag); here N_PIX only needs to match that grid size so
# heatmaps/hitmaps are sized correctly: row/col go 0..15 (16 values), while
# row_global/col_global go 0..31 (32 values), matching do_bootstrap_quality.py.
DO_GLOBAL_COORDINATES = False
N_PIX                 = 16

# ============================================================================
# Helper functions
# ============================================================================

def setup_output_directories(base_output_dir="output", n_iterations=10):
    """
    Create organized output directory structure

    Args:
        base_output_dir: Base output directory name
        n_iterations: Number of bootstrap iterations

    Returns:
        Dictionary with paths to different output subdirectories
    """
    # Create main directories
    dirs = {
        'base': base_output_dir,
        'heatmaps': os.path.join(base_output_dir, 'heatmaps'),
        'twc': os.path.join(base_output_dir, 'TWC'),
        'twc_iterations': [],
        'distributions': os.path.join(base_output_dir, 'distributions'),
        'final': os.path.join(base_output_dir, 'final'),
        'iterations': [],
        'step1': os.path.join(base_output_dir, 'step1'),
        'step2': os.path.join(base_output_dir, 'step2'),
        'step3': os.path.join(base_output_dir, 'step3')
    }

    # Create base directories
    os.makedirs(dirs['heatmaps'], exist_ok=True)
    os.makedirs(dirs['twc'], exist_ok=True)
    os.makedirs(dirs['distributions'], exist_ok=True)
    os.makedirs(dirs['final'], exist_ok=True)
    os.makedirs(dirs['step1'], exist_ok=True)
    os.makedirs(dirs['step2'], exist_ok=True)
    os.makedirs(dirs['step3'], exist_ok=True)

    # Create iteration-specific directories
    for i in range(n_iterations):
        iter_dir = os.path.join(base_output_dir, f'iter_{i:02d}')
        os.makedirs(iter_dir, exist_ok=True)
        dirs['iterations'].append(iter_dir)
        twc_iter_dir = os.path.join(dirs['twc'], f'iter_{i:02d}')
        os.makedirs(twc_iter_dir, exist_ok=True)
        dirs['twc_iterations'].append(twc_iter_dir)

    return dirs


def plot_twc_rms_heatmaps(fit_records, output_dir, layers, n_iterations,
                          tot_min=0.0, tot_max=12.5, n_tot_points=251):
    """Plot pixel-to-module TWC differences from the saved fit records.

    The per-iteration maps compare the sum of a pixel's corrections through
    the current iteration with the corresponding module-average sum.  One
    final signed net-change map compares the total pixel-specific correction
    with the total correction obtained by using the module average at every
    step.  All curve comparisons use one common physical-TOT grid.
    """
    if not fit_records:
        return
    if not tot_max > tot_min:
        raise ValueError("TWC RMS TOT maximum must be greater than its minimum")
    if n_tot_points < 2:
        raise ValueError("TWC RMS TOT grid must contain at least two points")

    os.makedirs(output_dir, exist_ok=True)
    tot_grid = np.linspace(tot_min, tot_max, n_tot_points)
    records = {
        (int(r['layer']), int(r['row']), int(r['col']), int(r['iteration'])):
            np.array([r['a'], r['b'], r['c']], dtype=float)
        for r in fit_records
    }
    pixels_by_layer = {
        layer: sorted({(key[1], key[2]) for key in records if key[0] == layer})
        for layer in layers
    }

    cumulative = {}
    for layer in layers:
        for row, col in pixels_by_layer[layer]:
            running = np.zeros(3, dtype=float)
            complete = True
            for iteration in range(n_iterations):
                coeffs = records.get((layer, row, col, iteration))
                if coeffs is None:
                    complete = False
                if not complete:
                    continue
                running = running + coeffs
                cumulative[(layer, row, col, iteration)] = running.copy()

    def make_maps(iteration, coefficient_source):
        maps = []
        for layer in layers:
            curves = []
            curve_pixels = []
            for row, col in pixels_by_layer[layer]:
                coeffs = coefficient_source.get((layer, row, col, iteration))
                if coeffs is not None:
                    curves.append(np.polyval(coeffs, tot_grid))
                    curve_pixels.append((row, col))

            layer_map = np.full((N_PIX, N_PIX), np.nan)
            if curves:
                curves = np.asarray(curves)
                module_average = np.mean(curves, axis=0)
                rms_values = np.sqrt(np.mean((curves - module_average) ** 2, axis=1))
                for (row, col), value in zip(curve_pixels, rms_values):
                    if 0 <= row < N_PIX and 0 <= col < N_PIX:
                        layer_map[row, col] = value
            maps.append(layer_map)
        return maps

    def make_final_net_change_maps(iteration):
        maps = []
        for layer in layers:
            curves = []
            curve_pixels = []
            for row, col in pixels_by_layer[layer]:
                coeffs = cumulative.get((layer, row, col, iteration))
                if coeffs is not None:
                    curves.append(np.polyval(coeffs, tot_grid))
                    curve_pixels.append((row, col))

            layer_map = np.full((N_PIX, N_PIX), np.nan)
            if curves:
                curves = np.asarray(curves)
                module_average = np.mean(curves, axis=0)
                # Preserve the sign: positive means the pixel-specific total
                # correction is larger than the module-average total.
                differences = np.mean(curves - module_average, axis=1)
                for (row, col), value in zip(curve_pixels, differences):
                    if 0 <= row < N_PIX and 0 <= col < N_PIX:
                        layer_map[row, col] = value
            maps.append(layer_map)
        return maps

    def draw_maps(maps, filename, title, signed=False):
        finite_parts = [m[np.isfinite(m)] for m in maps if np.any(np.isfinite(m))]
        finite_values = np.concatenate(finite_parts) if finite_parts else np.array([])
        vmax = (np.max(np.abs(finite_values)) if signed else np.max(finite_values)) \
            if finite_values.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        vmin = -vmax if signed else 0
        cmap = 'coolwarm' if signed else 'viridis'
        scale = 4 if DO_GLOBAL_COORDINATES else 1
        fig, axes = plt.subplots(1, len(layers), figsize=(33 * scale, 11 * scale),
                                 squeeze=False)
        axes = axes[0]
        image = None
        for index, (layer, layer_map) in enumerate(zip(layers, maps)):
            image = axes[index].matshow(layer_map, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[index].set_title(f'Layer {layer}', fontsize=28 * scale)
            axes[index].set_xlabel('Column', fontsize=24 * scale)
            if index == 0:
                axes[index].set_ylabel('Row', fontsize=24 * scale)
            axes[index].tick_params(labelsize=20 * scale)
            if DO_GLOBAL_COORDINATES:
                axes[index].axhline(15.5, color='red', linewidth=4)
                axes[index].axvline(15.5, color='red', linewidth=4)
        fig.suptitle(title, fontsize=28 * scale)
        cbar = fig.colorbar(image, ax=axes)
        colorbar_label = ('Mean total correction difference [ns]' if signed
                          else 'Unweighted RMS TWC difference [ns]')
        cbar.set_label(colorbar_label, fontsize=22 * scale)
        cbar.ax.tick_params(labelsize=20 * scale)
        fig.savefig(filename + '.png', dpi=150, bbox_inches='tight')
        fig.savefig(filename + '.pdf', bbox_inches='tight')
        plt.close(fig)

    for iteration in range(n_iterations):
        iteration_dir = os.path.join(output_dir, f'iter_{iteration:02d}')
        os.makedirs(iteration_dir, exist_ok=True)
        draw_maps(
            make_maps(iteration, cumulative),
            os.path.join(iteration_dir, 'twc_rms_cumulative'),
            f'Cumulative pixel TWC vs module-average TWC, iteration {iteration}'
        )

    final_iteration = n_iterations - 1
    draw_maps(
        make_final_net_change_maps(final_iteration),
        os.path.join(output_dir, 'twc_net_change'),
        'Net total correction: pixel TWC minus module-average TWC',
        signed=True,
    )

    with open(os.path.join(output_dir, 'twc_rms_grid.json'), 'w') as f:
        json.dump({
            'tot_min_ns': float(tot_min), 'tot_max_ns': float(tot_max),
            'n_tot_points': int(n_tot_points), 'weighting': 'uniform',
            'cumulative_definition': 'sum of fits from iteration 0 through iteration i',
            'net_change_definition': ('mean over TOT of the cumulative pixel-specific '
                                      'correction minus the cumulative module-average correction'),
        }, f, indent=2)

def mode_cal(cal):
    x = ak.to_numpy(ak.flatten(cal, axis=None)).astype(float)
    counts, edges = np.histogram(x, bins=200)
    return (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2.0

def func_lineal(x, a, b):
    """Linear function for fitting"""
    return a * x + b

def func_cubic(x, a, b, c, d):
    """Cubic function for fitting"""
    return a * x**3 + b * x**2 + c * x + d

def func_inverse(x, p0, p1, p2):
    """Inverse polynomial: p0 + p1/x + p2/x^2"""
    return p0 + p1 / x + p2 / x**2

def gaussian(x, amp, mean, sigma):
    """Gaussian function for fitting"""
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

def double_gaussian(x, amp, frac, mean1, mean2, sigma1, sigma2):
    """Double gaussian function for fitting"""
    return amp* frac * np.exp(-(x - mean1)**2 / (2 * sigma1**2)) + amp * (1 - frac) * np.exp(-(x - mean2)**2 / (2 * sigma2**2))


def eval_lineal(x, coeffs):
    """Evaluate linear polynomial with given coefficients"""
    return coeffs[0] * x + coeffs[1]

def eval_cubic(x, coeffs):
    """Evaluate cubic polynomial with given coefficients"""
    return coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]

#def eval_gaussian_mixture(x, coeffs, scale):
#    y = np.zeros_like(x, dtype=float)
#    for i in range(0, len(coeffs), 3):
#        amp, mu, sigma = coeffs[i], coeffs[i+1], coeffs[i+2]
#        y += scale * amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
#    return y

def eval_gaussian_mixture(x, coeffs, scale):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(coeffs), 3):
        amp, mu, sigma = coeffs[i], coeffs[i+1], coeffs[i+2]
        y += scale * amp / (sigma * np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y

def create_dt_tot_dict():
    """Create nested dictionary structure for dt-tot data"""
    return {a: {b: {'dt': [], 'tot': []} for b in range(N_PIX)} for a in range(N_PIX)}

def find_all_crossings(xs, ys, level):
    crossings = []
    for i in range(len(ys) - 1):
        if (ys[i] - level) * (ys[i+1] - level) < 0:
            #t = (level - ys[i]) / (ys[i+1] - ys[i])
            #crossings.append(xs[i] + t * (xs[i+1] - xs[i]))
            crossings.append(xs[i])
    #print("Crossings found: ", crossings)
    return crossings

def compute_fwhm(xs, ys):
    half_max = max(ys) / 2
    crossings = find_all_crossings(xs, ys, level=half_max)

    if len(crossings) < 2:
        raise ValueError("Less than 2 crossings.")

    return crossings[-1] - crossings[0], crossings[0], crossings[-1]

def plot_2d_histograms(events_list, x_field, y_field, xlabel_base, ylabel_base, filename, bins=100, dpi=150):
    """
    Create side-by-side 2D histograms for three layers

    Args:
        events_list: List of three event arrays [events0, events1, events2]
        x_field: Field name for x-axis
        y_field: Field name for y-axis
        xlabel_base: Base label for x-axis (layer index will be added)
        ylabel_base: Base label for y-axis (layer index will be added)
        filename: Output filename
        bins: Number of bins for histogram
        dpi: DPI for saved figure
    """
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    for i, events in enumerate(events_list):
        # axis=None: the trigger layer's "_sel" fields are depth-1 option
        # scalars (from ak.firsts in preselection), not depth-2 jagged lists
        # like the other two layers, so a fixed axis=1 flatten would fail.
        x_data = ak.to_numpy(ak.flatten(getattr(events, x_field), axis=None))
        y_data = ak.to_numpy(ak.flatten(getattr(events, y_field), axis=None))
        ax[i].hist2d(x_data, y_data, bins=bins, cmap='viridis')
        ax[i].set_xlabel(rf"${xlabel_base}_{{{i}}}$")
        ax[i].set_ylabel(rf"${ylabel_base}_{{{i}}}$")

    fig.savefig(f"{filename}.png", dpi=dpi)
    plt.close(fig)

def fit_histogram_gaussian(data, bins=40, range_tuple=(-1, 1), fitType="Gaussian"):
    """
    Fit a Gaussian to histogram data

    Args:
        data: Input data array
        bins: Number of bins
        range_tuple: Histogram range

    Returns:
        popt: Optimal parameters (amp, mean, sigma)
        pcov: Covariance matrix
        bin_centers: Bin centers
        counts: Histogram counts
    """
    counts, bin_edges = np.histogram(data, bins=bins, range=range_tuple)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    amp0   = max(counts)
    mean0  = bin_centers[np.argmax(counts)]
    # sigma: half-width
    half_max = amp0 / 2
    above = bin_centers[counts > half_max]
    sigma0 = (above[-1] - above[0]) / 2 if len(above) > 1 else np.std(data)

    if fitType=="Gaussian":
        # Initial guess for Gaussian parameters
        p0 = [amp0, mean0, sigma0]
    
        # Fit Gaussian
        popt, pcov = curve_fit(
            gaussian, 
            bin_centers.to_numpy(), 
            counts.to_numpy(), 
            p0=p0,
            bounds=(
                [0, range_tuple[0], 0],
                [np.inf, range_tuple[1], range_tuple[1]-range_tuple[0]]
            )
        )

    elif fitType=="DoubleGaussian":
        # Initial guess for Gaussian parameters
        p0 = [amp0, 0.95, mean0, mean0, sigma0, 6*sigma0]
    
        # Fit Double Gaussian
        popt, pcov = curve_fit(
            double_gaussian, 
            bin_centers.to_numpy(), 
            counts.to_numpy(), 
            p0=p0
        )

    return popt, pcov, bin_centers, counts

from sklearn.mixture import GaussianMixture

def fit_gaussian_mixture(data, n_gaussians=2, bins=40, range_tuple=(-1, 1)):
    mask = (data >= range_tuple[0]) & (data <= range_tuple[1])
    data_clipped = data[mask]

    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='full')
    gmm.fit(data_clipped.reshape(-1, 1))

    bin_width = (range_tuple[1] - range_tuple[0]) / bins
    scale = len(data_clipped) * bin_width

    popt = []
    for i in range(n_gaussians):
        amp   = gmm.weights_[i]
        mu    = gmm.means_[i, 0]
        sigma = np.sqrt(gmm.covariances_[i, 0, 0])
        popt += [amp, mu, sigma]

    return np.array(popt), gmm, scale


def plot_gaussian_mixture(ax, popt, scale, x_range=(-1, 1)):
    x = np.linspace(*x_range, 500)
    y_total = np.zeros_like(x)

    for i in range(0, len(popt), 3):
        amp, mu, sigma = popt[i], popt[i+1], popt[i+2]
        y = scale * amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y_total += y
        ax.plot(x, y, linestyle='--', label=f'G{i//3 + 1}: μ={mu:.3f}, σ={sigma:.3f}')

    ax.plot(x, y_total, color='black', linewidth=2, label='Total')


def load_preselected_data(presel_dir):
    """
    Load preselected data from parquet files

    Args:
        presel_dir: Directory containing preselected parquet files

    Returns:
        Dictionary with layer events
    """
    print(f"Loading preselected data from {presel_dir}")
    layers = {}

    for layer in range(3):
        # Find all chunk files for this layer
        chunk_files = sorted(glob(f"{presel_dir}/layer{layer}_chunk_*.parquet"))
        print(f"Layer {layer}: Loading {len(chunk_files)} chunks...")

        # Load all chunks
        chunks = []
        for chunk_file in chunk_files:
            chunk = ak.from_parquet(chunk_file, columns=PRESELECTED_COLUMNS)
            chunks.append(chunk)
            print(f"  Loaded {chunk_file}: {len(chunk)} events")

        # Concatenate all chunks
        if len(chunks) > 0:
            layers[layer] = {"events": ak.concatenate(chunks)}
            print(f"Layer {layer} total events: {len(layers[layer]['events'])}")
        else:
            print(f"WARNING: No data found for layer {layer}")
            layers[layer] = {"events": None}

    return layers

def plotLayerMaps(name, eventsi, eventsj, eventsk, mc='viridis', row_field='row', col_field='col'):
    """
    Plot heatmaps for all three layers

    Args:
        name: Output filename (without extension)
        events0, events1, events2: Event data for each layer
        mc: Matplotlib colormap
        row_field, col_field: field names to read pixel coordinates from
    """
    # Create histograms for each layer
    histograms = []
    events_list = [eventsi, eventsj, eventsk]

    for events in events_list:
        hist = np.zeros((N_PIX, N_PIX), dtype=int)
        # axis=None: the trigger layer's "_sel" fields are depth-1 option
        # scalars, not depth-2 jagged lists like the other two layers.
        np.add.at(hist, (ak.flatten(events[row_field], axis=None), ak.flatten(events[col_field], axis=None)), 1)
        histograms.append(hist)

    # Configure colormap and text color
    if mc == 'viridis':
        mc = plt.get_cmap('viridis')
        cp = 'w'
    else:
        cp = 'k'

    maxcmap = max(np.max(h) for h in histograms)

    # Same scaling/style as plot_hit_map() in do_bootstrap_quality.py, so
    # doGlobal hitmaps look consistent across the two scripts.
    scale = 4 if DO_GLOBAL_COORDINATES else 1

    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(33 * scale, 11 * scale))

    # Plot each layer (reversed order: 2, 1, 0)
    idxs = ['i', 'j', 'k']
    for _,idx in enumerate(idxs):
        hist = histograms[_]

        ax[_].set_title(f"Layer %s"%(idx), fontsize=28 * scale)
        cax = ax[_].matshow(hist, cmap=mc, vmin=0, vmax=maxcmap)

        # Add text annotations
        for i in range(N_PIX):
            for j in range(N_PIX):
                if int(hist[i, j]):
                    ax[_].text(j, i, int(hist[i, j]),
                                     ha="center", va="center",
                                     color=cp, fontsize=6 * scale, fontweight='bold')

        # Set labels
        if _ == 0:
            ax[_].set_ylabel(r'$Row$', fontsize=24 * scale)
        ax[_].set_xlabel(r'$Column$', fontsize=24 * scale)
        ax[_].tick_params(labelsize=20 * scale)
        if DO_GLOBAL_COORDINATES:
            ax[_].axhline(15.5, color='red', linewidth=4)
            ax[_].axvline(15.5, color='red', linewidth=4)

    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=20 * scale)
    fig.savefig(f"{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_mode_map(events):
    def to_flat(arr):
        if arr.ndim == 1:
            return ak.to_numpy(arr)
        else:
            return ak.to_numpy(ak.flatten(arr))

    rows = to_flat(events.row_sel)
    cols = to_flat(events.col_sel)
    vals = to_flat(events.cal_code_sel)

    pixel_vals = {}
    for r, c, v in zip(rows, cols, vals):
        pixel_vals.setdefault((int(r), int(c)), []).append(int(v))

    mode_map = np.full((N_PIX, N_PIX), np.nan)
    for (r, c), vlist in pixel_vals.items():
        varray = np.array(vlist)
        unique, counts = np.unique(varray, return_counts=True)
        mode_map[r, c] = unique[np.argmax(counts)]  # <- valor real, no índice

    return mode_map

def plotLayerMaps_mode(name, eventsi, eventsj, eventsk, variable, mc='viridis', row_field='row', col_field='col'):
    """
    Plot heatmaps for all three layers showing the mode of a variable per pixel.

    Args:
        name: Output filename (without extension)
        eventsi, eventsj, eventsk: Event data for each layer
        variable: Field name to compute the mode of (e.g. 'cal_code')
        mc: Matplotlib colormap
        row_field, col_field: field names to read pixel coordinates from
    """
    events_list = [eventsi, eventsj, eventsk]
    maps = []

    for events in events_list:
        value_map = np.full((N_PIX, N_PIX), np.nan)
        # axis=None: the trigger layer's "_sel" fields are depth-1 option
        # scalars, not depth-2 jagged lists like the other two layers.
        rows = ak.to_numpy(ak.flatten(events[row_field], axis=None))
        cols = ak.to_numpy(ak.flatten(events[col_field], axis=None))
        vals = ak.to_numpy(ak.flatten(events[variable], axis=None))

        for r, c, v in zip(rows, cols, vals):
            # collect all values per pixel then take mode
            pass

        # Use a dict to accumulate values per pixel
        pixel_vals = {}
        for r, c, v in zip(rows, cols, vals):
            key = (r, c)
            if key not in pixel_vals:
                pixel_vals[key] = []
            pixel_vals[key].append(v)

        for (r, c), vlist in pixel_vals.items():
            counts = np.bincount(np.array(vlist, dtype=int))
            value_map[r, c] = np.argmax(counts)

        maps.append(value_map)

    if mc == 'viridis':
        cmap = plt.get_cmap('viridis')
        cp = 'w'
    else:
        cmap = plt.get_cmap(mc)
        cp = 'k'

    valid_vals = np.concatenate([m[~np.isnan(m)] for m in maps])
    vmin = np.min(valid_vals) if len(valid_vals) > 0 else 0
    vmax = np.max(valid_vals) if len(valid_vals) > 0 else 1

    # Same scaling/style as plot_hit_map() in do_bootstrap_quality.py, so
    # doGlobal heatmaps look consistent across the two scripts.
    scale = 4 if DO_GLOBAL_COORDINATES else 1

    fig, ax = plt.subplots(1, 3, figsize=(33 * scale, 11 * scale))
    idxs = ['i', 'j', 'k']

    for _, idx in enumerate(idxs):
        value_map = maps[_]
        cmap.set_bad(color='#94a4a2')

        ax[_].set_title(f"Layer {idx} — mode({variable})", fontsize=28 * scale)
        cax = ax[_].matshow(value_map, cmap=cmap, vmin=vmin, vmax=vmax)

        for i in range(N_PIX):
            for j in range(N_PIX):
                if not np.isnan(value_map[i, j]):
                    ax[_].text(j, i, f"{int(value_map[i, j])}",
                               ha="center", va="center",
                               color=cp, fontsize=6 * scale, fontweight='bold')

        if _ == 0:
            ax[_].set_ylabel(r'$Row$', fontsize=24 * scale)
        ax[_].set_xlabel(r'$Column$', fontsize=24 * scale)
        ax[_].tick_params(labelsize=20 * scale)
        if DO_GLOBAL_COORDINATES:
            ax[_].axhline(15.5, color='red', linewidth=4)
            ax[_].axvline(15.5, color='red', linewidth=4)

    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=20 * scale)
    fig.savefig(f"{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def plotLayerMaps_mean(name, eventsi, eventsj, eventsk, variable, mc='viridis', row_field='row', col_field='col'):
    """
    Plot heatmaps for all three layers showing the mean of a variable per pixel.

    Args:
        name: Output filename (without extension)
        eventsi, eventsj, eventsk: Event data for each layer
        variable: Field name to compute the mean of (e.g. 'toa_code')
        mc: Matplotlib colormap
        row_field, col_field: field names to read pixel coordinates from
    """
    events_list = [eventsi, eventsj, eventsk]
    maps = []

    for events in events_list:
        value_map = np.full((N_PIX, N_PIX), np.nan)
        # axis=None: the trigger layer's "_sel" fields are depth-1 option
        # scalars, not depth-2 jagged lists like the other two layers.
        rows = ak.to_numpy(ak.flatten(events[row_field], axis=None))
        cols = ak.to_numpy(ak.flatten(events[col_field], axis=None))
        vals = ak.to_numpy(ak.flatten(events[variable], axis=None))

        pixel_vals = {}
        for r, c, v in zip(rows, cols, vals):
            pixel_vals.setdefault((r, c), []).append(v)

        for (r, c), vlist in pixel_vals.items():
            value_map[r, c] = np.mean(vlist)

        maps.append(value_map)

    if mc == 'viridis':
        cmap = plt.get_cmap('viridis')
        cp = 'w'
    else:
        cmap = plt.get_cmap(mc)
        cp = 'k'

    valid_vals = np.concatenate([m[~np.isnan(m)] for m in maps])
    vmin = np.min(valid_vals) if len(valid_vals) > 0 else 0
    vmax = np.max(valid_vals) if len(valid_vals) > 0 else 1

    # Same scaling/style as plot_hit_map() in do_bootstrap_quality.py, so
    # doGlobal heatmaps look consistent across the two scripts.
    scale = 4 if DO_GLOBAL_COORDINATES else 1

    fig, ax = plt.subplots(1, 3, figsize=(33 * scale, 11 * scale))
    idxs = ['i', 'j', 'k']

    for _, idx in enumerate(idxs):
        value_map = maps[_]
        cmap.set_bad(color='#94a4a2')

        ax[_].set_title(f"Layer {idx} — mean({variable})", fontsize=28 * scale)
        cax = ax[_].matshow(value_map, cmap=cmap, vmin=vmin, vmax=vmax)

        for i in range(N_PIX):
            for j in range(N_PIX):
                if not np.isnan(value_map[i, j]):
                    ax[_].text(j, i, f"{value_map[i, j]:.0f}",
                               ha="center", va="center",
                               color=cp, fontsize=6 * scale, fontweight='bold')

        if _ == 0:
            ax[_].set_ylabel(r'$Row$', fontsize=24 * scale)
        ax[_].set_xlabel(r'$Column$', fontsize=24 * scale)
        ax[_].tick_params(labelsize=20 * scale)
        if DO_GLOBAL_COORDINATES:
            ax[_].axhline(15.5, color='red', linewidth=4)
            ax[_].axvline(15.5, color='red', linewidth=4)

    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=20 * scale)
    fig.savefig(f"{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_toa_tot_cal_distributions(events_i, events_j, events_k, layer_i, layer_j, layer_k,
                                   output_dir, variable="toa_code", name="distribution",
                                   cuts_i=None, cuts_j=None, cuts_k=None):
    """
    Plot distribution of a variable for all events in layers i, j, k.
    Computes combined cut efficiency (all 3 layers pass simultaneously) if cuts are provided.

    Args:
        variable  : "toa_code", "tot_code" or "cal_code"
        name      : output filename prefix
        cuts_i/j/k: None | float | (lo, hi)
    """
    COLORS = ['#3f90da', '#ffa90e', '#bd1f01']

    def parse_cuts(cuts):
        if cuts is None:                   return []
        if isinstance(cuts, (int, float)): return [cuts]
        return list(cuts)

    def to_np(arr):
        return ak.to_numpy(ak.flatten(arr, axis=None)).astype(float)

    def compute_efficiency(data_i, data_j, data_k, cuts_i, cuts_j, cuts_k):
        if all(len(c) == 0 for c in [cuts_i, cuts_j, cuts_k]):
            return None
        def mask(data, cuts):
            if len(cuts) == 0:    return np.ones(len(data), dtype=bool)
            if len(cuts) == 1:    return data > cuts[0]
            return (data > cuts[0]) & (data < cuts[1])
        combined = mask(data_i, cuts_i) & mask(data_j, cuts_j) & mask(data_k, cuts_k)
        return 100.0 * np.sum(combined) / len(data_i) if len(data_i) > 0 else 0.0

    layers    = [events_i,  events_j,  events_k]
    layer_ids = [layer_i,   layer_j,   layer_k]
    all_cuts  = [parse_cuts(cuts_i), parse_cuts(cuts_j), parse_cuts(cuts_k)]

    range_defaults = {
        "toa_code": (0, 750),
        "tot_code": (0, 250),
        "cal_code": (0, 300),
    }
    plot_range = range_defaults.get(variable, (None, None))

    data_i = to_np(events_i[variable])
    data_j = to_np(events_j[variable])
    data_k = to_np(events_k[variable])

    eff_combined = compute_efficiency(data_i, data_j, data_k, all_cuts[0], all_cuts[1], all_cuts[2])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    #fig.suptitle(f"{variable} — {name}", fontsize=16)

    for ax, data, layer_idx, color, cuts in zip(axes, [data_i, data_j, data_k], layer_ids, COLORS, all_cuts):
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Entries")

        counts, edges = np.histogram(data, bins=50,
                                     range=plot_range if plot_range[0] is not None
                                           else (data.min(), data.max()))
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.step(centers, counts, where="mid", color=color, linewidth=1.5)
        ax.fill_between(centers, counts, step="mid", alpha=0.15, color=color)

        for cut_val in cuts:
            ax.axvline(cut_val, color="red", linewidth=1.8, linestyle="--")

        if eff_combined is not None:
            cut_str = f"({cuts[0]:.0f}, {cuts[1]:.0f})" if len(cuts) == 2 else f"> {cuts[0]:.0f}" if len(cuts) == 1 else ""
            ax.text(0.97, 0.95, f"cut {cut_str}\nefficiency = {eff_combined:.1f}%",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=11, color="red",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.tight_layout()
    out = os.path.join(output_dir, f"{name}_{variable}")
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}.png")
    if eff_combined is not None:
        print(f"  Combined efficiency (all 3 layers): {eff_combined:.2f}%")


def plot_correction_fit(dtoa_data, tot_data, tw_corrected, filename, useCubic=True, useTWC=True):

        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        #
        tot_axes = [
            np.linspace(min(tot_data[0]) - 0.5, max(tot_data[0]) + 0.5, 100),
            np.linspace(min(tot_data[1]) - 0.5, max(tot_data[1]) + 0.5, 100),
            np.linspace(min(tot_data[2]) - 0.5, max(tot_data[2]) + 0.5, 100)
        ]
        #
        if useTWC:
            # Same tools as twc.py's three_board_iterative_timewalk_correction: np.polyfit + np.poly1d
            fits = [
                np.poly1d(tw_corrected[0])(tot_axes[0]),
                np.poly1d(tw_corrected[1])(tot_axes[1]),
                np.poly1d(tw_corrected[2])(tot_axes[2])
            ]
        elif useCubic:
            fits = [
                eval_cubic(tot_axes[0], tw_corrected[0]),
                eval_cubic(tot_axes[1], tw_corrected[1]),
                eval_cubic(tot_axes[2], tw_corrected[2])
            ]
        else:
            fits = [
                func_lineal(tot_axes[0], *tw_corrected[0]),
                func_lineal(tot_axes[1], *tw_corrected[1]),
                func_lineal(tot_axes[2], *tw_corrected[2])
            ]
        for idx in range(3):
            ax[idx].scatter(tot_data[idx], dtoa_data[idx])
            ax[idx].set_title(f"Layer {idx}")
            ax[idx].set_ylim([min(dtoa_data[idx]) - 1., max(dtoa_data[idx]) + 1.])
            ax[idx].set_xlabel(rf"$TOT_{{{idx}}}$")
            ax[idx].plot(tot_axes[idx], fits[idx], c='r')
        fig.savefig(filename)
        plt.close(fig)


def fit_deltaT(T_ij, T_jk, T_ki, fitType, outDir, suffix, plot=True):

    # Fit Gaussians to Tij distributions
    T_data = [T_ij, T_jk, T_ki]
    fit_results = []
    
    fitType = "GaussianMixture"
    for T in T_data:
        if (fitType=="Gaussian") or (fitType=="DoubleGaussian"):
            popt, pcov, bin_centers, counts = fit_histogram_gaussian(T, range_tuple=(np.mean(T) - 4.*np.std(T), np.mean(T) + 4.*np.std(T)), fitType=fitType)
            fit_results.append((popt, pcov, bin_centers, counts))
        elif fitType=="GaussianMixture":
            popt, gmm, scale = fit_gaussian_mixture(ak.to_numpy(T), n_gaussians=3, bins=25, range_tuple=(np.mean(T) - 4.*np.std(T), np.mean(T) + 4.*np.std(T)))
            fit_results.append((popt, gmm, [], [], scale))
    
    # Extract fit parameters
    scale_ij = -1
    scale_jk = -1
    scale_ki = -1
    if fitType=="GaussianMixture":
        scale_ij = fit_results[0][4]
        scale_jk = fit_results[1][4]
        scale_ki = fit_results[2][4]
        popt_ij = fit_results[0][0]
        popt_jk = fit_results[1][0]
        popt_ki = fit_results[2][0]
    else:
        popt_ij, pcov_ij, bin_centers_ij, counts_ij = fit_results[0]
        popt_jk, pcov_jk, bin_centers_jk, counts_jk = fit_results[1]
        popt_ki, pcov_ki, bin_centers_ki, counts_ki = fit_results[2]
    
    if fitType=="Gaussian":
        amp_ij, mean_ij, sigma_ij = popt_ij
        amp_jk, mean_jk, sigma_jk = popt_jk
        amp_ki, mean_ki, sigma_ki = popt_ki
        perr_ij = np.sqrt(np.diag(pcov_ij))
        perr_jk = np.sqrt(np.diag(pcov_jk))
        perr_ki = np.sqrt(np.diag(pcov_ki))
    elif fitType=="DoubleGaussian":
        amp_ij, frac_ij, mean1_ij, mean2_ij, sigma_ij, sigma2_ij = popt_ij
        amp_jk, frac_jk, mean1_jk, mean2_jk, sigma_jk, sigma2_jk = popt_jk
        amp_ki, frac_ki, mean1_ki, mean2_ki, sigma_ki, sigma2_ki = popt_ki
        perr_ij = np.sqrt(np.diag(pcov_ij))
        perr_jk = np.sqrt(np.diag(pcov_jk))
        perr_ki = np.sqrt(np.diag(pcov_ki))
    
    
    # Plot Tij distributions with fits
    x_fit = np.linspace(-3, 3, 600)
    if fitType=="DoubleGaussian":
        fit_y_ij = double_gaussian(x_fit, *popt_ij)
        fit_y_jk = double_gaussian(x_fit, *popt_jk)
        fit_y_ki = double_gaussian(x_fit, *popt_ki)
        fit1_y_ij = gaussian(x_fit, *[amp_ij*frac_ij, mean1_ij, sigma_ij])
        fit1_y_jk = gaussian(x_fit, *[amp_jk*frac_jk, mean1_jk, sigma_jk])
        fit1_y_ki = gaussian(x_fit, *[amp_ki*frac_ki, mean1_ki, sigma_ki])
        fit2_y_ij = gaussian(x_fit, *[amp_ij*(1 - frac_ij), mean2_ij, sigma2_ij])
        fit2_y_jk = gaussian(x_fit, *[amp_jk*(1 - frac_jk), mean2_jk, sigma2_jk])
        fit2_y_ki = gaussian(x_fit, *[amp_ki*(1 - frac_ki), mean2_ki, sigma2_ki])
    elif fitType=="Gaussian":
        fit_y_ij = gaussian(x_fit, *popt_ij)
        fit_y_jk = gaussian(x_fit, *popt_jk)
        fit_y_ki = gaussian(x_fit, *popt_ki)
    
    elif fitType=="GaussianMixture":
        x_fit_ij = np.linspace(np.mean(T_ij) - 4.*np.std(T_ij), np.mean(T_ij) + 4.*np.std(T_ij), 100)
        x_fit_jk = np.linspace(np.mean(T_jk) - 4.*np.std(T_jk), np.mean(T_jk) + 4.*np.std(T_jk), 100)
        x_fit_ki = np.linspace(np.mean(T_ki) - 4.*np.std(T_ki), np.mean(T_ki) + 4.*np.std(T_ki), 100)
        fit_y_ij = eval_gaussian_mixture(x_fit_ij, popt_ij, scale_ij)
        fit_y_jk = eval_gaussian_mixture(x_fit_jk, popt_jk, scale_jk)
        fit_y_ki = eval_gaussian_mixture(x_fit_ki, popt_ki, scale_ki)

    if plot:
    
        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        if fitType=="GaussianMixture":
            counts_ij, bin_edges_ij = np.histogram(T_ij, bins=25, range=(np.mean(T_ij) - 4.*np.std(T_ij), np.mean(T_ij) + 4.*np.std(T_ij)))
            counts_jk, bin_edges_jk = np.histogram(T_jk, bins=25, range=(np.mean(T_jk) - 4.*np.std(T_jk), np.mean(T_jk) + 4.*np.std(T_jk)))
            counts_ki, bin_edges_ki = np.histogram(T_ki, bins=25, range=(np.mean(T_ki) - 4.*np.std(T_ki), np.mean(T_ki) + 4.*np.std(T_ki)))
            bin_centers_ij = (bin_edges_ij[:-1] + bin_edges_ij[1:]) / 2
            bin_centers_jk = (bin_edges_jk[:-1] + bin_edges_jk[1:]) / 2
            bin_centers_ki = (bin_edges_ki[:-1] + bin_edges_ki[1:]) / 2
        ax[0].errorbar(bin_centers_ij, counts_ij, yerr=np.sqrt(counts_ij), fmt='o', markersize=6, linewidth=2, label=r'$T_{ij}^{corr}$', color='k')
        ax[1].errorbar(bin_centers_jk, counts_jk, yerr=np.sqrt(counts_jk), fmt='o', markersize=6, linewidth=2, label=r'$T_{jk}^{corr}$', color='k')
        ax[2].errorbar(bin_centers_ki, counts_ki, yerr=np.sqrt(counts_ki), fmt='o', markersize=6, linewidth=2, label=r'$T_{ki}^{corr}$', color='k')
        if fitType=="DoubleGaussian":
            ax[0].plot(x_fit, fit_y_ij, c='r', label=r'Double gaussian')
            ax[1].plot(x_fit, fit_y_jk, c='r', label=r'Double gaussian')
            ax[2].plot(x_fit, fit_y_ki, c='r', label=r'Double gaussian')

            ax[0].plot(x_fit, fit1_y_ij, c='b', label=r'Gaussian 1: $\sigma_{ij}$' + ' = %.4f'%(sigma_ij))
            ax[1].plot(x_fit, fit1_y_jk, c='b', label=r'Gaussian 1: $\sigma_{jk}$' + ' = %.4f'%(sigma_jk))
            ax[2].plot(x_fit, fit1_y_ki, c='b', label=r'Gaussian 1: $\sigma_{ki}$' + ' = %.4f'%(sigma_ki))
            ax[0].plot(x_fit, fit2_y_ij, c='g', label=r'Gaussian 2: $\sigma_{ij}$' + ' = %.4f'%(sigma2_ij))
            ax[1].plot(x_fit, fit2_y_jk, c='g', label=r'Gaussian 2: $\sigma_{jk}$' + ' = %.4f'%(sigma2_jk))
            ax[2].plot(x_fit, fit2_y_ki, c='g', label=r'Gaussian 2: $\sigma_{ki}$' + ' = %.4f'%(sigma2_ki))
        elif fitType=="Gaussian":
            ax[0].plot(x_fit, fit_y_ij, c='r', label=r'$\sigma_{ij}$' + ' = %.4f'%(sigma_ij) + r'$\pm$' + '%.4f'%(perr_ij[0]))
            ax[1].plot(x_fit, fit_y_jk, c='r', label=r'$\sigma_{jk}$' + ' = %.4f'%(sigma_jk) + r'$\pm$' + '%.4f'%(perr_jk[0]))
            ax[2].plot(x_fit, fit_y_ki, c='r', label=r'$\sigma_{ki}$' + ' = %.4f'%(sigma_ki) + r'$\pm$' + '%.4f'%(perr_ki[0]))
        elif fitType=="GaussianMixture":
            ax[0].plot(x_fit_ij, fit_y_ij, c='r', label=r'Gaussian mixture (3)')
            ax[1].plot(x_fit_jk, fit_y_jk, c='r', label=r'Gaussian mixture (3)')
            ax[2].plot(x_fit_ki, fit_y_ki, c='r', label=r'Gaussian mixture (3)')
        ax[0].axhline(y=max(fit_y_ij)/2., color='gray', linestyle='--', linewidth=1, label="fwhm")
        ax[1].axhline(y=max(fit_y_jk)/2., color='gray', linestyle='--', linewidth=1, label="fwhm")
        ax[2].axhline(y=max(fit_y_ki)/2., color='gray', linestyle='--', linewidth=1, label="fwhm")
        ax[0].set_xlabel(r"Corrected $\Delta T_{ij}$")
        ax[1].set_xlabel(r"Corrected $\Delta T_{jk}$")
        ax[2].set_xlabel(r"Corrected $\Delta T_{ki}$")
        ax[0].set_ylabel("Counts")
        ax[1].set_ylabel("Counts")
        ax[2].set_ylabel("Counts")
        ax[0].set_ylim([0., 1.5 * max(counts_ij)])
        ax[1].set_ylim([0., 1.5 * max(counts_jk)])
        ax[2].set_ylim([0., 1.5 * max(counts_ki)])
        ax[0].set_xlim([np.mean(T_ij) - 4.*np.std(T_ij), np.mean(T_ij) + 4.*np.std(T_ij)])
        ax[1].set_xlim([np.mean(T_jk) - 4.*np.std(T_jk), np.mean(T_jk) + 4.*np.std(T_jk)])
        ax[2].set_xlim([np.mean(T_ki) - 4.*np.std(T_ki), np.mean(T_ki) + 4.*np.std(T_ki)])
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        fig.savefig(f'{outDir}/distribution_dT_{suffix}.png', dpi=150)
        plt.close(fig)

    return [x_fit_ij, x_fit_jk, x_fit_ki], [fit_y_ij, fit_y_jk, fit_y_ki]


def run_bootstrap_analysis(row_i, col_i, presel_events_i, presel_events_j, presel_events_k,
                           output_base_dir, pixel_id, layer_i, layer_j, layer_k, iterations=10, doFWMH=True, useBest=False, doIterPlotting=True, twFitType='linear'):
    """
    Run bootstrap analysis for a specific pixel in layer i

    Args:
        row_i, col_i: Pixel coordinates in layer i
        presel_events_i, presel_events_j, presel_events_k: Preselected events for each layer
        output_base_dir: Base output directory
        pixel_id: Identifier for this pixel (for output organization)
        layer_i, layer_j, layer_k: Physical layer numbers (0, 1, or 2) for i, j, k
        iterations: Number of bootstrap iterations
        twFitType: Time-walk correction fit to use per iteration: 'linear' (curve_fit + func_lineal, default)
            or 'twc' (np.polyfit/np.poly1d degree-2 fit, matching twc.py's three_board_iterative_timewalk_correction)

    Returns:
        Dictionary with analysis results
    """

    # Create output directory for this pixel
    pixel_output_dir = os.path.join(output_base_dir, f'pixel_{pixel_id}_row{row_i}_col{col_i}')
    output_dirs = setup_output_directories(base_output_dir=pixel_output_dir, n_iterations=iterations)

    print(f"\n{'='*80}")
    print(f"Analyzing pixel {pixel_id}: row={row_i}, col={col_i}")
    print(f"Output directory: {pixel_output_dir}")
    print(f"{'='*80}\n")

    # First pixel preselection
    MASK_TARGET_I = ak.flatten((presel_events_i.row==row_i) & (presel_events_i.col==col_i))

    maski_events_i = presel_events_i[MASK_TARGET_I]
    maski_events_j = presel_events_j[MASK_TARGET_I]
    maski_events_k = presel_events_k[MASK_TARGET_I]

    print(f"Events after layer i pixel selection: {len(maski_events_i)}")

    plotLayerMaps(
        os.path.join(output_dirs['heatmaps'], 'heatmap_Target'),
        maski_events_i, maski_events_j, maski_events_k
    )

    # Find combinations for layers j and k (symmetric approach)
    combinations = []
    rates = []

    # Get all (j, k) pixel pairs from events that passed through layer i
    # This is symmetric - order of j and k doesn't matter
    pixels_j = ak.flatten(maski_events_j.row), ak.flatten(maski_events_j.col)
    pixels_k = ak.flatten(maski_events_k.row), ak.flatten(maski_events_k.col)

    # Create triplets (row_i, col_i, row_j, col_j, row_k, col_k) for each event
    triplets = list(zip(pixels_j[0], pixels_j[1], pixels_k[0], pixels_k[1]))

    # Count unique (j, k) combinations
    unique_triplets, triplet_rates = np.unique(triplets, axis=0, return_counts=True)

    # Filter combinations with at least 3 events
    valid_mask = triplet_rates >= 3
    unique_triplets = unique_triplets[valid_mask]
    triplet_rates = triplet_rates[valid_mask]

    # Sort by rate (descending)
    sort_idx = np.argsort(triplet_rates)[::-1]
    unique_triplets = unique_triplets[sort_idx]
    triplet_rates = triplet_rates[sort_idx]

    # Build combinations list
    for triplet, rate in zip(unique_triplets, triplet_rates):
        row_j, col_j, row_k, col_k = triplet
        combinations.append([[row_i, col_i], [row_j, col_j], [row_k, col_k]])
        rates.append(rate)

    fit_parameter_records = []

    if len(combinations) == 0:
        print(f"WARNING: No valid combinations found for pixel row={row_i}, col={col_i}")
        return fit_parameter_records


    # Initial dtoa's and tot's
    init_dtoa_i  = np.array([])
    init_tot_i   = np.array([])
    final_dtoa_i = np.array([])
    final_tot_i  = np.array([])

    # Full corrected dT's
    full_corr_Tij = np.array([])
    full_corr_Tjk = np.array([])
    full_corr_Tki = np.array([])

    # Full corrected dT's
    full_tot_i      = np.array([])
    full_tot_code_i = np.array([])
    full_toa_i      = np.array([])
    full_toa_code_i = np.array([])

    # Select the pixels
    doCombination = True
    final_combinations = []
    if doCombination:
        for c,comb in enumerate(combinations):
            if rates[c] > 200:
                final_combinations.append(comb)
    else:
        final_combinations = combinations[:1]

    print(f"Found {len(combinations)} valid combinations")
    if not useBest:
        print(f"Final combinations to try: {len(final_combinations)}")
        for c in range(0, len(final_combinations)):
            print(final_combinations[c], rates[c])
        print(f"Top combination: {combinations[0]} with {rates[0]} events")
    else:
        final_combinations = final_combinations[:1]
        print(f"Only using one combination: {combinations[0]} with {rates[0]} events")

    # Loop over pixels:
    for c in tqdm(range(0, len(final_combinations)), desc="Pixel combinations"):

        # Select final pixel combination (highest rate)
        sel_row_i, sel_col_i = combinations[c][0]
        sel_row_j, sel_col_j = combinations[c][1]
        sel_row_k, sel_col_k = combinations[c][2]

        MASK_SEL = ak.flatten((maski_events_i.row==sel_row_i) & (maski_events_i.col==sel_col_i) &
                              (maski_events_j.row==sel_row_j) & (maski_events_j.col==sel_col_j) &
                              (maski_events_k.row==sel_row_k) & (maski_events_k.col==sel_col_k))

        sel_events_i = maski_events_i[MASK_SEL]
        sel_events_j = maski_events_j[MASK_SEL]
        sel_events_k = maski_events_k[MASK_SEL]

        # Calculate TOA and TOT values
        toa_code_i = ak.flatten(sel_events_i.toa_code)
        toa_code_j = ak.flatten(sel_events_j.toa_code)
        toa_code_k = ak.flatten(sel_events_k.toa_code)

        tot_code_i = ak.flatten(sel_events_i.tot_code)
        tot_code_j = ak.flatten(sel_events_j.tot_code)
        tot_code_k = ak.flatten(sel_events_k.tot_code)

        cal_code_i = ak.flatten(sel_events_i.cal_code)
        cal_code_j = ak.flatten(sel_events_j.cal_code)
        cal_code_k = ak.flatten(sel_events_k.cal_code)

        # Calculate TOA and TOT
        toa_i = 12.5 - 3.125 / cal_code_i * toa_code_i
        toa_j = 12.5 - 3.125 / cal_code_j * toa_code_j
        toa_k = 12.5 - 3.125 / cal_code_k * toa_code_k

        tot_i = ((2*tot_code_i - np.floor(tot_code_i/32))*3.125 / cal_code_i)
        tot_j = ((2*tot_code_j - np.floor(tot_code_j/32))*3.125 / cal_code_j)
        tot_k = ((2*tot_code_k - np.floor(tot_code_k/32))*3.125 / cal_code_k)

        # dT differences before any TWC correction is applied
        T_ij_raw = toa_i - toa_j
        T_jk_raw = toa_j - toa_k
        T_ki_raw = toa_k - toa_i

        # Bootstrap iterations
        #for ITER in range(0, iterations):
        for ITER in tqdm(range(iterations), desc=f"Iterations for combination {final_combinations[c]}", leave=False):

            # Calculate delta TOA for each layer
            dtoa_i = (toa_j + toa_k) / 2.0 - toa_i
            dtoa_j = (toa_k + toa_i) / 2.0 - toa_j
            dtoa_k = (toa_i + toa_j) / 2.0 - toa_k

            if ITER==0:
                init_dtoa_i = np.concatenate([init_dtoa_i, dtoa_i])
                init_tot_i = np.concatenate([init_tot_i, tot_i])
            if ITER==(iterations-1):
                final_dtoa_i = np.concatenate([final_dtoa_i, dtoa_i])
                final_tot_i = np.concatenate([final_tot_i, tot_i])

            # Plot delta TOA distributions
            if doIterPlotting:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                bins = 50
                ax.hist(dtoa_i, bins=bins, histtype='step', linewidth=2, label=r'$\Delta TOA_{i}$')
                ax.hist(dtoa_j, bins=bins, histtype='step', linewidth=2, label=r'$\Delta TOA_{j}$')
                ax.hist(dtoa_k, bins=bins, histtype='step', linewidth=2, label=r'$\Delta TOA_{k}$')
                ax.set_xlabel(r"$\Delta TOA$")
                ax.set_ylabel("Counts")
                ax.set_title(r"$\Delta TOA$ Distribution - All Layers")
                ax.legend()
                fig.savefig(os.path.join(output_dirs['iterations'][ITER], 'layers_dTOA_histograms.png'), dpi=150)
                plt.close(fig)

            tot_i_forfit  = tot_i
            dtoa_i_forfit = dtoa_i
            toa_i_forfit  = toa_i
            tot_j_forfit  = tot_j
            dtoa_j_forfit = dtoa_j
            toa_j_forfit  = toa_j
            tot_k_forfit  = tot_k
            dtoa_k_forfit = dtoa_k
            toa_k_forfit  = toa_k

            tot_data = [tot_i, tot_j, tot_k]
            dtoa_data = [dtoa_i, dtoa_j, dtoa_k]

            if twFitType == 'twc':
                # Same tools as twc.py's three_board_iterative_timewalk_correction: np.polyfit + np.poly1d
                tw_corrected_i = np.polyfit(tot_i_forfit.to_list(), dtoa_i_forfit.to_list(), 2)
                tw_corrected_j = np.polyfit(tot_j_forfit.to_list(), dtoa_j_forfit.to_list(), 2)
                tw_corrected_k = np.polyfit(tot_k_forfit.to_list(), dtoa_k_forfit.to_list(), 2)

                # Store one fit per target pixel and iteration, using its
                # highest-statistics pixel triplet (the first combination).
                if c == 0:
                    for layer, row, col, coefficients in [
                        (layer_i, sel_row_i, sel_col_i, tw_corrected_i),
                        (layer_j, sel_row_j, sel_col_j, tw_corrected_j),
                        (layer_k, sel_row_k, sel_col_k, tw_corrected_k),
                    ]:
                        fit_parameter_records.append({
                            "layer": int(layer), "row": int(row), "col": int(col),
                            "iteration": int(ITER), "n": int(len(sel_events_i)),
                            "a": float(coefficients[0]),
                            "b": float(coefficients[1]),
                            "c": float(coefficients[2]),
                        })

                # Plot fits
                plot_correction_fit(dtoa_data, tot_data, [tw_corrected_i, tw_corrected_j, tw_corrected_k], os.path.join(output_dirs['iterations'][ITER], 'fits_deltaTOA_TOT.png'), useTWC=True)

                # Apply corrections to all events
                toa_i = toa_i + np.poly1d(tw_corrected_i)(np.asarray(tot_i))
                toa_j = toa_j + np.poly1d(tw_corrected_j)(np.asarray(tot_j))
                toa_k = toa_k + np.poly1d(tw_corrected_k)(np.asarray(tot_k))
            else:
                # Fit linear corrections to TOT vs dTOA
                tw_corrected_i = curve_fit(func_lineal, tot_i_forfit.to_list(), dtoa_i_forfit.to_list())[0]
                tw_corrected_j = curve_fit(func_lineal, tot_j_forfit.to_list(), dtoa_j_forfit.to_list())[0]
                tw_corrected_k = curve_fit(func_lineal, tot_k_forfit.to_list(), dtoa_k_forfit.to_list())[0]

                # Plot fits
                plot_correction_fit(dtoa_data, tot_data, [tw_corrected_i, tw_corrected_j, tw_corrected_k], os.path.join(output_dirs['iterations'][ITER], 'fits_deltaTOA_TOT.png'), useCubic=False)

                # Apply corrections to all events
                toa_i = toa_i + func_lineal(np.asarray(tot_i), *tw_corrected_i)
                toa_j = toa_j + func_lineal(np.asarray(tot_j), *tw_corrected_j)
                toa_k = toa_k + func_lineal(np.asarray(tot_k), *tw_corrected_k)

            # Plot corrected TOA distributions
            if doIterPlotting:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                ax.hist(toa_i, bins=50, histtype='step', linewidth=2, label=r'$TOA_{i}^{corr}$')
                ax.hist(toa_j, bins=50, histtype='step', linewidth=2, label=r'$TOA_{j}^{corr}$')
                ax.hist(toa_k, bins=50, histtype='step', linewidth=2, label=r'$TOA_{k}^{corr}$')
                ax.set_xlabel(r"Corrected $TOA$")
                ax.set_ylabel("Counts")
                ax.set_title(r"Corrected $TOA$ Distribution - All Layers")
                ax.legend()
                fig.savefig(os.path.join(output_dirs['iterations'][ITER], 'distribution_TOA_corr.png'), dpi=150)
                plt.close(fig)

            # Calculate dT differences
            T_ij = toa_i - toa_j
            T_jk = toa_j - toa_k
            T_ki = toa_k - toa_i

            # Fit dT
            x_fit, fit_y = fit_deltaT(T_ij, T_jk, T_ki, fitType="GaussianMixture", outDir=os.path.join(output_dirs['iterations'][ITER]), suffix=f"_{sel_row_i}-{sel_col_i}_{sel_row_j}-{sel_col_j}_{sel_row_k}-{sel_col_k}", plot=doIterPlotting)

        # End of iterations -> save files
        with uproot.recreate(os.path.join(output_base_dir + '/step1/', f'corrected_deltaT_{sel_row_i:02d}{sel_col_i:02d}_{sel_row_j:02d}{sel_col_j:02d}_{sel_row_k:02d}{sel_col_k:02d}.root')) as f:
            f["tracks"] = {
                "Tij": T_ij,
                "Tjk": T_jk,
                "Tki": T_ki,
                "Tij_raw": T_ij_raw,
                "Tjk_raw": T_jk_raw,
                "Tki_raw": T_ki_raw
            }

    return fit_parameter_records


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":

    # ========================================================================
    # Parse command line arguments
    # ========================================================================

    parser = argparse.ArgumentParser(description='Bootstrap analysis for ETL timing resolution')
    parser.add_argument('--layer-i', type=int, default=0, choices=[0, 1, 2],
                       help='Physical layer number to use as layer i (default: 0)')
    parser.add_argument('--layer-j', type=int, default=1, choices=[0, 1, 2],
                       help='Physical layer number to use as layer j (default: 1)')
    parser.add_argument('--layer-k', type=int, default=2, choices=[0, 1, 2],
                       help='Physical layer number to use as layer k (default: 2)')
    parser.add_argument('--min-hits', type=int, default=500,
                       help='Minimum hits required in layer i pixel (default: 500)')
    parser.add_argument('--iterations', type=int, default=4,
                       help='Number of bootstrap iterations (default: 2)')
    parser.add_argument('--output-dir', type=str, default='',
                       help='Base output directory (default: None)')
    parser.add_argument('--input-dir', type=str, default='',
                       help='Input directory')
    parser.add_argument('--do-limit', type=int, default=0,
                       help='Limit of pixels')
    parser.add_argument('--use-best', type=int, default=0,
                       help='Decide if we limit to best combination')
    parser.add_argument('--do-light', type=int, default=0,
                   help='Dont plot many things')
    parser.add_argument('--row-i', type=int, default=-1,
                   help='Select one pixel')
    parser.add_argument('--col-i', type=int, default=-1,
                   help='Select one pixel')
    parser.add_argument('--trigger', type=str, default='k',
                   help='Select trigger layer')
    parser.add_argument('--tw-fit-type', type=str, default='twc', choices=['linear', 'twc'],
                   help="Time-walk correction fit: 'linear' (curve_fit + func_lineal, default) or "
                        "'twc' (np.polyfit/np.poly1d degree-2 fit, matching twc.py)")
    parser.add_argument('--twc-rms-tot-min', type=float, default=0.0,
                   help='Lower physical-TOT bound in ns for unweighted TWC RMS maps (default: 0)')
    parser.add_argument('--twc-rms-tot-max', type=float, default=12.5,
                   help='Upper physical-TOT bound in ns for unweighted TWC RMS maps (default: 12.5)')
    parser.add_argument('--twc-rms-tot-points', type=int, default=251,
                   help='Number of uniformly spaced TOT points used by TWC RMS maps (default: 251)')
    parser.add_argument('--doGlobal', action='store_true',
                   help="Use row_global/col_global (range 0..31) instead of row/col "
                        "(range 0..16). Must match what do_bootstrap_preselection.py "
                        "used to produce the input preselected data.")



    args = parser.parse_args()

    # Validate that i, j, k are all different
    if len(set([args.layer_i, args.layer_j, args.layer_k])) != 3:
        parser.error("layer-i, layer-j, and layer-k must all be different")

    # Configuration
    DO_GLOBAL_COORDINATES = args.doGlobal
    N_PIX                 = 32 if DO_GLOBAL_COORDINATES else 16

    LAYER_I = args.layer_i
    LAYER_J = args.layer_j
    LAYER_K = args.layer_k
    MIN_HITS_THRESHOLD = args.min_hits
    ITERATIONS = args.iterations
    BASE_OUTPUT_DIR = args.output_dir if args.output_dir!='' else f'bootstrapResults_i-{args.layer_i}_j-{args.layer_j}_i-{args.layer_k}'
    if args.row_i > 0:
        BASE_OUTPUT_DIR = BASE_OUTPUT_DIR + f'_row-{args.row_i}'
    if args.col_i > 0:
        BASE_OUTPUT_DIR = BASE_OUTPUT_DIR + f'_row-{args.col_i}'
    DOLIMIT = args.do_limit
    USEBEST = args.use_best
    DOLIGHT = args.do_light
    TRIGGER = args.trigger
    DOITERPLOTTING = not DOLIGHT
    do1HitUniform = False

    print(f"\n{'='*80}")
    print("Bootstrap Analysis Configuration")
    print(f"{'='*80}")
    print(f"Layer mapping: i={LAYER_I}, j={LAYER_J}, k={LAYER_K}")
    print(f"Minimum hits threshold: {MIN_HITS_THRESHOLD}")
    print(f"Bootstrap iterations: {ITERATIONS}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"{'='*80}\n")

    # Setup initial output directories for global plots
    output_dirs = setup_output_directories(base_output_dir=BASE_OUTPUT_DIR, n_iterations=ITERATIONS)
    print(f"Output will be saved to: {BASE_OUTPUT_DIR}/")

    # Load preselected data
    #presel_dir = "/eos/user/f/fernance/ETL/TEST-Bootstrap/Nov2025-TEST/Test-0T-preselected"
    #presel_dir = "/eos/user/f/fernance/ETL/TEST-Bootstrap/Nov2025-TEST/Test-bootstrap_132592_132781"
    presel_dir = args.input_dir
    layers = load_preselected_data(presel_dir)

    # Load events variables (using layer mapping)
    all_events_i = layers[LAYER_I]["events"]
    all_events_j = layers[LAYER_J]["events"]
    all_events_k = layers[LAYER_K]["events"]

    plotLayerMaps(
        os.path.join(output_dirs['heatmaps'], 'heatmap_allPixels_allEvents'),
        all_events_i, all_events_j, all_events_k,
        row_field='row_sel', col_field='col_sel'
    )

    # ========================================================================
    # Preselect data
    # ========================================================================

    presel_events_i = all_events_i
    presel_events_j = all_events_j
    presel_events_k = all_events_k

    # Plot TOA_CODE vs TOT_CODE distributions
    plot_2d_histograms(
        [presel_events_i, presel_events_j, presel_events_k],
        'toa_code_sel', 'tot_code_sel',
        r'TOA\_CODE', r'TOT\_CODE',
        os.path.join(output_dirs['distributions'], 'distribution_TOACODE-TOTCODE')
    )

    # Plot TOA_CODE vs CAL_CODE distributions
    plot_2d_histograms(
        [presel_events_i, presel_events_j, presel_events_k],
        'toa_code_sel', 'cal_code_sel',
        r'TOA\_CODE', r'CAL\_CODE',
        os.path.join(output_dirs['distributions'], 'distribution_TOACODE-CALCODE')
    )

    # Plot TOT_CODE vs CAL_CODE distributions
    plot_2d_histograms(
        [presel_events_i, presel_events_j, presel_events_k],
        'tot_code_sel', 'cal_code_sel',
        r'TOT\_CODE', r'CAL\_CODE',
        os.path.join(output_dirs['distributions'], 'distribution_TOTCODE-CALCODE')
    )

    # CAL CODE mode
    plotLayerMaps_mode(os.path.join(output_dirs['heatmaps'], 'heatmap_CAL_CODE_mode'),
        presel_events_i, presel_events_j, presel_events_k, 'cal_code_sel',
        row_field='row_sel', col_field='col_sel'
    )

    ## Qualities vs event
    #plot_evolution(
    #    [presel_events_i, presel_events_j, presel_events_k],
    #    'tot_code',
    #    r'TOT\_CODE',
    #    os.path.join(output_dirs['distributions'], 'distribution_event-TOTCODE')
    #)


    def to_flat(arr):
        if arr.ndim == 1:
            return ak.to_numpy(arr)
        else:
            return ak.to_numpy(ak.flatten(arr))

    # Apply preselection masks
    if do1HitUniform:
        PRESEL_I = (ak.flatten(presel_events_i.toa_code) > MIN_TOACODE) & (ak.flatten(presel_events_i.toa_code) < MAX_TOACODE) & (ak.flatten(presel_events_i.tot_code) > MIN_TOTCODE) & (ak.flatten(presel_events_i.tot_code) < MAX_TOTCODE) & (ak.flatten(presel_events_i.cal_code) > MIN_CALCODE) & (ak.flatten(presel_events_i.cal_code) < MAX_CALCODE)
        PRESEL_J = (ak.flatten(presel_events_j.toa_code) > MIN_TOACODE) & (ak.flatten(presel_events_j.toa_code) < MAX_TOACODE) & (ak.flatten(presel_events_j.tot_code) > MIN_TOTCODE) & (ak.flatten(presel_events_j.tot_code) < MAX_TOTCODE) & (ak.flatten(presel_events_j.cal_code) > MIN_CALCODE) & (ak.flatten(presel_events_j.cal_code) < MAX_CALCODE)
        PRESEL_K = (ak.flatten(presel_events_k.toa_code) > MIN_TOACODE) & (ak.flatten(presel_events_k.toa_code) < MAX_TOACODE) & (ak.flatten(presel_events_k.tot_code) > MIN_TOTCODE) & (ak.flatten(presel_events_k.tot_code) < MAX_TOTCODE) & (ak.flatten(presel_events_k.cal_code) > MIN_CALCODE) & (ak.flatten(presel_events_k.cal_code) < MAX_CALCODE)

        presel_events_i = presel_events_i[PRESEL_I & PRESEL_J & PRESEL_K]
        presel_events_j = presel_events_j[PRESEL_I & PRESEL_J & PRESEL_K]
        presel_events_k = presel_events_k[PRESEL_I & PRESEL_J & PRESEL_K]
    else:

        # TRIGGER TOA WINDOW
        MIN_TOACODE_WINDOW = 50
        MAX_TOACODE_WINDOW = 1000

        if TRIGGER=='i':
            GOOD_TOA_WINDOW = (to_flat(presel_events_i.toa_code_sel) > MIN_TOACODE_WINDOW) & (to_flat(presel_events_i.toa_code_sel) < MAX_TOACODE_WINDOW)
            plot_toa_tot_cal_distributions(
                presel_events_i, presel_events_j, presel_events_k,
                LAYER_I, LAYER_J, LAYER_K,
                output_dir=output_dirs['distributions'],
                variable="toa_code_sel",
                name="beforeTrgTOAwindow",
                cuts_i=(MIN_TOACODE_WINDOW, MAX_TOACODE_WINDOW),
                cuts_j=None,
                cuts_k=None,
            )
        #
        if TRIGGER=='j':
            GOOD_TOA_WINDOW = (to_flat(presel_events_j.toa_code_sel) > MIN_TOACODE_WINDOW) & (to_flat(presel_events_j.toa_code_sel) < MAX_TOACODE_WINDOW)
            plot_toa_tot_cal_distributions(
                presel_events_i, presel_events_j, presel_events_k,
                LAYER_I, LAYER_J, LAYER_K,
                output_dir=output_dirs['distributions'],
                variable="toa_code_sel",
                name="beforeTrgTOAwindow",
                cuts_i=None,
                cuts_j=(MIN_TOACODE_WINDOW, MAX_TOACODE_WINDOW),
                cuts_k=None,
            )
        #
        if TRIGGER=='k':
            GOOD_TOA_WINDOW = (to_flat(presel_events_k.toa_code_sel) > MIN_TOACODE_WINDOW) & (to_flat(presel_events_k.toa_code_sel) < MAX_TOACODE_WINDOW)
            plot_toa_tot_cal_distributions(
                presel_events_i, presel_events_j, presel_events_k,
                LAYER_I, LAYER_J, LAYER_K,
                output_dir=output_dirs['distributions'],
                variable="toa_code_sel",
                name="beforeTrgTOAwindow",
                cuts_i=None,
                cuts_j=None,
                cuts_k=(MIN_TOACODE_WINDOW, MAX_TOACODE_WINDOW),
            )


        presel_events_i = presel_events_i[GOOD_TOA_WINDOW]
        presel_events_j = presel_events_j[GOOD_TOA_WINDOW]
        presel_events_k = presel_events_k[GOOD_TOA_WINDOW]

        # TOT selection
        useTOTCodeSelection = True

        def mode_window_percentile(arr, percentiles, window=100):
            values, counts = np.unique(arr, return_counts=True)
            mode = values[np.argmax(counts)]
            half_window = window / 2.
            window_values = arr[(arr > mode - half_window) & (arr < mode + half_window)]
            return np.percentile(window_values, percentiles)

        presel_events_i = ak.with_field(presel_events_i, (2*presel_events_i.tot_code_sel - np.floor(presel_events_i.tot_code_sel/32))*3.125 / presel_events_i.cal_code_sel, "tot_sel")
        presel_events_j = ak.with_field(presel_events_j, (2*presel_events_j.tot_code_sel - np.floor(presel_events_j.tot_code_sel/32))*3.125 / presel_events_j.cal_code_sel, "tot_sel")
        presel_events_k = ak.with_field(presel_events_k, (2*presel_events_k.tot_code_sel - np.floor(presel_events_k.tot_code_sel/32))*3.125 / presel_events_k.cal_code_sel, "tot_sel")

        if useTOTCodeSelection:

            MIN_TOTCODE_SELECTION = 50
            MAX_TOTCODE_SELECTION = 150

            plotLayerMaps_mean(os.path.join(output_dirs['heatmaps'], 'heatmap_TOT_CODE_mean'),
                presel_events_i, presel_events_j, presel_events_k, 'tot_code_sel',
                row_field='row_sel', col_field='col_sel'
            )

            tot_code_sel_i = to_flat(presel_events_i.tot_code_sel)
            tot_code_sel_j = to_flat(presel_events_j.tot_code_sel)
            tot_code_sel_k = to_flat(presel_events_k.tot_code_sel)

            MIN_TOTCODE_SELECTION_i, MAX_TOTCODE_SELECTION_i = mode_window_percentile(tot_code_sel_i, [0.5, 99.5])
            MIN_TOTCODE_SELECTION_j, MAX_TOTCODE_SELECTION_j = mode_window_percentile(tot_code_sel_j, [0.5, 99.5])
            MIN_TOTCODE_SELECTION_k, MAX_TOTCODE_SELECTION_k = mode_window_percentile(tot_code_sel_k, [0.5, 99.5])

            MAX_TOTCODE_SELECTION_i = min([MAX_TOTCODE_SELECTION_i, 200])
            MAX_TOTCODE_SELECTION_j = min([MAX_TOTCODE_SELECTION_j, 200])
            MAX_TOTCODE_SELECTION_k = min([MAX_TOTCODE_SELECTION_k, 200])

            print(MIN_TOTCODE_SELECTION_i, MAX_TOTCODE_SELECTION_i)
            print(MIN_TOTCODE_SELECTION_j, MAX_TOTCODE_SELECTION_j)
            print(MIN_TOTCODE_SELECTION_k, MAX_TOTCODE_SELECTION_k)

            GOOD_TOT_VALUE = ((tot_code_sel_i > MIN_TOTCODE_SELECTION_i) & (tot_code_sel_j > MIN_TOTCODE_SELECTION_j) & (tot_code_sel_k > MIN_TOTCODE_SELECTION_k)
                                        & (tot_code_sel_i < MAX_TOTCODE_SELECTION_i) & (tot_code_sel_j < MAX_TOTCODE_SELECTION_j) & (tot_code_sel_k < MAX_TOTCODE_SELECTION_k))

            plot_toa_tot_cal_distributions(
                presel_events_i, presel_events_j, presel_events_k,
                LAYER_I, LAYER_J, LAYER_K,
                output_dir=output_dirs['distributions'],
                variable="tot_code_sel",
                name="beforeTOTSel",
                cuts_i=(MIN_TOTCODE_SELECTION_i, MAX_TOTCODE_SELECTION_i),
                cuts_j=(MIN_TOTCODE_SELECTION_j, MAX_TOTCODE_SELECTION_j),
                cuts_k=(MIN_TOTCODE_SELECTION_k, MAX_TOTCODE_SELECTION_k),
            )

            presel_events_i = presel_events_i[GOOD_TOT_VALUE]
            presel_events_j = presel_events_j[GOOD_TOT_VALUE]
            presel_events_k = presel_events_k[GOOD_TOT_VALUE]

        else:

            tot_sel_i = to_flat(presel_events_i.tot_sel)
            tot_sel_j = to_flat(presel_events_j.tot_sel)
            tot_sel_k = to_flat(presel_events_k.tot_sel)

            MIN_TOT_SELECTION_i, MAX_TOT_SELECTION_i = np.percentile(tot_sel_i, [0.0, 99.0])
            MIN_TOT_SELECTION_j, MAX_TOT_SELECTION_j = np.percentile(tot_sel_j, [0.0, 99.0])
            MIN_TOT_SELECTION_k, MAX_TOT_SELECTION_k = np.percentile(tot_sel_k, [0.0, 99.0])

            MIN_TOT_SELECTION_i = 0.
            MIN_TOT_SELECTION_j = 0.
            MIN_TOT_SELECTION_k = 0.

            plot_toa_tot_cal_distributions(
                presel_events_i, presel_events_j, presel_events_k,
                LAYER_I, LAYER_J, LAYER_K,
                output_dir=output_dirs['distributions'],
                variable="tot_sel",
                name="beforeTOTSel",
                cuts_i=(MIN_TOT_SELECTION_i, MAX_TOT_SELECTION_i),
                cuts_j=(MIN_TOT_SELECTION_j, MAX_TOT_SELECTION_j),
                cuts_k=(MIN_TOT_SELECTION_k, MAX_TOT_SELECTION_k),
            )

            GOOD_TOT_VALUE = ((tot_sel_i > MIN_TOT_SELECTION_i) & (tot_sel_j > MIN_TOT_SELECTION_j) & (tot_sel_k > MIN_TOT_SELECTION_k)
                                        & (tot_sel_i < MAX_TOT_SELECTION_i) & (tot_sel_j < MAX_TOT_SELECTION_j) & (tot_sel_k < MAX_TOT_SELECTION_k))

            presel_events_i = presel_events_i[GOOD_TOT_VALUE]
            presel_events_j = presel_events_j[GOOD_TOT_VALUE]
            presel_events_k = presel_events_k[GOOD_TOT_VALUE]



        # TOA selection
        MIN_TOACODE_SELECTION = 100
        MAX_TOACODE_SELECTION = 550

        plot_toa_tot_cal_distributions(
            presel_events_i, presel_events_j, presel_events_k,
            LAYER_I, LAYER_J, LAYER_K,
            output_dir=output_dirs['distributions'],
            variable="toa_code_sel",
            name="beforeTOASel",
            cuts_i=(MIN_TOACODE_SELECTION, MAX_TOACODE_SELECTION),
            cuts_j=(MIN_TOACODE_SELECTION, MAX_TOACODE_SELECTION),
            cuts_k=(MIN_TOACODE_SELECTION, MAX_TOACODE_SELECTION),
        )

        toa_code_sel_i = to_flat(presel_events_i.toa_code_sel)
        toa_code_sel_j = to_flat(presel_events_j.toa_code_sel)
        toa_code_sel_k = to_flat(presel_events_k.toa_code_sel)

        GOOD_TOA_VALUE = ((toa_code_sel_i > MIN_TOACODE_SELECTION) & (toa_code_sel_j > MIN_TOACODE_SELECTION) & (toa_code_sel_k > MIN_TOACODE_SELECTION) & (toa_code_sel_i < MAX_TOACODE_SELECTION) & (toa_code_sel_j < MAX_TOACODE_SELECTION) & (toa_code_sel_k < MAX_TOACODE_SELECTION))

        # Apply selection
        #presel_events_i = presel_events_i[GOOD_TOA_VALUE]
        #presel_events_j = presel_events_j[GOOD_TOA_VALUE]
        #presel_events_k = presel_events_k[GOOD_TOA_VALUE]

        # CAL_CODE selection
        mode_map_i = build_mode_map(presel_events_i)
        mode_map_j = build_mode_map(presel_events_j)
        mode_map_k = build_mode_map(presel_events_k)

        print(mode_map_i)

        rows_i = to_flat(presel_events_i.row_sel)
        cols_i = to_flat(presel_events_i.col_sel)
        rows_j = to_flat(presel_events_j.row_sel)
        cols_j = to_flat(presel_events_j.col_sel)
        rows_k = to_flat(presel_events_k.row_sel)
        cols_k = to_flat(presel_events_k.col_sel)
        
        cal_i = to_flat(presel_events_i.cal_code_sel)
        cal_j = to_flat(presel_events_j.cal_code_sel)
        cal_k = to_flat(presel_events_k.cal_code_sel)

        GOOD_CAL_VALUE = (
            (np.abs(cal_i - mode_map_i[rows_i, cols_i]) < 2) &
            (np.abs(cal_j - mode_map_j[rows_j, cols_j]) < 2) &
            (np.abs(cal_k - mode_map_k[rows_k, cols_k]) < 2)
        )

        presel_events_i = presel_events_i[GOOD_CAL_VALUE]
        presel_events_j = presel_events_j[GOOD_CAL_VALUE]
        presel_events_k = presel_events_k[GOOD_CAL_VALUE]

        # Redefine the values
        presel_events_i = ak.with_field(presel_events_i, presel_events_i.toa_code_sel, "toa_code")
        presel_events_i = ak.with_field(presel_events_i, presel_events_i.tot_code_sel, "tot_code")
        presel_events_i = ak.with_field(presel_events_i, presel_events_i.cal_code_sel, "cal_code")
        presel_events_i = ak.with_field(presel_events_i, presel_events_i.row_sel, "row")
        presel_events_i = ak.with_field(presel_events_i, presel_events_i.col_sel, "col")

        presel_events_j = ak.with_field(presel_events_j, presel_events_j.toa_code_sel, "toa_code")
        presel_events_j = ak.with_field(presel_events_j, presel_events_j.tot_code_sel, "tot_code")
        presel_events_j = ak.with_field(presel_events_j, presel_events_j.cal_code_sel, "cal_code")
        presel_events_j = ak.with_field(presel_events_j, presel_events_j.row_sel, "row")
        presel_events_j = ak.with_field(presel_events_j, presel_events_j.col_sel, "col")

        presel_events_k = ak.with_field(presel_events_k, presel_events_k.toa_code_sel, "toa_code")
        presel_events_k = ak.with_field(presel_events_k, presel_events_k.tot_code_sel, "tot_code")
        presel_events_k = ak.with_field(presel_events_k, presel_events_k.cal_code_sel, "cal_code")
        presel_events_k = ak.with_field(presel_events_k, presel_events_k.row_sel, "row")
        presel_events_k = ak.with_field(presel_events_k, presel_events_k.col_sel, "col")

        def ensure_vector_fields(events, fields=["toa_code", "tot_code", "cal_code", "row", "col"]):
            for field in fields:
                # ak.firsts() (used to build the trigger layer's *_sel fields) always
                # returns an option ("?") type, even though mask_1hit guarantees no
                # actual None survives. Strip it here so it doesn't propagate into
                # arithmetic downstream and break the final ROOT write with uproot.
                arr = ak.fill_none(events[field], 0)
                if arr.ndim == 1:
                    arr = ak.unflatten(arr, 1)
                events = ak.with_field(events, arr, field)
            return events

        
        presel_events_i = ensure_vector_fields(presel_events_i)
        presel_events_j = ensure_vector_fields(presel_events_j)
        presel_events_k = ensure_vector_fields(presel_events_k)



    print(f'Events with preselected: {len(presel_events_i)}')

    plotLayerMaps(
        os.path.join(output_dirs['heatmaps'], 'heatmap_allPixels_preselected'),
        presel_events_i, presel_events_j, presel_events_k
    )

    # ========================================================================
    # Find all high-rate pixels in layer i
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"Finding all pixels in layer i with > {MIN_HITS_THRESHOLD} hits...")
    print(f"{'='*80}\n")

    # Count hits per pixel in layer i
    pixels_i = list(zip(ak.flatten(presel_events_i.row), ak.flatten(presel_events_i.col)))
    unique_pixels_i, pixel_counts_i = np.unique(pixels_i, axis=0, return_counts=True)

    # Filter pixels with more than MIN_HITS_THRESHOLD hits
    high_rate_mask = pixel_counts_i > MIN_HITS_THRESHOLD
    high_rate_pixels = unique_pixels_i[high_rate_mask]
    high_rate_counts = pixel_counts_i[high_rate_mask]

    # Sort by hit count (descending)
    sorted_indices = np.argsort(high_rate_counts)[::-1]
    high_rate_pixels = high_rate_pixels[sorted_indices]
    high_rate_counts = high_rate_counts[sorted_indices]

    print(f"Found {len(high_rate_pixels)} pixels with > {MIN_HITS_THRESHOLD} hits")
    print(f"\nPixel list (row, col, hits):")
    for idx, (pixel, count) in enumerate(zip(high_rate_pixels, high_rate_counts)):
        print(f"  {idx+1}. Row={pixel[0]}, Col={pixel[1]}, Hits={count}")

    # ========================================================================
    # Run bootstrap analysis for each high-rate pixel
    # ========================================================================

    limit = DOLIMIT
    ilimit = 0
    fit_parameter_records = []

    for pixel_id, (pixel, count) in enumerate(zip(high_rate_pixels, high_rate_counts)):
        if ilimit > limit and DOLIMIT:
            break
        ilimit +=1

        row_i, col_i = pixel

        if args.row_i > 0:
            if not (args.row_i==row_i):
                continue
        if args.col_i > 0:
            if not (args.col_i==col_i):
                continue

        pixel_fit_parameters = run_bootstrap_analysis(
            row_i=row_i,
            col_i=col_i,
            presel_events_i=presel_events_i,
            presel_events_j=presel_events_j,
            presel_events_k=presel_events_k,
            output_base_dir=BASE_OUTPUT_DIR,
            pixel_id=pixel_id,
            layer_i=LAYER_I,
            layer_j=LAYER_J,
            layer_k=LAYER_K,
            iterations=ITERATIONS,
            doFWMH=True,
            useBest=USEBEST,
            doIterPlotting=True,
            twFitType=args.tw_fit_type
        )
        fit_parameter_records.extend(pixel_fit_parameters)

    if fit_parameter_records:
        # A reference-layer pixel may occur in several target-pixel triplets.
        # Keep only its highest-statistics fit in each iteration.
        best_records = {}
        for record in fit_parameter_records:
            key = (record['layer'], record['row'], record['col'], record['iteration'])
            if key not in best_records or record['n'] > best_records[key]['n']:
                best_records[key] = record
        fit_parameter_records = list(best_records.values())

        with open(os.path.join(output_dirs['twc'], 'twc_fit_parameters.json'), 'w') as f:
            json.dump(fit_parameter_records, f, indent=2)

        plot_twc_rms_heatmaps(
            fit_parameter_records,
            output_dirs['twc'],
            [LAYER_I, LAYER_J, LAYER_K],
            ITERATIONS,
            tot_min=args.twc_rms_tot_min,
            tot_max=args.twc_rms_tot_max,
            n_tot_points=args.twc_rms_tot_points,
        )

        for iteration in range(ITERATIONS):
            iteration_dir = output_dirs['twc_iterations'][iteration]
            for layer in [LAYER_I, LAYER_J, LAYER_K]:
                iteration_records = [
                    record for record in fit_parameter_records
                    if record['layer'] == layer and record['iteration'] == iteration
                ]
                for coefficient in ['a', 'b', 'c']:
                    values = [record[coefficient] for record in iteration_records]
                    if not values:
                        continue
                    fig, ax = plt.subplots(figsize=(9, 7))
                    ax.hist(values, bins=30, histtype='step', linewidth=1.5)
                    ax.set_xlabel(f'Quadratic TWC coefficient {coefficient}')
                    ax.set_ylabel('Pixels')
                    ax.set_title(f'Layer {layer}, iteration {iteration}')
                    fig.tight_layout()
                    name = f'twc_coefficient_{coefficient}_layer{layer}'
                    fig.savefig(os.path.join(iteration_dir, name + '.png'), dpi=150)
                    fig.savefig(os.path.join(iteration_dir, name + '.pdf'))
                    plt.close(fig)

    print(f"\nAnalysis complete!")

