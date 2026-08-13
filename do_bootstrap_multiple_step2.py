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
import json
import argparse
from tqdm import tqdm
import mplhep as hep
import glob, re
hep.style.use("CMS")

#reload(au)
#reload(tbplt)

# Constants
BRANCHES = ["row", "col", "tot_code", "cal_code", "toa_code", "mcp_volts", "mcp_seconds", "clock_seconds", "clock_volts", "chipid", "nhits"]
ITERATIONS = 10
CANDIDATE_COLORS = {1: '#3f90da', 2: '#ffa90e', 3: '#bd1f01'}
LEGEND_FONTSIZE = matplotlib.font_manager.FontProperties(size='small').get_size_in_points() - 1

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
        'distributions': os.path.join(base_output_dir, 'distributions'),
        'final': os.path.join(base_output_dir, 'final'),
        'iterations': [],
        'step1': os.path.join(base_output_dir, 'step1'),
        'step2': os.path.join(base_output_dir, 'step2'),
        'step3': os.path.join(base_output_dir, 'step3')
    }

    # Create base directories
    os.makedirs(dirs['heatmaps'], exist_ok=True)
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

    return dirs

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
    return {a: {b: {'dt': [], 'tot': []} for b in range(16)} for a in range(16)}

def compute_fwhm(xs, ys):
    """FWHM from a curve, same method as twc.py's fit_gmm_and_get_fwhm:
    take all grid points above half the peak and use the outermost ones."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    peak_val = np.max(ys)
    half_max_indices = np.where(ys >= peak_val / 2.0)[0]

    if len(half_max_indices) < 2:
        raise ValueError("Less than 2 points above half max.")

    return (xs[half_max_indices[-1]] - xs[half_max_indices[0]],
            xs[half_max_indices[0]], xs[half_max_indices[-1]])

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
        x_data = ak.to_numpy(ak.flatten(getattr(events, x_field)))
        y_data = ak.to_numpy(ak.flatten(getattr(events, y_field)))
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
from scipy.stats import kstest, norm

def fit_gaussian_mixture(data, n_gaussians=2, bins=40, range_tuple=(-1, 1), n_init=1):
    mask = (data >= range_tuple[0]) & (data <= range_tuple[1])
    data_clipped = data[mask]

    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='full', n_init=n_init)
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


def _gmm_ks_score(data, gmm):
    """KS goodness-of-fit score for a fitted GMM (same idea as twc.py's calculate_gmm_cdf)."""
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())

    def gmm_cdf(x):
        cdf_val = np.zeros_like(x, dtype=float)
        for w, m, s in zip(weights, means, stds):
            cdf_val += w * norm.cdf(x, m, s)
        return cdf_val

    ks_score, _ = kstest(np.sort(data), gmm_cdf)
    return ks_score


def fit_gmm_candidates(data, bins, range_tuple, components_to_try=(1, 2, 3), n_init=3):
    """Fit GMMs with 1/2/3 components and pick the best one via a KS test,
    same selection logic as twc.py's fit_gmm_and_get_fwhm."""
    fits = {}
    best_n, best_ks = None, np.inf

    for n_comp in components_to_try:
        popt, gmm, scale = fit_gaussian_mixture(data, n_gaussians=n_comp, bins=bins,
                                                  range_tuple=range_tuple, n_init=n_init)
        ks_score = _gmm_ks_score(data, gmm)
        fits[n_comp] = {'popt': popt, 'scale': scale, 'ks': ks_score}
        if ks_score < best_ks:
            best_ks, best_n = ks_score, n_comp

    return fits, best_n


def plot_gaussian_mixture(ax, popt, scale, x_range=(-1, 1)):
    x = np.linspace(*x_range, 500)
    y_total = np.zeros_like(x)

    for i in range(0, len(popt), 3):
        amp, mu, sigma = popt[i], popt[i+1], popt[i+2]
        y = scale * amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y_total += y
        ax.plot(x, y, linestyle='--', label=f'G{i//3 + 1}: μ={mu:.3f}, σ={sigma:.3f}')

    ax.plot(x, y_total, color='black', linewidth=2, label='Total')


def load_events(run_files, branches):
    """Load events from ROOT files in batches"""
    events_list = []
    for i, batch in enumerate(uproot.iterate(run_files, branches, step_size=5000)):
        print(f"Processing batch {i+1}, events: {len(batch)}")
        events_list.append(batch)
    print("Concatenating all batches...")
    all_events = ak.concatenate(events_list)
    print(f"Total events loaded: {len(all_events)}")
    return all_events

def plotLayerMaps(name, eventsi, eventsj, eventsk, mc='viridis'):
    """
    Plot heatmaps for all three layers

    Args:
        name: Output filename (without extension)
        events0, events1, events2: Event data for each layer
        mc: Matplotlib colormap
    """
    # Create histograms for each layer
    histograms = []
    events_list = [eventsi, eventsj, eventsk]

    for events in events_list:
        hist = np.zeros((16, 16), dtype=int)
        np.add.at(hist, (ak.flatten(events.row), ak.flatten(events.col)), 1)
        histograms.append(hist)

    # Configure colormap and text color
    if mc == 'viridis':
        mc = plt.get_cmap('viridis')
        cp = 'w'
    else:
        cp = 'k'

    maxcmap = max(np.max(h) for h in histograms)

    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(33, 11))

    # Plot each layer (reversed order: 2, 1, 0)
    idxs = ['i', 'j', 'k']
    for _,idx in enumerate(idxs):
        hist = histograms[_]

        ax[_].set_title(f"Layer %s"%(idx))
        cax = ax[_].matshow(hist, cmap=mc, vmin=0, vmax=maxcmap)

        # Add text annotations
        for i in range(16):
            for j in range(16):
                if int(hist[i, j]):
                    ax[_].text(j, i, int(hist[i, j]),
                                     ha="center", va="center",
                                     color=cp, fontsize="xx-small", fontweight='bold')

        # Set labels
        if _ == 0:
            ax[_].set_ylabel(r'$Row$')
        ax[_].set_xlabel(r'$Column$')

    fig.colorbar(cax, ax=ax)
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png")
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

    mode_map = np.full((16, 16), np.nan)
    for (r, c), vlist in pixel_vals.items():
        varray = np.array(vlist)
        unique, counts = np.unique(varray, return_counts=True)
        mode_map[r, c] = unique[np.argmax(counts)]  # <- valor real, no índice

    return mode_map

def plotLayerMaps_mode(name, eventsi, eventsj, eventsk, variable, mc='viridis'):
    """
    Plot heatmaps for all three layers showing the mode of a variable per pixel.

    Args:
        name: Output filename (without extension)
        eventsi, eventsj, eventsk: Event data for each layer
        variable: Field name to compute the mode of (e.g. 'cal_code')
        mc: Matplotlib colormap
    """
    events_list = [eventsi, eventsj, eventsk]
    maps = []

    for events in events_list:
        value_map = np.full((16, 16), np.nan)
        rows = ak.to_numpy(ak.flatten(events.row))
        cols = ak.to_numpy(ak.flatten(events.col))
        vals = ak.to_numpy(ak.flatten(events[variable]))

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

    fig, ax = plt.subplots(1, 3, figsize=(33, 11))
    idxs = ['i', 'j', 'k']

    for _, idx in enumerate(idxs):
        value_map = maps[_]
        cmap.set_bad(color='#94a4a2')

        ax[_].set_title(f"Layer {idx} — mode({variable})")
        cax = ax[_].matshow(value_map, cmap=cmap, vmin=vmin, vmax=vmax)

        for i in range(16):
            for j in range(16):
                if not np.isnan(value_map[i, j]):
                    ax[_].text(j, i, f"{int(value_map[i, j])}",
                               ha="center", va="center",
                               color=cp, fontsize="xx-small", fontweight='bold')

        if _ == 0:
            ax[_].set_ylabel(r'$Row$')
        ax[_].set_xlabel(r'$Column$')

    fig.colorbar(cax, ax=ax)
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png")
    plt.close(fig)

def plotLayerMaps_mean(name, eventsi, eventsj, eventsk, variable, mc='viridis'):
    """
    Plot heatmaps for all three layers showing the mean of a variable per pixel.

    Args:
        name: Output filename (without extension)
        eventsi, eventsj, eventsk: Event data for each layer
        variable: Field name to compute the mean of (e.g. 'toa_code')
        mc: Matplotlib colormap
    """
    events_list = [eventsi, eventsj, eventsk]
    maps = []

    for events in events_list:
        value_map = np.full((16, 16), np.nan)
        rows = ak.to_numpy(ak.flatten(events.row))
        cols = ak.to_numpy(ak.flatten(events.col))
        vals = ak.to_numpy(ak.flatten(events[variable]))

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

    fig, ax = plt.subplots(1, 3, figsize=(33, 11))
    idxs = ['i', 'j', 'k']

    for _, idx in enumerate(idxs):
        value_map = maps[_]
        cmap.set_bad(color='#94a4a2')

        ax[_].set_title(f"Layer {idx} — mean({variable})")
        cax = ax[_].matshow(value_map, cmap=cmap, vmin=vmin, vmax=vmax)

        for i in range(16):
            for j in range(16):
                if not np.isnan(value_map[i, j]):
                    ax[_].text(j, i, f"{value_map[i, j]:.0f}",
                               ha="center", va="center",
                               color=cp, fontsize="xx-small", fontweight='bold')

        if _ == 0:
            ax[_].set_ylabel(r'$Row$')
        ax[_].set_xlabel(r'$Column$')

    fig.colorbar(cax, ax=ax)
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png")
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


def plot_correction_fit(dtoa_data, tot_data, tw_corrected, filename, useCubic = True):

        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        # 
        tot_axes = [
            np.linspace(min(tot_data[0]) - 0.5, max(tot_data[0]) + 0.5, 100),
            np.linspace(min(tot_data[1]) - 0.5, max(tot_data[1]) + 0.5, 100),
            np.linspace(min(tot_data[2]) - 0.5, max(tot_data[2]) + 0.5, 100)
        ]
        #
        if useCubic:
            fits = [
                eval_cubic(tot_axes[0], tw_corrected[0]),
                eval_cubic(tot_axes[1], tw_corrected[1]),
                eval_cubic(tot_axes[2], tw_corrected[2])
            ]
        else:
            fits = [
                eval_lineal(tot_axes[0], tw_corrected[0]),
                eval_lineal(tot_axes[1], tw_corrected[1]),
                eval_lineal(tot_axes[2], tw_corrected[2])
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

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":

    # ========================================================================
    # Parse command line arguments
    # ========================================================================

    parser = argparse.ArgumentParser(description='Bootstrap analysis for ETL timing resolution')
    parser.add_argument('--output-dir', type=str, default='',
                       help='Base output directory (default: None)')
    parser.add_argument('--input-dir', type=str, default='',
                       help='Input directory')
    parser.add_argument('--pair', type=str, default='', choices=['', 'ij', 'jk', 'ki'],
                       help='Run only this pair (ij, jk or ki). Default: run all three')
    parser.add_argument('--do-raw', type=int, default=0,
                       help='Fit the raw (pre-TWC-correction) Tij/Tjk/Tki branches instead of the corrected ones')

    args = parser.parse_args()

    # Which pairs to run
    RUN_PAIRS = {'ij', 'jk', 'ki'} if not args.pair else {args.pair}

    # Raw (pre-correction) vs corrected branches/outputs
    DO_RAW = bool(args.do_raw)
    BRANCH_SUFFIX = "_raw" if DO_RAW else ""
    NAME_SUFFIX = "_raw" if DO_RAW else ""

    # Configuration
    BASE_OUTPUT_DIR = args.output_dir
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Identify pairs
    INPUT = f"{args.input_dir}"
    files = glob.glob(f"{INPUT}/corrected_deltaT_*.root")

    print(files)

    all_pairs_ij = set()
    all_pairs_jk = set()
    all_pairs_ki = set()

    for f in files:
        m = re.search(r'corrected_deltaT_(\d{4})_(\d{4})_(\d{4})\.root', f)
        if not m:
            continue
        i, j, k = m.group(1), m.group(2), m.group(3)
        all_pairs_ij.add((i, j))  # ij
        all_pairs_jk.add((j, k))  # jk
        all_pairs_ki.add((k, i))  # ik

    for p in sorted(all_pairs_ij):
        print(f"{p[0]} - {p[1]}")

    if 'ij' in RUN_PAIRS:
        #
        ## Sigma ij
        sigma_ij_results = {}

        for p in tqdm(range(0, len(all_pairs_ij)), desc="Sigma_ij pairs"):
            pair = sorted(all_pairs_ij)[p]

            pixel_i = pair[0] # e.g. 0407 for (4,7)
            pixel_j = pair[1] # e.g. 0507 for (5,7)
            arrays = []

            files = glob.glob(f"{INPUT}/corrected_deltaT_{pixel_i}_{pixel_j}_*.root")

            #print(f'Running cross-sigma i-j extraction for ({pixel_i},{pixel_j}) -> ({len(files)} combinations):')
            for f in files:
                try:
                    with uproot.open(f"{f}:tracks") as tree:
                        arrays.append(tree[f"Tij{BRANCH_SUFFIX}"].array())
                except Exception as e:
                    print(f'Bad file: {f} -> skipping')

            if len(arrays) == 0:
                print(f"WARNING: no valid files for ({pixel_i},{pixel_j}) - skipping")
                continue  

            Tij = ak.concatenate(arrays)

            # Fit 1/2/3-Gaussian mixtures and pick the best one via KS test
            range_ij = (np.mean(Tij) - 4.*np.std(Tij), np.mean(Tij) + 4.*np.std(Tij))
            fits_ij, best_n_ij = fit_gmm_candidates(ak.to_numpy(Tij), bins=25, range_tuple=range_ij)
            x_fit_ij = np.linspace(*range_ij, 1000)
            curves_ij = {n: eval_gaussian_mixture(x_fit_ij, fits_ij[n]['popt'], fits_ij[n]['scale']) for n in fits_ij}
            fit_y_ij = curves_ij[best_n_ij]

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            counts_ij, bin_edges_ij = np.histogram(Tij, bins=25, range=range_ij)
            bin_centers_ij = (bin_edges_ij[:-1] + bin_edges_ij[1:]) / 2

            ax.errorbar(bin_centers_ij, counts_ij, yerr=np.sqrt(counts_ij), fmt='o', markersize=6, linewidth=2, label=r'$T_{ij}^{corr}$', color='k')
            for n_comp, color in CANDIDATE_COLORS.items():
                selected = (n_comp == best_n_ij)
                label = f'{n_comp} Gaussian(s)' + (' (selected)' if selected else f' (KS={fits_ij[n_comp]["ks"]:.3f})')
                ax.plot(x_fit_ij, curves_ij[n_comp], c=color, linewidth=3 if selected else 1.5,
                        linestyle='-' if selected else '--', label=label)
            ax.axhline(y=max(fit_y_ij)/2., color='gray', linestyle=':', linewidth=1, label="fwhm (selected)")
            ax.set_xlabel(r"Corrected $\Delta T_{ij}$")
            ax.set_ylabel("Counts")
            ax.set_ylim([0., 1.5 * max(counts_ij)])
            ax.set_xlim(range_ij)
            ax.legend()
            fig.savefig(f'{BASE_OUTPUT_DIR}/dT_fit_ij{NAME_SUFFIX}_{pair[0]}-{pair[1]}.png', dpi=150)
            plt.close(fig)

            # Compute sigma (from the selected model)
            fwhm_ij, cross1_ij, cross2_ij = compute_fwhm(x_fit_ij, fit_y_ij)
            #print(fwhm_ij, cross1_ij, cross2_ij)
            sigma_ij_ps = fwhm_ij/2.355 * 1000

            # Save to dic for future json
            if str(pair[0]) not in sigma_ij_results.keys():
                sigma_ij_results[str(pair[0])] = {}
        
            sigma_ij_results[pixel_i][pixel_j] = {}
            sigma_ij_results[pixel_i][pixel_j]['fwhm'] = fwhm_ij
            sigma_ij_results[pixel_i][pixel_j]['sigma'] = sigma_ij_ps
            sigma_ij_results[pixel_i][pixel_j]['n'] = len(Tij)
            sigma_ij_results[pixel_i][pixel_j]['n_gaussians'] = best_n_ij

        # Save to JSON
        json_output_path = os.path.join(BASE_OUTPUT_DIR, f'sigma_ij{NAME_SUFFIX}.json')
        with open(json_output_path, 'w') as f:
            json.dump(sigma_ij_results, f, indent=2)

    if 'jk' in RUN_PAIRS:
        #
        ## Sigma jk
        sigma_jk_results = {}

        for p in tqdm(range(0, len(all_pairs_jk)), desc="Sigma_jk pairs"):
            pair = sorted(all_pairs_jk)[p]

            pixel_j = pair[0] # e.g. 0407 for (4,7)
            pixel_k = pair[1] # e.g. 0507 for (5,7)
            arrays = []

            files = glob.glob(f"{INPUT}/corrected_deltaT_*_{pixel_j}_{pixel_k}.root")

            for f in files:
                try:
                    with uproot.open(f"{f}:tracks") as tree:
                        arrays.append(tree[f"Tjk{BRANCH_SUFFIX}"].array())
                except Exception as e:
                    print(f'Bad file: {f} -> skipping')

            if len(arrays) == 0:
                print(f"WARNING: no valid files for ({pixel_j},{pixel_k}) - skipping")
                continue  

            Tjk = ak.concatenate(arrays)

            # Fit 1/2/3-Gaussian mixtures and pick the best one via KS test
            range_jk = (np.mean(Tjk) - 4.*np.std(Tjk), np.mean(Tjk) + 4.*np.std(Tjk))
            fits_jk, best_n_jk = fit_gmm_candidates(ak.to_numpy(Tjk), bins=25, range_tuple=range_jk)
            x_fit_jk = np.linspace(*range_jk, 1000)
            curves_jk = {n: eval_gaussian_mixture(x_fit_jk, fits_jk[n]['popt'], fits_jk[n]['scale']) for n in fits_jk}
            fit_y_jk = curves_jk[best_n_jk]

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            counts_jk, bin_edges_jk = np.histogram(Tjk, bins=25, range=range_jk)
            bin_centers_jk = (bin_edges_jk[:-1] + bin_edges_jk[1:]) / 2

            ax.errorbar(bin_centers_jk, counts_jk, yerr=np.sqrt(counts_jk), fmt='o', markersize=6, linewidth=2, label=r'$T_{jk}^{corr}$', color='k')
            for n_comp, color in CANDIDATE_COLORS.items():
                selected = (n_comp == best_n_jk)
                label = f'{n_comp} Gaussian(s)' + (' (selected)' if selected else f' (KS={fits_jk[n_comp]["ks"]:.3f})')
                ax.plot(x_fit_jk, curves_jk[n_comp], c=color, linewidth=3 if selected else 1.5,
                        linestyle='-' if selected else '--', label=label)
            ax.axhline(y=max(fit_y_jk)/2., color='gray', linestyle=':', linewidth=1, label="fwhm (selected)")
            ax.set_xlabel(r"Corrected $\Delta T_{jk}$")
            ax.set_ylabel("Counts")
            ax.set_ylim([0., 1.5 * max(counts_jk)])
            ax.set_xlim(range_jk)
            ax.legend()
            fig.savefig(f'{BASE_OUTPUT_DIR}/dT_fit_jk{NAME_SUFFIX}_{pair[0]}-{pair[1]}.png', dpi=150)
            plt.close(fig)

            # Compute sigma (from the selected model)
            fwhm_jk, cross1_jk, cross2_jk = compute_fwhm(x_fit_jk, fit_y_jk)
            sigma_jk_ps = fwhm_jk/2.355 * 1000

            # Save to dic for future json
            if str(pair[0]) not in sigma_jk_results.keys():
                sigma_jk_results[pixel_j] = {}

            sigma_jk_results[pixel_j][pixel_k] = {}
            sigma_jk_results[pixel_j][pixel_k]['fwhm'] = fwhm_jk
            sigma_jk_results[pixel_j][pixel_k]['sigma'] = sigma_jk_ps
            sigma_jk_results[pixel_j][pixel_k]['n'] = len(Tjk)
            sigma_jk_results[pixel_j][pixel_k]['n_gaussians'] = best_n_jk

        # Save to JSON
        json_output_path = os.path.join(BASE_OUTPUT_DIR, f'sigma_jk{NAME_SUFFIX}.json')
        with open(json_output_path, 'w') as f:
            json.dump(sigma_jk_results, f, indent=2)



    if 'ki' in RUN_PAIRS:
        #
        ## Sigma jk
        sigma_ki_results = {}

        for p in tqdm(range(0, len(all_pairs_ki)), desc="Sigma_ki pairs"):
            pair = sorted(all_pairs_ki)[p]

            pixel_k = pair[0] # e.g. 0407 for (4,7)
            pixel_i = pair[1] # e.g. 0507 for (5,7)
            arrays = []

            files = glob.glob(f"{INPUT}/corrected_deltaT_{pixel_i}_*_{pixel_k}.root")

            for f in files:
                try:
                    with uproot.open(f"{f}:tracks") as tree:
                        arrays.append(tree[f"Tki{BRANCH_SUFFIX}"].array())
                except Exception as e:
                    print(f'Bad file: {f} -> skipping')

            if len(arrays) == 0:
                print(f"WARNING: no valid files for ({pixel_k},{pixel_i}) - skipping")
                continue  

            Tki = ak.concatenate(arrays)

            # Fit 1/2/3-Gaussian mixtures and pick the best one via KS test
            range_ki = (np.mean(Tki) - 4.*np.std(Tki), np.mean(Tki) + 4.*np.std(Tki))
            fits_ki, best_n_ki = fit_gmm_candidates(ak.to_numpy(Tki), bins=25, range_tuple=range_ki)
            x_fit_ki = np.linspace(*range_ki, 1000)
            curves_ki = {n: eval_gaussian_mixture(x_fit_ki, fits_ki[n]['popt'], fits_ki[n]['scale']) for n in fits_ki}
            fit_y_ki = curves_ki[best_n_ki]

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            counts_ki, bin_edges_ki = np.histogram(Tki, bins=25, range=range_ki)
            bin_centers_ki = (bin_edges_ki[:-1] + bin_edges_ki[1:]) / 2

            ax.errorbar(bin_centers_ki, counts_ki, yerr=np.sqrt(counts_ki), fmt='o', markersize=6, linewidth=2, label=r'$T_{ki}^{corr}$', color='k')
            for n_comp, color in CANDIDATE_COLORS.items():
                selected = (n_comp == best_n_ki)
                label = f'{n_comp} Gaussian(s)' + (' (selected)' if selected else f' (KS={fits_ki[n_comp]["ks"]:.3f})')
                ax.plot(x_fit_ki, curves_ki[n_comp], c=color, linewidth=3 if selected else 1.5,
                        linestyle='-' if selected else '--', label=label)
            ax.axhline(y=max(fit_y_ki)/2., color='gray', linestyle=':', linewidth=1, label="fwhm (selected)")
            ax.set_xlabel(r"Corrected $\Delta T_{ki}$")
            ax.set_ylabel("Counts")
            ax.set_ylim([0., 1.5 * max(counts_ki)])
            ax.set_xlim(range_ki)
            ax.legend()
            fig.savefig(f'{BASE_OUTPUT_DIR}/dT_fit_ki{NAME_SUFFIX}_{pixel_k}-{pixel_i}.png', dpi=150)
            plt.close(fig)

            # Compute sigma (from the selected model)
            fwhm_ki, cross1_ki, cross2_ki = compute_fwhm(x_fit_ki, fit_y_ki)
            sigma_ki_ps = fwhm_ki/2.355 * 1000

            # Save to dic for future json
            if str(pixel_k) not in sigma_ki_results.keys():
                sigma_ki_results[pixel_k] = {}

            sigma_ki_results[pixel_k][pixel_i] = {}
            sigma_ki_results[pixel_k][pixel_i]['fwhm'] = fwhm_ki
            sigma_ki_results[pixel_k][pixel_i]['sigma'] = sigma_ki_ps
            sigma_ki_results[pixel_k][pixel_i]['n'] = len(Tki)
            sigma_ki_results[pixel_k][pixel_i]['n_gaussians'] = best_n_ki

        # Save to JSON
        json_output_path = os.path.join(BASE_OUTPUT_DIR, f'sigma_ki{NAME_SUFFIX}.json')
        with open(json_output_path, 'w') as f:
            json.dump(sigma_ki_results, f, indent=2)



