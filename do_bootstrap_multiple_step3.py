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
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
import os
import json
import argparse
from tqdm import tqdm
import mplhep as hep
import glob, re
hep.style.use("CMS")

COLORS = ['#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6',
          '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd']

# Set from --doGlobal in main(). Pixel keys in the sigma_*.json files (from
# step2) already reflect whichever coordinate choice step1/step2 used - this
# only needs to match that grid size so the (N_PIX, N_PIX) maps built here
# are large enough: row/col go 0..15 (16 values) normally, or 0..31 (32
# values) with row_global/col_global, matching do_bootstrap_multiple_step1.py.
DO_GLOBAL_COORDINATES = False
N_PIX                 = 16

# Hardcoded pixel masks (row, col) used to select the pixels that go into the
# Gaussian fit for the mean resolution and sigma of each layer.
# Square (1,2)-(1,14)-(12,14)-(12,1)
MASK_I = [
    (1, 2),  (1, 3),  (1, 4),  (1, 5),  (1, 6),  (1, 7),  (1, 8),  (1, 9),  (1, 10),  (1, 11),  (1, 12),  (1, 13),  (1, 14),
    (2, 2),  (2, 3),  (2, 4),  (2, 5),  (2, 6),  (2, 7),  (2, 8),  (2, 9),  (2, 10),  (2, 11),  (2, 12),  (2, 13),  (2, 14),
    (3, 2),  (3, 3),  (3, 4),  (3, 5),  (3, 6),  (3, 7),  (3, 8),  (3, 9),  (3, 10),  (3, 11),  (3, 12),  (3, 13),  (3, 14),
    (4, 2),  (4, 3),  (4, 4),  (4, 5),  (4, 6),  (4, 7),  (4, 8),  (4, 9),  (4, 10),  (4, 11),  (4, 12),  (4, 13),  (4, 14),
    (5, 2),  (5, 3),  (5, 4),  (5, 5),  (5, 6),  (5, 7),  (5, 8),  (5, 9),  (5, 10),  (5, 11),  (5, 12),  (5, 13),  (5, 14),
    (6, 2),  (6, 3),  (6, 4),  (6, 5),  (6, 6),  (6, 7),  (6, 8),  (6, 9),  (6, 10),  (6, 11),  (6, 12),  (6, 13),  (6, 14),
    (7, 2),  (7, 3),  (7, 4),  (7, 5),  (7, 6),  (7, 7),  (7, 8),  (7, 9),  (7, 10),  (7, 11),  (7, 12),  (7, 13),  (7, 14),
    (8, 2),  (8, 3),  (8, 4),  (8, 5),  (8, 6),  (8, 7),  (8, 8),  (8, 9),  (8, 10),  (8, 11),  (8, 12),  (8, 13),  (8, 14),
    (9, 2),  (9, 3),  (9, 4),  (9, 5),  (9, 6),  (9, 7),  (9, 8),  (9, 9),  (9, 10),  (9, 11),  (9, 12),  (9, 13),  (9, 14),
    (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14),
    (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14),
    (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14),
]
MASK_I = None
# Square (1,2)-(1,14)-(12,14)-(12,1)
MASK_J = [
    (1, 2),  (1, 3),  (1, 4),  (1, 5),  (1, 6),  (1, 7),  (1, 8),  (1, 9),  (1, 10),  (1, 11),  (1, 12),  (1, 13),  (1, 14),
    (2, 2),  (2, 3),  (2, 4),  (2, 5),  (2, 6),  (2, 7),  (2, 8),  (2, 9),  (2, 10),  (2, 11),  (2, 12),  (2, 13),  (2, 14),
    (3, 2),  (3, 3),  (3, 4),  (3, 5),  (3, 6),  (3, 7),  (3, 8),  (3, 9),  (3, 10),  (3, 11),  (3, 12),  (3, 13),  (3, 14),
    (4, 2),  (4, 3),  (4, 4),  (4, 5),  (4, 6),  (4, 7),  (4, 8),  (4, 9),  (4, 10),  (4, 11),  (4, 12),  (4, 13),  (4, 14),
    (5, 2),  (5, 3),  (5, 4),  (5, 5),  (5, 6),  (5, 7),  (5, 8),  (5, 9),  (5, 10),  (5, 11),  (5, 12),  (5, 13),  (5, 14),
    (6, 2),  (6, 3),  (6, 4),  (6, 5),  (6, 6),  (6, 7),  (6, 8),  (6, 9),  (6, 10),  (6, 11),  (6, 12),  (6, 13),  (6, 14),
    (7, 2),  (7, 3),  (7, 4),  (7, 5),  (7, 6),  (7, 7),  (7, 8),  (7, 9),  (7, 10),  (7, 11),  (7, 12),  (7, 13),  (7, 14),
    (8, 2),  (8, 3),  (8, 4),  (8, 5),  (8, 6),  (8, 7),  (8, 8),  (8, 9),  (8, 10),  (8, 11),  (8, 12),  (8, 13),  (8, 14),
    (9, 2),  (9, 3),  (9, 4),  (9, 5),  (9, 6),  (9, 7),  (9, 8),  (9, 9),  (9, 10),  (9, 11),  (9, 12),  (9, 13),  (9, 14),
    (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14),
    (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14),
    (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14),
]
MASK_J = None
# Square (1,2)-(1,14)-(12,14)-(12,1)
MASK_K = [
    (1, 2),  (1, 3),  (1, 4),  (1, 5),  (1, 6),  (1, 7),  (1, 8),  (1, 9),  (1, 10),  (1, 11),  (1, 12),  (1, 13),  (1, 14),
    (2, 2),  (2, 3),  (2, 4),  (2, 5),  (2, 6),  (2, 7),  (2, 8),  (2, 9),  (2, 10),  (2, 11),  (2, 12),  (2, 13),  (2, 14),
    (3, 2),  (3, 3),  (3, 4),  (3, 5),  (3, 6),  (3, 7),  (3, 8),  (3, 9),  (3, 10),  (3, 11),  (3, 12),  (3, 13),  (3, 14),
    (4, 2),  (4, 3),  (4, 4),  (4, 5),  (4, 6),  (4, 7),  (4, 8),  (4, 9),  (4, 10),  (4, 11),  (4, 12),  (4, 13),  (4, 14),
    (5, 2),  (5, 3),  (5, 4),  (5, 5),  (5, 6),  (5, 7),  (5, 8),  (5, 9),  (5, 10),  (5, 11),  (5, 12),  (5, 13),  (5, 14),
    (6, 2),  (6, 3),  (6, 4),  (6, 5),  (6, 6),  (6, 7),  (6, 8),  (6, 9),  (6, 10),  (6, 11),  (6, 12),  (6, 13),  (6, 14),
    (7, 2),  (7, 3),  (7, 4),  (7, 5),  (7, 6),  (7, 7),  (7, 8),  (7, 9),  (7, 10),  (7, 11),  (7, 12),  (7, 13),  (7, 14),
    (8, 2),  (8, 3),  (8, 4),  (8, 5),  (8, 6),  (8, 7),  (8, 8),  (8, 9),  (8, 10),  (8, 11),  (8, 12),  (8, 13),  (8, 14),
    (9, 2),  (9, 3),  (9, 4),  (9, 5),  (9, 6),  (9, 7),  (9, 8),  (9, 9),  (9, 10),  (9, 11),  (9, 12),  (9, 13),  (9, 14),
    (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14),
    (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14),
    (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14),
]
MASK_K = None


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

def eval_gaussian_mixture(x, coeffs, scale):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(coeffs), 3):
        amp, mu, sigma = coeffs[i], coeffs[i+1], coeffs[i+2]
        y += scale * amp / (sigma * np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y

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


def fit_histogram_gaussian(data, bins=40, range_tuple=(-1, 1), fitType="Gaussian", method="binned"):
    """
    Fit a Gaussian to data.

    Args:
        data: Input data array
        bins: Number of bins (histogram is only used for the initial guess and,
              in "binned" mode, for the fit itself; in "unbinned" mode it is
              only used to rescale amp for overlaying the fit on the histogram)
        range_tuple: Histogram/fit range
        fitType: "Gaussian" or "DoubleGaussian"
        method: "binned" (chi2 fit of the model to histogram counts via curve_fit)
                or "unbinned" (maximum-likelihood fit to the raw data)

    Returns:
        popt: Optimal parameters (amp, mean, sigma) or
              (amp, frac, mean1, mean2, sigma1, sigma2) for DoubleGaussian
        pcov: Covariance matrix (None for "unbinned")
        bin_centers: Bin centers
        counts: Histogram counts
    """
    data = np.asarray(data)
    if range_tuple is not None:
        data = data[(data >= range_tuple[0]) & (data <= range_tuple[1])]

    counts, bin_edges = np.histogram(data, bins=bins, range=range_tuple)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    amp0   = max(counts)
    mean0  = bin_centers[np.argmax(counts)]
    # sigma: half-width
    half_max = amp0 / 2
    above = bin_centers[counts > half_max]
    sigma0 = (above[-1] - above[0]) / 2 if len(above) > 1 else np.std(data)

    if fitType=="Gaussian":
        if method == "unbinned":
            # Exact MLE for a Gaussian: sample mean and (biased) std of the raw data.
            mean_mle, sigma_mle = norm.fit(data)
            # amp is not a fit parameter here; it only rescales the curve so it
            # overlays nicely on the (binned) histogram used for plotting.
            amp = len(data) * bin_width / (sigma_mle * np.sqrt(2 * np.pi))
            popt = np.array([amp, mean_mle, sigma_mle])
            pcov = None
        else:
            # Initial guess for Gaussian parameters
            p0 = [amp0, mean0, sigma0]

            # Fit Gaussian
            popt, pcov = curve_fit(
                gaussian,
                bin_centers,
                counts,
                p0=p0,
                bounds=(
                    [0, range_tuple[0], 0],
                    [np.inf, range_tuple[1], range_tuple[1]-range_tuple[0]]
                )
            )

    elif fitType=="DoubleGaussian":
        if method == "unbinned":
            def neg_log_likelihood(params):
                frac, mean1, mean2, sigma1, sigma2 = params
                if not (0 < frac < 1) or sigma1 <= 0 or sigma2 <= 0:
                    return np.inf
                pdf = (frac * norm.pdf(data, mean1, sigma1)
                       + (1 - frac) * norm.pdf(data, mean2, sigma2))
                return -np.sum(np.log(np.clip(pdf, 1e-300, None)))

            p0 = [0.95, mean0, mean0, sigma0, 6*sigma0]
            res = minimize(neg_log_likelihood, p0, method="Nelder-Mead")
            frac, mean1, mean2, sigma1, sigma2 = res.x
            amp = len(data) * bin_width
            popt = np.array([amp, frac, mean1, mean2, sigma1, sigma2])
            pcov = None
        else:
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


def load_json(path):
    with open(path) as f:
        return json.load(f)

# Physical 2x2 layout of the 4 ETROC ASICs on a module, each contributing its
# own local (16,16) threshold map. Confirmed by the user: etroc_0=top-left,
# etroc_2=top-right, etroc_1=bottom-left, etroc_3=bottom-right. "Top"/"left"
# here means low row/col index, matching matshow's default origin (row 0 at
# the top) used by _plot_sigma_heatmap. Assumes each ETROC's local (row, col)
# maps directly into its quadrant with no additional flip/rotation.
ETROC_QUADRANTS = {0: (0, 0), 2: (0, 1), 1: (1, 0), 3: (1, 1)}

def stitch_global_threshold_map(thresholds, rb_prefix, field, n_pix):
    """
    Stitch the 4 per-ETROC (16,16) threshold maps (e.g. "baseline" or
    "noise_width") of one module into a single (n_pix,n_pix) global map,
    placed according to ETROC_QUADRANTS. Used only in --doGlobal mode, where
    sigma_map is (n_pix,n_pix) too and a single ETROC's map is no longer the
    right shape to correlate against it.
    """
    half = n_pix // 2
    global_map = np.full((n_pix, n_pix), np.nan)
    for entry, data in thresholds.items():
        if rb_prefix not in entry:
            continue
        m = re.search(r'etroc_(\d+)', entry)
        if not m or int(m.group(1)) not in ETROC_QUADRANTS:
            continue
        row_q, col_q = ETROC_QUADRANTS[int(m.group(1))]
        r0, c0 = row_q * half, col_q * half
        global_map[r0:r0 + half, c0:c0 + half] = np.array(data[field])
    return global_map

def plot_sigma_vs_log(sigma_map, noise_map, baseline_map, output_dir, layer):
    """Scatter plot of sigma_i vs noise_width and baseline for analyzed pixels."""
    sigmas, noises, baselines, labels = [], [], [], []

    sigmas    = np.array(sigma_map)
    noises    = np.array(noise_map)
    baselines = np.array(baseline_map)

    flat_sigma = []
    flat_noise = []
    flat_bases = []

    for i in range(0, N_PIX):
        for j in range(0, N_PIX):
            if not np.isnan(noises[i,j]) and not np.isnan(baselines[i,j]) and not np.isnan(sigmas[i, j]):

                flat_sigma.append(sigmas[i, j])
                flat_noise.append(noises[i,j])
                flat_bases.append(baselines[i,j])

    flat_sigma = np.array(flat_sigma)
    flat_noise = np.array(flat_noise)
    flat_bases = np.array(flat_bases)

    # sigma_i vs noise_width
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.scatter(noises, sigmas, color=COLORS[0], s=60, zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (noises[i], sigmas[i]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_ylabel(f"Layer {layer} $\\sigma$ (ps)")
    ax.set_title(f"$\\sigma_i$ vs noise_width")
    fig.savefig(f"{output_dir}/noise_vs_sigma_layer{layer}.pdf")
    fig.savefig(f"{output_dir}/noise_vs_sigma_layer{layer}.png", dpi=150)
    plt.close(fig)
    #
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist2d(flat_noise, flat_sigma, bins=[len(np.unique(noises)), 20], cmap='viridis')
    ax.set_ylabel(f"Layer {layer} $\\sigma$ (ps)")
    ax.set_title(f"$\\sigma_i$ vs noise_width")
    fig.savefig(f"{output_dir}/noise_vs_sigma_layer{layer}_hist.pdf")
    fig.savefig(f"{output_dir}/noise_vs_sigma_layer{layer}_hist.png", dpi=150)
    plt.close(fig)
    

    # sigma_i vs baseline
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.scatter(baselines, sigmas, color=COLORS[1], s=60, zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (baselines[i], sigmas[i]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"Baseline")
    ax.set_ylabel(f"Layer {layer} $\\sigma$ (ps)")
    ax.set_title(f"$\\sigma$ vs baseline")
    ax.set_ylim([0., 100.])
    fig.savefig(f"{output_dir}/baseline_vs_sigma_layer{layer}.pdf")
    fig.savefig(f"{output_dir}/baseline_vs_sigma_layer{layer}.png", dpi=150)
    plt.close(fig)
    #
    print(f"baselines: min={np.nanmin(baselines)}, max={np.nanmax(baselines)}, nans={np.sum(np.isnan(baselines))}, len={len(baselines)}")
    print(f"sigmas:    min={np.nanmin(sigmas)},    max={np.nanmax(sigmas)},    nans={np.sum(np.isnan(sigmas))},    len={len(sigmas)}")
    #
    #fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    #ax.hist2d(baselines, sigmas, bins=[30, 15], cmap='viridis')
    #ax.set_xlabel(f"Baseline")
    #ax.set_ylabel(f"Layer {layer} $\\sigma$ (ps)")
    #ax.set_title(f"$\\sigma$ vs baseline")
    #fig.savefig(f"test_baseline.pdf")
    #fig.savefig(f"test_baseline.png", dpi=150)
    #plt.close(fig)
    #print(f"Saved: test_baseline.png")

def _masked_valid_sigmas(sigma_map, mask):
    """Flatten a (N_PIX,N_PIX) sigma map to the values inside `mask` (or all valid
    pixels if mask is None), dropping NaNs."""
    if mask is None:
        return sigma_map[~np.isnan(sigma_map)]
    values = np.array([sigma_map[row, col] for row, col in mask])
    return values[~np.isnan(values)]


def _plot_sigma_heatmap(sigma_map, moduleid, output_dir, stat_mean, stat_std, suffix="", label_suffix=""):
    """Standalone (N_PIX,N_PIX) resolution heatmap, saved as sigma_{moduleid}_resolution{suffix}.png/pdf."""
    # Same scaling/style as plotLayerMaps() in do_bootstrap_multiple_step1.py, so
    # doGlobal heatmaps look consistent across the two scripts. Font sizes are
    # only overridden in --doGlobal mode (values checked by eye at N_PIX=32);
    # non-global keeps the original defaults untouched.
    scale = 4 if DO_GLOBAL_COORDINATES else 1

    fig, ax = plt.subplots(1, 1, figsize=(12 * scale, 11 * scale))

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='lightgray')  # Color for NaN values (unanalyzed pixels)

    vmin, vmax = 0, 100
    im = ax.matshow(sigma_map, cmap=cmap, vmin=vmin, vmax=vmax)

    annot_fontsize = 24 if DO_GLOBAL_COORDINATES else 16
    for i in range(N_PIX):
        for j in range(N_PIX):
            if not np.isnan(sigma_map[i, j]):
                ax.text(j, i, f'{sigma_map[i, j]:.0f}',
                        ha="center", va="center",
                        color='white', fontsize=annot_fontsize, fontweight='bold')

    if DO_GLOBAL_COORDINATES:
        ax.set_xlabel(r'$Column$', fontsize=24)
        ax.set_ylabel(r'$Row$', fontsize=24)
        ax.tick_params(labelsize=20)
    else:
        ax.set_xlabel(r'$Column$')
        ax.set_ylabel(r'$Row$')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if DO_GLOBAL_COORDINATES:
        cbar.set_label(r'$\sigma$ (ps)', rotation=90, fontsize=20)
        cbar.ax.tick_params(labelsize=16)
    else:
        cbar.set_label(r'$\sigma$ (ps)', rotation=90)

    #hep.cms.text(exp="SPS May TB", text="", ax=ax)
    #hep.cms.lumitext(f'Module {moduleid}{label_suffix}, average: {stat_mean:.1f} ' + r'$\pm$' + f' {stat_std:.1f}',
    #                  ax=ax, fontsize=26, fontname=None)
    hep.cms.label("ETL Preliminary", data=True, ax=ax, rlabel='',
                   **({'fontsize': 18} if DO_GLOBAL_COORDINATES else {}))

    hep.cms.lumitext(f'Module {moduleid}{label_suffix} - {stat_mean:.1f} ' + r'$\pm$' + f' {stat_std:.1f}',
                      ax=ax, fontsize=24, fontname=None)

    fig.savefig(f"{output_dir}/sigma_{moduleid}_resolution{suffix}.pdf")
    fig.savefig(f"{output_dir}/sigma_{moduleid}_resolution{suffix}.png", dpi=150)
    plt.close(fig)


def plot_resolution_plots(sigma_map, info, output_dir, mask, sigma_map_raw=None):
    """
    Build the resolution heatmap and histogram (with Gaussian fit) for a layer.

    If `sigma_map_raw` (the same per-pixel resolution map, but computed from the
    raw/uncorrected Tij/Tjk/Tki) is provided, it gets its own heatmap and is
    overlaid, with its own fit, on the same histogram plot as the corrected one.
    """

    moduleid = info.split('_')[3]

    sigmas_for_stats = _masked_valid_sigmas(sigma_map, mask)

    if len(sigmas_for_stats) < 2:
        print(f"[plot_resolution_plots] WARNING: module {moduleid} has no valid pixels in the mask "
              f"({len(sigmas_for_stats)} found) - skipping Gaussian fit and plots.")
        return np.nan, np.nan

    fit_bins = 50
    fit_range = (0, 140)
    popt, pcov, _, _ = fit_histogram_gaussian(sigmas_for_stats, bins=fit_bins, range_tuple=fit_range, method="binned")
    stat_mean = popt[1]
    stat_std  = abs(popt[2])

    _plot_sigma_heatmap(sigma_map, moduleid, output_dir, stat_mean, stat_std)

    # Raw (uncorrected) map + fit, only if it was actually passed in.
    popt_raw = None
    sigmas_for_stats_raw = None
    if sigma_map_raw is not None:
        sigmas_for_stats_raw = _masked_valid_sigmas(sigma_map_raw, mask)
        if len(sigmas_for_stats_raw) < 2:
            print(f"[plot_resolution_plots] WARNING: module {moduleid} has no valid raw pixels in the mask "
                  f"({len(sigmas_for_stats_raw)} found) - skipping raw Gaussian fit and plots.")
            sigmas_for_stats_raw = None
        else:
            popt_raw, _, _, _ = fit_histogram_gaussian(sigmas_for_stats_raw, bins=fit_bins, range_tuple=fit_range, method="binned")
            stat_mean_raw = popt_raw[1]
            stat_std_raw  = abs(popt_raw[2])
            _plot_sigma_heatmap(sigma_map_raw, moduleid, output_dir, stat_mean_raw, stat_std_raw,
                                 suffix="_raw", label_suffix=" (raw)")

    # Histogram (masked pixels only), corrected + raw overlaid, each with its own Gaussian fit
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    bins = np.linspace(*fit_range, fit_bins + 1)
    x_fit = np.linspace(*fit_range, 500)

    ax.hist(sigmas_for_stats, bins=bins, histtype="stepfilled", alpha=0.6,
            color="teal", label="Data (corrected)")
    ax.plot(x_fit, gaussian(x_fit, *popt), color='darkslategray', linewidth=2, label="Gaussian fit (corrected)")

    ax.set_xlabel("$\\sigma$ (ps)")
    ax.set_ylabel("Pixels")

    hep.cms.label("ETL Preliminary", data=True, ax=ax, rlabel='')

    ax.text(0.05, 0.6, 'Corrected:', fontsize=16, color='teal', transform=ax.transAxes)
    ax.text(0.05, 0.55, f'$\\mu$ = {stat_mean:.1f} ps', fontsize=16, color='teal', transform=ax.transAxes)
    ax.text(0.05, 0.50, f'$\\sigma$ = {stat_std:.1f} ps', fontsize=16, color='teal', transform=ax.transAxes)

    if sigmas_for_stats_raw is not None:
        ax.hist(sigmas_for_stats_raw, bins=bins, histtype="stepfilled", alpha=0.4,
                color='orange', label="Data (raw)")
        ax.plot(x_fit, gaussian(x_fit, *popt_raw), color='orange', linewidth=2, linestyle="--", label="Gaussian fit (raw)")
        ax.text(0.05, 0.4, 'Uncorrected:', fontsize=16, color='orange', transform=ax.transAxes)
        ax.text(0.05, 0.35, f'$\\mu$ = {stat_mean_raw:.1f} ps', fontsize=16, color='orange', transform=ax.transAxes)
        ax.text(0.05, 0.3, f'$\\sigma$ = {stat_std_raw:.1f} ps', fontsize=16, color='orange', transform=ax.transAxes)

    ax.legend(loc="upper right")

    fig.savefig(f"{output_dir}/sigma_{moduleid}_resolution_histogram.pdf")
    fig.savefig(f"{output_dir}/sigma_{moduleid}_resolution_histogram.png", dpi=150)
    plt.close(fig)

    return stat_mean, stat_std


def compute_layer_sigma_map(layer, sigma_ij_results, sigma_jk_results, sigma_ki_results, min_n=200):
    """
    Compute the per-pixel resolution map for one layer ('i', 'j' or 'k') out of the
    three pairwise sigma jsons (sigma_ij, sigma_jk, sigma_ki), using the standard
    three-technique decomposition sigma_T^2 = 0.5*(sigma_TN^2 + sigma_PT^2 - sigma_NP^2),
    where N/P are the next/previous layers in the cyclic order i->j->k->i.

    For each pixel of the target layer, the combo (over all matching pixel triplets)
    with the largest total statistics is kept, provided each of the three pairwise
    sigmas it comes from has more than `min_n` entries.
    """
    dicts = {'ij': sigma_ij_results, 'jk': sigma_jk_results, 'ki': sigma_ki_results}
    if layer == 'i':
        d_TN, d_NP, d_PT = dicts['ij'], dicts['jk'], dicts['ki']
    elif layer == 'j':
        d_TN, d_NP, d_PT = dicts['jk'], dicts['ki'], dicts['ij']
    elif layer == 'k':
        d_TN, d_NP, d_PT = dicts['ki'], dicts['ij'], dicts['jk']
    else:
        raise ValueError(f"Unknown layer: {layer}")

    sigma_map  = np.full((N_PIX, N_PIX), np.nan)
    number_map = np.full((N_PIX, N_PIX), np.nan)

    for pixel_T in d_TN.keys():
        row, col = int(pixel_T[:2]), int(pixel_T[2:])

        values, numbers, n_TNs, n_NPs, n_PTs = [], [], [], [], []

        for pixel_N in d_TN[pixel_T].keys():
            if pixel_N not in d_NP:
                continue
            for pixel_P in d_NP[pixel_N].keys():
                if pixel_P not in d_PT or pixel_T not in d_PT[pixel_P]:
                    continue

                sigma_TN = d_TN[pixel_T][pixel_N]['sigma']
                sigma_NP = d_NP[pixel_N][pixel_P]['sigma']
                sigma_PT = d_PT[pixel_P][pixel_T]['sigma']

                n_TN = d_TN[pixel_T][pixel_N]['n']
                n_NP = d_NP[pixel_N][pixel_P]['n']
                n_PT = d_PT[pixel_P][pixel_T]['n']

                resolution = (0.5)**0.5 * (sigma_TN**2 + sigma_PT**2 - sigma_NP**2)**0.5
                number = n_TN + n_NP + n_PT

                values.append(resolution)
                numbers.append(number)
                n_TNs.append(n_TN)
                n_NPs.append(n_NP)
                n_PTs.append(n_PT)

        if len(numbers) > 0:
            best_idx = np.argmax(numbers)
            best_resolution = values[best_idx]
            best_number     = numbers[best_idx]

            if n_TNs[best_idx] > min_n and n_NPs[best_idx] > min_n and n_PTs[best_idx] > min_n and not np.iscomplex(best_resolution):
                sigma_map[row, col]  = best_resolution
                number_map[row, col] = best_number

    return sigma_map, number_map



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
    parser.add_argument('--run-start', type=str, default='',
                       help='Runs')
    parser.add_argument('--doGlobal', action='store_true',
                   help="Use row_global/col_global (range 0..31) instead of row/col "
                        "(range 0..16). Must match what step1/step2 used to produce "
                        "the input sigma_*.json files.")

    args = parser.parse_args()

    # Configuration
    DO_GLOBAL_COORDINATES = args.doGlobal
    N_PIX                 = 32 if DO_GLOBAL_COORDINATES else 16

    BASE_OUTPUT_DIR = args.output_dir
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    RUN = args.run_start # expect something like 135958_136884

    # Load logs:
    LOG_BASEDIR = '/eos/project/m/mtd-etl-system-test/public/Test Beam Data/SPS_April_2026/run_logs/'
    LOG_BASEDIR = '/eos/project/m/mtd-etl-system-test/public/Test Beam Data/PrepSPSJuly2026/run_logs/'
    for f in os.listdir(LOG_BASEDIR):
        if RUN in f:
            run_log = load_json(f'{LOG_BASEDIR}/{f}')
    for entry in run_log['thresholds'].keys():
        if 'rb_1' in entry:
            info_i = entry
            baseline_i = run_log['thresholds'][entry]['baseline']
            noise_i = run_log['thresholds'][entry]['noise_width']
        if 'rb_2' in entry:
            info_j = entry
            baseline_j = run_log['thresholds'][entry]['baseline']
            noise_j = run_log['thresholds'][entry]['noise_width']
        if 'rb_3' in entry:
            info_k = entry
            baseline_k = run_log['thresholds'][entry]['baseline']
            noise_k = run_log['thresholds'][entry]['noise_width']

    # In --doGlobal mode, sigma_map_* is (N_PIX,N_PIX) covering all 4 ETROCs of
    # the module, so the single-ETROC (16,16) baseline_*/noise_* picked above
    # (whichever entry the loop landed on last) are the wrong shape to
    # correlate against it. Stitch all 4 ETROCs into a matching (N_PIX,N_PIX)
    # map instead - info_*/moduleid is unaffected since it's the same module
    # number for all 4 ETROCs of a given rb.
    if DO_GLOBAL_COORDINATES:
        baseline_i = stitch_global_threshold_map(run_log['thresholds'], 'rb_1', 'baseline', N_PIX)
        noise_i    = stitch_global_threshold_map(run_log['thresholds'], 'rb_1', 'noise_width', N_PIX)
        baseline_j = stitch_global_threshold_map(run_log['thresholds'], 'rb_2', 'baseline', N_PIX)
        noise_j    = stitch_global_threshold_map(run_log['thresholds'], 'rb_2', 'noise_width', N_PIX)
        baseline_k = stitch_global_threshold_map(run_log['thresholds'], 'rb_3', 'baseline', N_PIX)
        noise_k    = stitch_global_threshold_map(run_log['thresholds'], 'rb_3', 'noise_width', N_PIX)


    # Identify pairs
    INPUT = f"{args.input_dir}"

    with open(os.path.join(args.input_dir, 'sigma_ij.json'), 'r') as f:
        sigma_ij_results = json.load(f)
    with open(os.path.join(args.input_dir, 'sigma_jk.json'), 'r') as f:
        sigma_jk_results = json.load(f)
    with open(os.path.join(args.input_dir, 'sigma_ki.json'), 'r') as f:
        sigma_ki_results = json.load(f)

    # Raw (pre-TWC-correction) pairwise sigmas: produced by step2 with --do-raw 1.
    # Optional - only used if all three files are found, so this stays a no-op
    # for older/partial input directories.
    raw_paths = {
        'ij': os.path.join(args.input_dir, 'sigma_ij_raw.json'),
        'jk': os.path.join(args.input_dir, 'sigma_jk_raw.json'),
        'ki': os.path.join(args.input_dir, 'sigma_ki_raw.json'),
    }
    HAS_RAW = all(os.path.exists(p) for p in raw_paths.values())
    if HAS_RAW:
        print(f"Found raw sigma jsons in {args.input_dir} - will also produce raw resolution plots.")
        with open(raw_paths['ij'], 'r') as f:
            sigma_ij_results_raw = json.load(f)
        with open(raw_paths['jk'], 'r') as f:
            sigma_jk_results_raw = json.load(f)
        with open(raw_paths['ki'], 'r') as f:
            sigma_ki_results_raw = json.load(f)
    else:
        print(f"Raw sigma jsons not found in {args.input_dir} - skipping raw resolution plots.")

    #
    # Layers i, j, k
    #
    sigma_map_i, number_map_i = compute_layer_sigma_map('i', sigma_ij_results, sigma_jk_results, sigma_ki_results)
    sigma_map_j, number_map_j = compute_layer_sigma_map('j', sigma_ij_results, sigma_jk_results, sigma_ki_results)
    sigma_map_k, number_map_k = compute_layer_sigma_map('k', sigma_ij_results, sigma_jk_results, sigma_ki_results)

    sigma_map_i_raw = sigma_map_j_raw = sigma_map_k_raw = None
    if HAS_RAW:
        sigma_map_i_raw, _ = compute_layer_sigma_map('i', sigma_ij_results_raw, sigma_jk_results_raw, sigma_ki_results_raw)
        sigma_map_j_raw, _ = compute_layer_sigma_map('j', sigma_ij_results_raw, sigma_jk_results_raw, sigma_ki_results_raw)
        sigma_map_k_raw, _ = compute_layer_sigma_map('k', sigma_ij_results_raw, sigma_jk_results_raw, sigma_ki_results_raw)

    mean_i, std_i = plot_resolution_plots(sigma_map_i, info_i, BASE_OUTPUT_DIR, mask=MASK_I, sigma_map_raw=sigma_map_i_raw)
    plot_sigma_vs_log(sigma_map_i, noise_i, baseline_i, BASE_OUTPUT_DIR, 'i')

    mean_j, std_j = plot_resolution_plots(sigma_map_j, info_j, BASE_OUTPUT_DIR, mask=MASK_J, sigma_map_raw=sigma_map_j_raw)
    plot_sigma_vs_log(sigma_map_j, noise_j, baseline_j, BASE_OUTPUT_DIR, 'j')

    mean_k, std_k = plot_resolution_plots(sigma_map_k, info_k, BASE_OUTPUT_DIR, mask=MASK_K, sigma_map_raw=sigma_map_k_raw)
    plot_sigma_vs_log(sigma_map_k, noise_k, baseline_k, BASE_OUTPUT_DIR, 'k')

    # ========================================================================
    # Save resolution summary JSON
    # ========================================================================

    config          = run_log['config']
    service_hybrids = config['telescope_config']['service_hybrids']
    rb_info         = {sh['rb']: sh for sh in service_hybrids}
    temperature     = config.get('temperature', config.get('run_config', {}).get('temperature', None))

    def _module_entry(rb, mean, std):
        sh = rb_info[rb]
        return {
            'name':          sh['modules'][0]['name'],
            'bias_voltage':  sh['bias_voltage'],
            'mean_sigma_ps': round(float(mean), 2) if not np.isnan(mean) else None,
            'std_sigma_ps':  round(float(std),  2) if not np.isnan(std)  else None,
        }

    summary = {
        'run':         RUN,
        'temperature': temperature,
        'modules': {
            'i': _module_entry(1, mean_i, std_i),
            'j': _module_entry(2, mean_j, std_j),
            'k': _module_entry(3, mean_k, std_k),
        }
    }

    summary_path = os.path.join(BASE_OUTPUT_DIR, 'resolution_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Resolution summary saved to: {summary_path}")
