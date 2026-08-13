"""
do_bootstrap_quality.py - Quality check for ETL telescope data.

Reads ROOT files batch by batch (same as do_bootstrap_preselection.py),
accumulating only hit maps and combination counters — never all events at once.

Usage:
    python do_bootstrap_quality.py <run_start> <run_stop> <trigger> [--data-path ...] [--output-dir ...]
"""

import sys
import os
import math
import argparse
import itertools
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm
from collections import defaultdict

hep.style.use("CMS")

BRANCHES           = ["event", "l1counter", "row", "col", "row_global", "col_global", "tot_code", "toa_code", "cal_code", "elink", "chipid", "bcid", "nhits"]
MIN_PIXEL_HITS     = 300
TOP_N_COMBINATIONS = 4

# Set from --doGlobalCoordinates in main(). When True, "row"/"col" are
# overwritten with "row_global"/"col_global" (range 0..31) right after each
# batch is read, so the rest of the pipeline is unaffected. N_PIX is the
# corresponding grid size used to size hit maps and loop ranges.
DO_GLOBAL_COORDINATES = False
N_PIX                 = 16


def _apply_coordinate_choice(batch):
    """Overwrite row/col with row_global/col_global when --doGlobalCoordinates
    is set. No-op otherwise, so default behavior is unchanged."""
    if not DO_GLOBAL_COORDINATES:
        return batch
    return ak.with_field(ak.with_field(batch, batch["row_global"], "row"),
                         batch["col_global"], "col")


# ============================================================================
# Plots
# ============================================================================

def plot_total_hits(hits_per_layer, output_path):
    colors = ['#3f90da', '#ffa90e', '#bd1f01']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([f"Layer {i}" for i in range(3)], hits_per_layer,
           color=colors, edgecolor='black')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total hits")
    ax.set_title("Total hits per layer (1-hit trigger selection)")
    for i, v in enumerate(hits_per_layer):
        ax.text(i, v * 1.01, f'{v:,}', ha='center', fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path + ".png", dpi=150)
    fig.savefig(output_path + ".pdf")
    plt.close(fig)
    print(f"Saved: {output_path}.png")


def plot_hit_map(name, hit_maps):
    maxcmap = max(np.max(h) for h in hit_maps)
    cmap    = plt.get_cmap('viridis')
    scale   = 4 if DO_GLOBAL_COORDINATES else 1
    fig, ax = plt.subplots(1, 3, figsize=(33 * scale, 11 * scale))
    for idx, hist in enumerate(hit_maps):
        ax[idx].set_title(f"Layer {idx}", fontsize=28 * scale)
        cax = ax[idx].matshow(hist, cmap=cmap, vmin=0, vmax=maxcmap)
        for i in range(N_PIX):
            for j in range(N_PIX):
                if hist[i, j] > 0:
                    ax[idx].text(j, i, str(int(hist[i, j])),
                                 ha='center', va='center',
                                 color='w', fontsize=6 * scale, fontweight='bold')
        ax[idx].set_ylabel(r'$Row$', fontsize=24 * scale)
        ax[idx].set_xlabel(r'$Column$', fontsize=24 * scale)
        ax[idx].tick_params(labelsize=20 * scale)
        if DO_GLOBAL_COORDINATES:
            ax[idx].axhline(15.5, color='red', linewidth=4)
            ax[idx].axvline(15.5, color='red', linewidth=4)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=20 * scale)
    fig.savefig(f"{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}.png")


def plot_hit_map_per_chip(name, hit_maps, rb_map=None):
    """One figure per layer (rb), with the 4 ETROC quadrants (0:mid, mid:N_PIX)
    each drawn in its own subplot with its own colorbar — so a quiet chip's
    scale isn't tied to a hot chip on the same module. Only meaningful when
    N_PIX == 32 (--doGlobalCoordinates)."""
    cmap  = plt.get_cmap('viridis')
    mid   = N_PIX // 2
    scale = 4 if DO_GLOBAL_COORDINATES else 1
    quadrants = [(slice(0, mid), slice(0, mid)),
                 (slice(0, mid), slice(mid, N_PIX)),
                 (slice(mid, N_PIX), slice(0, mid)),
                 (slice(mid, N_PIX), slice(mid, N_PIX))]

    for idx, hist in enumerate(hit_maps):
        rb = rb_map[idx] if rb_map is not None else idx
        fig, axes = plt.subplots(2, 2, figsize=(12 * scale, 11 * scale))
        fig.suptitle(f"Layer {idx} (rb{rb}) — per-ETROC scale", fontsize=14 * scale)
        for q, ((rs, cs), ax) in enumerate(zip(quadrants, axes.flat)):
            sub  = hist[rs, cs]
            vmax = max(np.max(sub), 1)
            im = ax.imshow(sub, cmap=cmap, vmin=0, vmax=vmax, origin='upper',
                            extent=(cs.start - 0.5, cs.stop - 0.5,
                                    rs.stop - 0.5, rs.start - 0.5))
            for i in range(rs.start, rs.stop):
                for j in range(cs.start, cs.stop):
                    if hist[i, j] > 0:
                        ax.text(j, i, str(int(hist[i, j])),
                                ha='center', va='center',
                                color='w', fontsize=6 * scale, fontweight='bold')
            ax.set_title(f"Chip {q}", fontsize=12 * scale)
            ax.set_xlim(cs.start - 0.5, cs.stop - 0.5)
            ax.set_ylim(rs.stop - 0.5, rs.start - 0.5)
            ax.set_aspect('equal')
            ax.set_ylabel(r'$Row$', fontsize=10 * scale)
            ax.set_xlabel(r'$Column$', fontsize=10 * scale)
            ax.tick_params(labelsize=8 * scale)
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8 * scale)
        fig.tight_layout()
        fig.savefig(f"{name}_rb{rb}.pdf")
        fig.savefig(f"{name}_rb{rb}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {name}_rb{rb}.png")


def _draw_quiver_map(ax, trigger_pixels, vectors_per_pixel, hit_counts,
                     layer_trigger, title, colors):
    """Draw a 16x16 hit map with direction arrows. vectors_per_pixel is a list
    of lists — one list of (dr,dc) per trigger pixel."""
    hit_map = np.zeros((N_PIX, N_PIX))
    for (r, c), count in zip(trigger_pixels, hit_counts):
        hit_map[r, c] = count

    ax.matshow(hit_map, cmap='Blues', alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    for x in np.arange(-0.5, N_PIX, 1):
        ax.axhline(x, color='gray', lw=0.3, alpha=0.5)
        ax.axvline(x, color='gray', lw=0.3, alpha=0.5)

    angles = []
    for (r, c), vecs in zip(trigger_pixels, vectors_per_pixel):
        for rank, (dr, dc) in enumerate(vecs):
            if dr == 0 and dc == 0:
                continue
            scale = 0.45 / max(abs(dr), abs(dc), 1e-9)
            color = colors[rank % len(colors)]
            ax.annotate("", xy=(c + dc * scale, r + dr * scale), xytext=(c, r),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            if rank == 0:
                angles.append(np.degrees(np.arctan2(dr, dc)))
    return angles


def plot_direction_vectors(trigger_pixels, dominant_vectors, all_vectors_list,
                           all_counts_list, hit_counts, layer_trigger, output_path):
    top3_colors = ['red', 'blue', 'green']

    # --- Plot 1: dominant direction map ---
    fig, ax = plt.subplots(figsize=(12, 12))
    angles = _draw_quiver_map(
        ax, trigger_pixels,
        [[v] for v in dominant_vectors],
        hit_counts, layer_trigger,
        f"Dominant beam direction — trigger layer {layer_trigger}",
        colors=['red']
    )
    fig.tight_layout()
    fig.savefig(output_path + ".png", dpi=150)
    fig.savefig(output_path + ".pdf")
    plt.close(fig)
    print(f"Saved: {output_path}.png")

    # --- Plot 1b: angle histogram ---
    fig, ax = plt.subplots(figsize=(8, 6))
    if angles:
        ax.hist(angles, bins=36, range=(-180, 180), color='#3f90da', edgecolor='black')
        ax.axvline(np.median(angles), color='red', linestyle='--',
                   label=f'Median: {np.median(angles):.1f}°   σ={np.std(angles):.1f}°')
        ax.legend()
    ax.set_xlabel("Dominant direction angle (degrees)")
    ax.set_ylabel("Number of trigger pixels")
    ax.set_title("Consistency of dominant beam direction")
    fig.tight_layout()
    fig.savefig(output_path + "_angles.png", dpi=150)
    fig.savefig(output_path + "_angles.pdf")
    plt.close(fig)
    print(f"Saved: {output_path}_angles.png")

    # --- Plot 2: rate-weighted average of top-3 combinations ---
    weighted_vectors = []
    for vecs, counts in zip(all_vectors_list, all_counts_list):
        vecs3   = vecs[:3]
        counts3 = np.array(counts[:3], dtype=float)
        if counts3.sum() == 0 or len(vecs3) == 0:
            weighted_vectors.append((0.0, 0.0))
            continue
        w       = counts3 / counts3.sum()
        dr_w    = sum(w[i] * vecs3[i][0] for i in range(len(vecs3)))
        dc_w    = sum(w[i] * vecs3[i][1] for i in range(len(vecs3)))
        weighted_vectors.append((dr_w, dc_w))

    # --- Plot 2: rate-weighted direction map ---
    fig2, ax2 = plt.subplots(figsize=(12, 12))
    angles_w = _draw_quiver_map(
        ax2, trigger_pixels,
        [[v] for v in weighted_vectors],
        hit_counts, layer_trigger,
        f"Rate-weighted direction (top-3) — trigger layer {layer_trigger}",
        colors=['red']
    )
    fig2.tight_layout()
    fig2.savefig(output_path + "_top3weighted.png", dpi=150)
    fig2.savefig(output_path + "_top3weighted.pdf")
    plt.close(fig2)
    print(f"Saved: {output_path}_top3weighted.png")

    # --- Plot 2b: angle histogram ---
    fig2b, ax2b = plt.subplots(figsize=(8, 6))
    if angles_w:
        ax2b.hist(angles_w, bins=36, range=(-180, 180), color='#3f90da', edgecolor='black')
        ax2b.axvline(np.median(angles_w), color='red', linestyle='--',
                     label=f'Median: {np.median(angles_w):.1f}°   σ={np.std(angles_w):.1f}°')
        ax2b.legend()
    ax2b.set_xlabel("Weighted direction angle (degrees)")
    ax2b.set_ylabel("Number of trigger pixels")
    ax2b.set_title("Consistency of rate-weighted direction")
    fig2b.tight_layout()
    fig2b.savefig(output_path + "_top3weighted_angles.png", dpi=150)
    fig2b.savefig(output_path + "_top3weighted_angles.pdf")
    plt.close(fig2b)
    print(f"Saved: {output_path}_top3weighted_angles.png")

    return angles


def plot_per_pixel_combinations(trigger_pixels, all_combinations, all_counts,
                                all_vectors, layer_j, layer_k, output_dir):
    """One PNG per trigger pixel showing its top-N combinations as arrows."""
    if len(trigger_pixels) == 0:
        return

    colors = ['red', 'blue', 'green', 'orange']
    os.makedirs(output_dir, exist_ok=True)

    for (r, c), combs, counts, vecs in zip(
            trigger_pixels, all_combinations, all_counts, all_vectors):

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-N_PIX / 2, N_PIX / 2)
        ax.set_ylim(-N_PIX / 2, N_PIX / 2)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel(r"$\Delta$col")
        ax.set_ylabel(r"$\Delta$row")
        ax.set_title(f"Trigger pixel ({r},{c}) — top-{TOP_N_COMBINATIONS} combinations\n"
                     f"layers j={layer_j}, k={layer_k}")

        for rank, (comb, count, (dr, dc)) in enumerate(zip(combs, counts, vecs)):
            rj, cj, rk, ck = comb
            color = colors[rank % len(colors)]
            ax.annotate("", xy=(dc, dr), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax.scatter([cj - c], [rj - r], marker='o', color=color, s=50, zorder=5)
            ax.scatter([ck - c], [rk - r], marker='s', color=color, s=50, zorder=5)
            ax.text(dc + 0.2, dr + 0.2,
                    f"#{rank+1}  n={count}\nj=({rj},{cj})  k=({rk},{ck})",
                    fontsize=7, color=color)

        fig.tight_layout()
        out = os.path.join(output_dir, f"pixel_r{r:02d}_c{c:02d}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)

    print(f"Saved {len(trigger_pixels)} per-pixel plots to: {output_dir}/")


def save_combination_table(trigger_pixels, all_combinations, all_counts,
                           all_vectors, layer_trigger, layer_j, layer_k, output_path):
    with open(output_path + ".txt", 'w') as f:
        f.write(f"Trigger layer: {layer_trigger} | j={layer_j} | k={layer_k}\n")
        f.write("=" * 95 + "\n")
        f.write(f"{'Trigger pixel':>16} | {'Rank':>4} | "
                f"{'(row_j,col_j)':>14} | {'(row_k,col_k)':>14} | "
                f"{'Rate':>6} | {'(dr,dc)':>12} | {'angle':>8}\n")
        f.write("-" * 95 + "\n")
        for (r, c), combs, counts, vecs in zip(
                trigger_pixels, all_combinations, all_counts, all_vectors):
            for rank, (comb, count, (dr, dc)) in enumerate(zip(combs, counts, vecs)):
                rj, cj, rk, ck = comb
                angle = np.degrees(np.arctan2(dr, dc))
                f.write(f"  ({r:2d},{c:2d})          | {rank+1:>4} | "
                        f"  ({rj:2d},{cj:2d})        | "
                        f"  ({rk:2d},{ck:2d})        | "
                        f"{count:>6} | ({dr:+5.1f},{dc:+5.1f}) | {angle:>7.1f}°\n")
            f.write("\n")
    print(f"Saved: {output_path}.txt")


def plot_toa_comparison(toa_trg_for_j, toa_trg_for_k, toa_j, toa_k,
                        layer_trigger, layer_j, layer_k, label, output_path):
    """Compare TOA of trigger pixel hit with all hits in other layers.
    toa_trg_for_j/k are the trigger TOA broadcast to match every hit in j/k.
    label is a free-form string identifying the selection (e.g. a pixel or
    'all pixels') used in plot titles."""
    toa_trg_for_j = np.asarray(toa_trg_for_j)
    toa_trg_for_k = np.asarray(toa_trg_for_k)
    toa_j = np.asarray(toa_j)
    toa_k = np.asarray(toa_k)

    # Drop entries with nan (from cal_code == 0)
    valid_j = np.isfinite(toa_trg_for_j) & np.isfinite(toa_j)
    valid_k = np.isfinite(toa_trg_for_k) & np.isfinite(toa_k)
    toa_trg_for_j, toa_j = toa_trg_for_j[valid_j], toa_j[valid_j]
    toa_trg_for_k, toa_k = toa_trg_for_k[valid_k], toa_k[valid_k]

    dt_j  = toa_j - toa_trg_for_j
    dt_k  = toa_k - toa_trg_for_k

    # --- Absolute TOA distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, toa, layer, color in zip(
            axes,
            [toa_trg_for_j, toa_j, toa_k],
            [layer_trigger, layer_j, layer_k],
            ['#bd1f01', '#3f90da', '#ffa90e']):
        ax.hist(toa, bins=100, color=color, edgecolor='black', linewidth=0.5)
        ax.set_xlabel("TOA (ns)")
        ax.set_ylabel("Entries")
        ax.set_title(f"Layer {layer} TOA  ({label})")
    fig.tight_layout()
    fig.savefig(output_path + "_toa_distributions.png", dpi=150)
    fig.savefig(output_path + "_toa_distributions.pdf")
    plt.close(fig)
    print(f"Saved: {output_path}_toa_distributions.png")

    # --- Delta TOA vs trigger ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 12))
    for ax, dt, layer, color in zip(axes, [dt_j, dt_k], [layer_j, layer_k],
                                    ['#3f90da', '#ffa90e']):
        mean, std = np.mean(dt), np.std(dt)
        counts_h, edges_h = np.histogram(dt, bins=500)
        mode_dt = (edges_h[np.argmax(counts_h)] + edges_h[np.argmax(counts_h) + 1]) / 2.0
        ax.hist(dt, bins=100, range=(mean - 4*std, mean + 4*std),
                color=color, edgecolor='black', linewidth=0.5)
        ax.axvline(mode_dt, color='red', linestyle='--',
                   label=f'Mode: {mode_dt:.1f} ns')
        ax.set_xlabel(rf"$TOA_{{layer{layer}}} - TOA_{{layer{layer_trigger}}}$ (ns)")
        ax.set_ylabel("Entries")
        ax.set_title(f"ΔTOA: layer {layer} − trigger layer {layer_trigger}")
        ax.set_xlim(-6, 6)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path + "_delta_toa.png", dpi=150)
    fig.savefig(output_path + "_delta_toa.pdf")
    plt.close(fig)
    print(f"Saved: {output_path}_delta_toa.png")

    # --- 2D maps: TOA_other vs TOA_trigger, reference from mode of delta TOA ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 9))
    for ax, toa_trg, toa_other, dt, layer in zip(
            axes,
            [toa_trg_for_j, toa_trg_for_k],
            [toa_j, toa_k],
            [dt_j, dt_k],
            [layer_j, layer_k]):
        ax.hist2d(toa_trg, toa_other, bins=100, cmap='viridis')
        # Mode of delta TOA: bin with highest count
        counts, edges = np.histogram(dt, bins=500)
        mode_dt = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2.0
        x_line  = np.linspace(toa_trg.min(), toa_trg.max(), 100)
        ax.plot(x_line, x_line + mode_dt,      color='red', lw=1.5,
                label=rf"$TOA_{{{layer}}} = TOA_{{{layer_trigger}}} + {mode_dt:.0f}$")
        ax.plot(x_line, x_line + mode_dt - 50, color='red', lw=1, linestyle='--')
        ax.plot(x_line, x_line + mode_dt + 50, color='red', lw=1, linestyle='--')
        ax.set_xlabel(rf"$TOA_{{layer\ {layer_trigger}}}$ (ns)")
        ax.set_ylabel(rf"$TOA_{{layer\ {layer}}}$ (ns)")
        ax.set_title(f"TOA layer {layer} vs trigger layer {layer_trigger} "
                     f"— {label}")
        ax.legend()
        ax.set_xlim(0, 12.5)
        ax.set_ylim(0, 12.5)
    fig.tight_layout()
    fig.savefig(output_path + "_toa_2d.png", dpi=150)
    fig.savefig(output_path + "_toa_2d.pdf")
    plt.close(fig)
    print(f"Saved: {output_path}_toa_2d.png")

def plot_tot_comparison(tot_trg_for_j, tot_trg_for_k, tot_j, tot_k,
                        layer_trigger, layer_j, layer_k, label, output_path):
    """Compare TOA of trigger pixel hit with all hits in other layers.
    toa_trg_for_j/k are the trigger TOA broadcast to match every hit in j/k.
    label is a free-form string identifying the selection (e.g. a pixel or
    'all pixels') used in plot titles."""
    tot_trg_for_j = np.asarray(tot_trg_for_j)
    tot_trg_for_k = np.asarray(tot_trg_for_k)
    tot_j = np.asarray(tot_j)
    tot_k = np.asarray(tot_k)

    # Drop entries with nan (from cal_code == 0)
    valid_j = np.isfinite(tot_trg_for_j) & np.isfinite(tot_j)
    valid_k = np.isfinite(tot_trg_for_k) & np.isfinite(tot_k)
    tot_trg_for_j, tot_j = tot_trg_for_j[valid_j], tot_j[valid_j]
    tot_trg_for_k, tot_k = tot_trg_for_k[valid_k], tot_k[valid_k]

    dtot_j  = tot_j - tot_trg_for_j
    dtot_k  = tot_k - tot_trg_for_k

    # --- Absolute TOA distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, toa, layer, color in zip(
            axes,
            [tot_trg_for_j, tot_j, tot_k],
            [layer_trigger, layer_j, layer_k],
            ['#bd1f01', '#3f90da', '#ffa90e']):
        ax.hist(toa, bins=100, color=color, edgecolor='black', linewidth=0.5)
        ax.set_xlabel("TOT (ns)")
        ax.set_ylabel("Entries")
        ax.set_title(f"Layer {layer} TOT  ({label})")
    fig.tight_layout()
    fig.savefig(output_path + "_tot_distributions.png", dpi=150)
    fig.savefig(output_path + "_tot_distributions.pdf")
    plt.close(fig)
    print(f"Saved: {output_path}_tot_distributions.png")

    # --- Delta TOT vs trigger ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 12))
    for ax, dt, layer, color in zip(axes, [dtot_j, dtot_k], [layer_j, layer_k],
                                    ['#3f90da', '#ffa90e']):
        mean, std = np.mean(dt), np.std(dt)
        counts_h, edges_h = np.histogram(dt, bins=500)
        mode_dtot = (edges_h[np.argmax(counts_h)] + edges_h[np.argmax(counts_h) + 1]) / 2.0
        ax.hist(dt, bins=100, range=(mean - 4*std, mean + 4*std),
                color=color, edgecolor='black', linewidth=0.5)
        ax.axvline(mode_dtot, color='red', linestyle='--',
                   label=f'Mode: {mode_dtot:.1f} ns')
        ax.set_xlabel(rf"$TOT_{{layer{layer}}} - TOT_{{layer{layer_trigger}}}$ (ns)")
        ax.set_ylabel("Entries")
        ax.set_title(f"ΔTOT: layer {layer} − trigger layer {layer_trigger}")
        ax.set_xlim(-6, 6)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path + "_delta_tot.png", dpi=150)
    fig.savefig(output_path + "_delta_tot.pdf")
    plt.close(fig)
    print(f"Saved: {output_path}_delta_tot.png")

    # --- 2D maps: TOA_other vs TOA_trigger, reference from mode of delta TOA ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 9))
    for ax, tot_trg, tot_other, dtot, layer in zip(
            axes,
            [tot_trg_for_j, tot_trg_for_k],
            [tot_j, tot_k],
            [dtot_j, dtot_k],
            [layer_j, layer_k]):
        ax.hist2d(tot_trg, tot_other, bins=100, cmap='viridis')
        # Mode of delta TOA: bin with highest count
        counts, edges = np.histogram(dtot, bins=500)
        mode_dtot = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2.0
        x_line  = np.linspace(tot_trg.min(), tot_trg.max(), 100)
        ax.plot(x_line, x_line + mode_dtot,      color='red', lw=1.5,
                label=rf"$TOA_{{{layer}}} = TOA_{{{layer_trigger}}} + {mode_dtot:.0f}$")
        ax.plot(x_line, x_line + mode_dtot - 50, color='red', lw=1, linestyle='--')
        ax.plot(x_line, x_line + mode_dtot + 50, color='red', lw=1, linestyle='--')
        ax.set_xlabel(rf"$TOA_{{layer\ {layer_trigger}}}$ (ns)")
        ax.set_ylabel(rf"$TOA_{{layer\ {layer}}}$ (ns)")
        ax.set_title(f"TOA layer {layer} vs trigger layer {layer_trigger} "
                     f"— {label}")
        ax.legend()
        ax.set_xlim(0, 12.5)
        ax.set_ylim(0, 12.5)
    fig.tight_layout()
    fig.savefig(output_path + "_tot_2d.png", dpi=150)
    fig.savefig(output_path + "_tot_2d.pdf")
    plt.close(fig)
    print(f"Saved: {output_path}_tot_2d.png")


def plot_event_quantities_comparison(
        event_trg_j, event_j, l1cnt_trg_j, l1cnt_j, bcid_trg_j, bcid_j,
        event_trg_k, event_k, l1cnt_trg_k, l1cnt_k, bcid_trg_k, bcid_k,
        layer_trigger, layer_j, layer_k, label, output_path):
    """One figure per quantity: X = trigger value, Y = other layer hit value.
    label is a free-form string identifying the selection used in plot titles."""
    quantities = [
        ("event",     "Event number", event_trg_j, event_j,     event_trg_k, event_k),
        ("l1counter", "L1 counter",   l1cnt_trg_j, l1cnt_j,     l1cnt_trg_k, l1cnt_k),
        ("bcid",      "BCID",         bcid_trg_j,  bcid_j,      bcid_trg_k,  bcid_k),
    ]
    for tag, qlabel, xj, yj, xk, yk in quantities:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, xv, yv, layer in zip(axes, [xj, xk], [yj, yk], [layer_j, layer_k]):
            if len(xv) == 0:
                ax.set_title(f"No data — layer {layer}")
                continue
            ax.hist2d(xv.astype(float), yv.astype(float), bins=100, cmap='viridis')
            ax.set_xlabel(f"{qlabel} — trigger ({label})")
            ax.set_ylabel(f"{qlabel} — layer {layer} hits")
            ax.set_title(f"{qlabel}: trigger vs layer {layer}\n{label}")
        fig.tight_layout()
        out = f"{output_path}_{tag}"
        fig.savefig(out + ".png", dpi=150)
        fig.savefig(out + ".pdf")
        plt.close(fig)
        print(f"Saved: {out}.png")


# ============================================================================
# Global (all-pixel) coincidence analysis
# ============================================================================

def analyze_global_one_hit_per_layer(run_files, trigger, layer_j, layer_k,
                                     step_size, total_batches, output_dir, tol=5):
    """TOA/event comparison using ALL events with exactly 1 hit in every
    layer — no restriction to a single trigger pixel. Mirrors the single-pixel
    deep-dive (CAL_CODE mode-map pass, then TOA/event collection pass) but the
    coincidence requirement (1 hit per layer) replaces the fixed-pixel mask."""
    print(f"\n{'='*50}")
    print("Global analysis: events with exactly 1 hit in every layer")

    out_dir = os.path.join(output_dir, "global_1hit_per_layer")
    os.makedirs(out_dir, exist_ok=True)

    def mask_one_hit_each(b):
        mask = ak.num(b[0].row) == 1
        for i in (1, 2):
            mask = mask & (ak.num(b[i].row) == 1)
        return mask

    # --- Pass A: CAL_CODE mode maps per (layer, row, col) ---
    print("\n  Pass A: computing CAL_CODE mode maps...")
    cal_hists = [np.zeros((N_PIX, N_PIX, 256), dtype=np.int64) for _ in range(3)]

    itersA = [uproot.iterate([f"{run_files[i]}:events"], BRANCHES, step_size=step_size)
              for i in range(3)]
    for batch0, batch1, batch2 in tqdm(itertools.islice(zip(*itersA), total_batches), desc="  CAL pass", total=total_batches):
        b = [_apply_coordinate_choice(x) for x in (batch0, batch1, batch2)]
        b = [x[mask_one_hit_each(b)] for x in b]
        if len(b[0]) == 0:
            continue
        for i in range(3):
            rows = ak.to_numpy(ak.firsts(b[i].row)).astype(int)
            cols = ak.to_numpy(ak.firsts(b[i].col)).astype(int)
            cals = np.clip(ak.to_numpy(ak.firsts(b[i].cal_code)).astype(int), 0, 255)
            np.add.at(cal_hists[i], (rows, cols, cals), 1)

    cal_mode_maps = [np.argmax(cal_hists[i], axis=2) for i in range(3)]
    print("  CAL_CODE mode maps computed.")

    # --- Pass B: apply the CAL cut, collect TOA / event / l1counter / bcid ---
    print("\n  Pass B: reading TOA and event-level quantities...")

    def to_toa_ns(toa_code, cal_code):
        cal_safe = np.where(cal_code > 0, cal_code, 1)
        toa = 12.5 - 3.125 / cal_safe * toa_code
        return np.where(cal_code > 0, toa, np.nan)
    
    def to_tot_ns(tot_code, cal_code):
        cal_safe = ak.where(cal_code > 0, cal_code, 1)
        tot = ((2*tot_code - np.floor(tot_code/32))*3.125 / cal_code)
        return ak.where(cal_code > 0, tot, float('nan'))


    toa_trg_list, toa_j_list, toa_k_list = [], [], []
    tot_trg_list, tot_j_list, tot_k_list = [], [], []
    event_trg_list, event_j_list, event_k_list = [], [], []
    l1cnt_trg_list, l1cnt_j_list, l1cnt_k_list = [], [], []
    bcid_trg_list,  bcid_j_list,  bcid_k_list  = [], [], []
    hmap_raw = [np.zeros((N_PIX, N_PIX), dtype=np.int64) for _ in range(3)]
    hmap_cut = [np.zeros((N_PIX, N_PIX), dtype=np.int64) for _ in range(3)]

    itersB = [uproot.iterate([f"{run_files[i]}:events"], BRANCHES, step_size=step_size)
              for i in range(3)]
    for batch0, batch1, batch2 in tqdm(itertools.islice(zip(*itersB), total_batches), desc="  TOA pass", total=total_batches):
        b = [_apply_coordinate_choice(x) for x in (batch0, batch1, batch2)]
        b = [x[mask_one_hit_each(b)] for x in b]
        if len(b[0]) == 0:
            continue

        rows  = [ak.to_numpy(ak.firsts(b[i].row)).astype(int)        for i in range(3)]
        cols  = [ak.to_numpy(ak.firsts(b[i].col)).astype(int)        for i in range(3)]
        cals  = [ak.to_numpy(ak.firsts(b[i].cal_code)).astype(int)   for i in range(3)]
        toas  = [ak.to_numpy(ak.firsts(b[i].toa_code)).astype(float) for i in range(3)]
        tots  = [ak.to_numpy(ak.firsts(b[i].tot_code)).astype(float) for i in range(3)]
        event = [ak.to_numpy(b[i]["event"]).astype(np.int64)         for i in range(3)]
        l1cnt = [ak.to_numpy(b[i]["l1counter"]).astype(np.int64)     for i in range(3)]
        bcid  = [ak.to_numpy(b[i]["bcid"]).astype(np.int64)          for i in range(3)]

        for i in range(3):
            np.add.at(hmap_raw[i], (rows[i], cols[i]), 1)

        good = np.ones(len(rows[0]), dtype=bool)
        for i in range(3):
            good &= np.abs(cals[i] - cal_mode_maps[i][rows[i], cols[i]]) < tol
        if not good.any():
            continue

        for i in range(3):
            np.add.at(hmap_cut[i], (rows[i][good], cols[i][good]), 1)

        toa_trg_list.append(to_toa_ns(toas[trigger][good], cals[trigger][good]))
        toa_j_list.append(to_toa_ns(toas[layer_j][good],   cals[layer_j][good]))
        toa_k_list.append(to_toa_ns(toas[layer_k][good],   cals[layer_k][good]))
        tot_trg_list.append(to_tot_ns(tots[trigger][good], cals[trigger][good]))
        tot_j_list.append(to_tot_ns(tots[layer_j][good],   cals[layer_j][good]))
        tot_k_list.append(to_tot_ns(tots[layer_k][good],   cals[layer_k][good]))
        event_trg_list.append(event[trigger][good])
        event_j_list.append(event[layer_j][good])
        event_k_list.append(event[layer_k][good])
        l1cnt_trg_list.append(l1cnt[trigger][good])
        l1cnt_j_list.append(l1cnt[layer_j][good])
        l1cnt_k_list.append(l1cnt[layer_k][good])
        bcid_trg_list.append(bcid[trigger][good])
        bcid_j_list.append(bcid[layer_j][good])
        bcid_k_list.append(bcid[layer_k][good])

    plot_hit_map(os.path.join(out_dir, "hitmap_raw"), hmap_raw)
    plot_hit_map(os.path.join(out_dir, "hitmap_calcut"), hmap_cut)

    if not toa_trg_list:
        print("  No events with 1 hit in every layer passed the CAL_CODE cut.")
        return

    toa_trg = np.concatenate(toa_trg_list)
    tot_trg = np.concatenate(tot_trg_list)
    print(f"  Events after cut: {len(toa_trg):,}")

    plot_toa_comparison(
        toa_trg, toa_trg, np.concatenate(toa_j_list), np.concatenate(toa_k_list),
        layer_trigger=trigger, layer_j=layer_j, layer_k=layer_k,
        label="all pixels — 1 hit/layer",
        output_path=os.path.join(out_dir, "toa")
    )

    plot_tot_comparison(
        tot_trg, tot_trg, np.concatenate(tot_j_list), np.concatenate(tot_k_list),
        layer_trigger=trigger, layer_j=layer_j, layer_k=layer_k,
        label="all pixels — 1 hit/layer",
        output_path=os.path.join(out_dir, "tot")
    )


    event_trg = np.concatenate(event_trg_list)
    l1cnt_trg = np.concatenate(l1cnt_trg_list)
    bcid_trg  = np.concatenate(bcid_trg_list)
    plot_event_quantities_comparison(
        event_trg_j=event_trg, event_j=np.concatenate(event_j_list),
        l1cnt_trg_j=l1cnt_trg, l1cnt_j=np.concatenate(l1cnt_j_list),
        bcid_trg_j=bcid_trg,   bcid_j=np.concatenate(bcid_j_list),
        event_trg_k=event_trg, event_k=np.concatenate(event_k_list),
        l1cnt_trg_k=l1cnt_trg, l1cnt_k=np.concatenate(l1cnt_k_list),
        bcid_trg_k=bcid_trg,   bcid_k=np.concatenate(bcid_k_list),
        layer_trigger=trigger, layer_j=layer_j, layer_k=layer_k,
        label="all pixels — 1 hit/layer",
        output_path=os.path.join(out_dir, "event_quantities")
    )

    print(f"  All outputs saved to: {out_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_start', type=int)
    parser.add_argument('run_stop',  type=int)
    parser.add_argument('trigger',   type=int, choices=[0, 1, 2])
    parser.add_argument('--data-path', type=str,
                        default="/eos/user/f/fernance/ETL/Bootstrap/July2026/Data/")
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--min-hits',   type=int, default=MIN_PIXEL_HITS)
    parser.add_argument('--top-n',      type=int, default=TOP_N_COMBINATIONS)
    parser.add_argument('--pixel-row',  type=int, default=8)
    parser.add_argument('--pixel-col',  type=int, default=8)
    parser.add_argument('--skip-pixel-analysis', action='store_true',
                        help="Skip the single-pixel deep-dive (--pixel-row/"
                             "--pixel-col), which runs two extra full passes "
                             "over the ROOT files.")
    parser.add_argument('--doGlobalCoordinates', action='store_true',
                        help="Use row_global/col_global (range 0..31) instead "
                             "of row/col (range 0..16) for the saved row/col "
                             "values.")
    parser.add_argument('--max-batches', type=int, default=0,
                        help="Process at most this many batches per pass "
                             "(0 = no limit). Useful for quick tests.")
    args = parser.parse_args()

    global DO_GLOBAL_COORDINATES, N_PIX
    DO_GLOBAL_COORDINATES = args.doGlobalCoordinates
    N_PIX                 = 32 if DO_GLOBAL_COORDINATES else 16

    trigger    = args.trigger
    all_layers = [0, 1, 2]
    layer_j, layer_k = [l for l in all_layers if l != trigger]

    output_dir = args.output_dir or f"quality_{args.run_start}_{args.run_stop}_trg{trigger}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    print(f"Trigger: {trigger}  j={layer_j}  k={layer_k}\n")

    # -------------------------------------------------------------------------
    # Locate ROOT files (same naming convention as preselection)
    # -------------------------------------------------------------------------
    rb_map = {0: 1, 1: 2, 2: 3}
    run_files = {}
    for layer in all_layers:
        path = os.path.join(
            args.data_path,
            f"final_run_{args.run_start}_{args.run_stop}_rb{rb_map[layer]}.root"
        )
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)
        run_files[layer] = path
        print(f"Layer {layer}: {path}")

    step_size = 500000
    n0 = uproot.open(f"{run_files[0]}:events").num_entries
    n1 = uproot.open(f"{run_files[1]}:events").num_entries
    n2 = uproot.open(f"{run_files[2]}:events").num_entries
    print(f"File sizes: layer0={n0}, layer1={n1}, layer2={n2}")
    if not (n0 == n1 == n2):
        print(f"WARNING: files have different number of events ({n0}, {n1}, {n2}).")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [n0, n1, n2]
    ax.bar([f"Layer {i}" for i in range(3)], counts, color='#3f90da', edgecolor='black')
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("Events")
    ax.set_title(f"Events per module — runs {args.run_start}–{args.run_stop}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "events_per_module.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, "events_per_module.pdf"))
    plt.close(fig)

    total_events  = n0
    total_batches = math.ceil(total_events / step_size)
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)
        print(f"\nTotal events: {total_events}  |  Batches: {total_batches} "
              f"(limited by --max-batches={args.max_batches})\n")
    else:
        print(f"\nTotal events: {total_events}  |  Batches: {total_batches}\n")

    # -------------------------------------------------------------------------
    # Streaming pass — same structure as preselection, no full data in memory.
    # We only accumulate three 16x16 arrays and a combination counter dict.
    # -------------------------------------------------------------------------
    hit_maps_raw = [np.zeros((N_PIX, N_PIX), dtype=np.int64) for _ in all_layers]
    hit_maps_sel = [np.zeros((N_PIX, N_PIX), dtype=np.int64) for _ in all_layers]
    total_hits_raw = [0, 0, 0]
    total_hits_sel = [0, 0, 0]
    # combination_counts[(r_trg, c_trg)][(rj, cj, rk, ck)] -> int
    combination_counts = defaultdict(lambda: defaultdict(int))

    iter0 = uproot.iterate([f"{run_files[0]}:events"], BRANCHES, step_size=step_size)
    iter1 = uproot.iterate([f"{run_files[1]}:events"], BRANCHES, step_size=step_size)
    iter2 = uproot.iterate([f"{run_files[2]}:events"], BRANCHES, step_size=step_size)

    for batch0, batch1, batch2 in tqdm(itertools.islice(zip(iter0, iter1, iter2), total_batches),
                                       desc="Processing batches",
                                       total=total_batches):
        b_raw = [_apply_coordinate_choice(x) for x in (batch0, batch1, batch2)]

        # Accumulate raw hit maps (before any selection)
        for i in all_layers:
            rows = ak.to_numpy(ak.flatten(b_raw[i].row, axis=None)).astype(int)
            cols = ak.to_numpy(ak.flatten(b_raw[i].col, axis=None)).astype(int)
            np.add.at(hit_maps_raw[i], (rows, cols), 1)
            total_hits_raw[i] += len(rows)

        # 1-hit mask on trigger layer
        mask_1hit = ak.num(b_raw[trigger].row) == 1
        b = [x[mask_1hit] for x in b_raw]
        if len(b[0]) == 0:
            continue

        # Accumulate hit maps after selection
        for i in all_layers:
            rows = ak.to_numpy(ak.flatten(b[i].row, axis=None)).astype(int)
            cols = ak.to_numpy(ak.flatten(b[i].col, axis=None)).astype(int)
            np.add.at(hit_maps_sel[i], (rows, cols), 1)
            total_hits_sel[i] += len(rows)

        # Trigger pixel per event (exactly 1 hit after mask)
        rows_trg = ak.to_numpy(ak.flatten(b[trigger].row)).astype(int)
        cols_trg = ak.to_numpy(ak.flatten(b[trigger].col)).astype(int)

        # First hit in j and k per event
        rj_raw = ak.firsts(b[layer_j].row, axis=1)
        cj_raw = ak.firsts(b[layer_j].col, axis=1)
        rk_raw = ak.firsts(b[layer_k].row, axis=1)
        ck_raw = ak.firsts(b[layer_k].col, axis=1)

        has_jk   = ~(ak.is_none(rj_raw) | ak.is_none(cj_raw) |
                     ak.is_none(rk_raw) | ak.is_none(ck_raw))
        has_jk_np = ak.to_numpy(has_jk)

        rows_j = ak.to_numpy(rj_raw[has_jk]).astype(int)
        cols_j = ak.to_numpy(cj_raw[has_jk]).astype(int)
        rows_k = ak.to_numpy(rk_raw[has_jk]).astype(int)
        cols_k = ak.to_numpy(ck_raw[has_jk]).astype(int)

        # Count unique (trg, j, k) combinations with numpy — no Python event loop
        triplets           = np.stack([rows_trg[has_jk_np], cols_trg[has_jk_np],
                                       rows_j, cols_j, rows_k, cols_k], axis=1)
        unique_t, counts_t = np.unique(triplets, axis=0, return_counts=True)
        for row, cnt in zip(unique_t, counts_t):
            rt, ct, rj, cj, rk, ck = row
            combination_counts[(rt, ct)][(rj, cj, rk, ck)] += int(cnt)

    # -------------------------------------------------------------------------
    # Plots and tables (from accumulators only)
    # -------------------------------------------------------------------------
    print(f"\nTotal hits (raw): layer0={total_hits_raw[0]:,}  layer1={total_hits_raw[1]:,}  layer2={total_hits_raw[2]:,}")
    print(f"Total hits (1-hit trigger): layer0={total_hits_sel[0]:,}  layer1={total_hits_sel[1]:,}  layer2={total_hits_sel[2]:,}")

    plot_total_hits(total_hits_raw,
                    output_path=os.path.join(output_dir, "total_hits_per_layer_raw"))
    plot_hit_map(os.path.join(output_dir, "hitmap_all_layers_raw"), hit_maps_raw)
    if N_PIX == 32:
        plot_hit_map_per_chip(os.path.join(output_dir, "hitmap_all_layers_raw_perchip"),
                              hit_maps_raw, rb_map=rb_map)

    plot_total_hits(total_hits_sel,
                    output_path=os.path.join(output_dir, "total_hits_per_layer_1hit"))
    plot_hit_map(os.path.join(output_dir, "hitmap_all_layers_1hit"), hit_maps_sel)

    # Select trigger pixels above threshold (from 1-hit selection)
    high_rate_pixels, high_rate_counts = [], []
    for r in range(N_PIX):
        for c in range(N_PIX):
            n = int(hit_maps_sel[trigger][r, c])
            if n >= args.min_hits:
                high_rate_pixels.append((r, c))
                high_rate_counts.append(n)

    order            = np.argsort(high_rate_counts)[::-1]
    high_rate_pixels = [high_rate_pixels[i] for i in order]
    high_rate_counts = [high_rate_counts[i] for i in order]
    print(f"\nTrigger pixels with >= {args.min_hits} hits: {len(high_rate_pixels)}")

    dominant_vectors = []
    all_combinations = []
    all_counts_list  = []
    all_vectors_list = []

    for (r, c) in high_rate_pixels:
        comb_dict    = combination_counts.get((r, c), {})
        sorted_combs = sorted(comb_dict.items(), key=lambda x: x[1], reverse=True)
        top_combs    = [list(k) for k, _ in sorted_combs[:args.top_n]]
        top_counts   = [v        for _, v in sorted_combs[:args.top_n]]

        vecs = []
        for comb in top_combs:
            rj, cj, rk, ck = comb
            # Fit a line through all 3 layers ordered by layer index (0, 1, 2)
            layer_pos  = {trigger: (r, c), layer_j: (rj, cj), layer_k: (rk, ck)}
            ordered    = sorted(layer_pos.keys())          # always [0, 1, 2]
            xs         = np.array(ordered, dtype=float)
            rows_l     = np.array([layer_pos[l][0] for l in ordered], dtype=float)
            cols_l     = np.array([layer_pos[l][1] for l in ordered], dtype=float)
            dr = np.polyfit(xs, rows_l, 1)[0]
            dc = np.polyfit(xs, cols_l, 1)[0]
            vecs.append((dr, dc))

        all_combinations.append(top_combs)
        all_counts_list.append(top_counts)
        all_vectors_list.append(vecs)
        dominant_vectors.append(vecs[0] if vecs else (0.0, 0.0))

    angles = plot_direction_vectors(
        high_rate_pixels, dominant_vectors, all_vectors_list, all_counts_list,
        high_rate_counts, layer_trigger=trigger,
        output_path=os.path.join(output_dir, "beam_direction_vectors")
    )
    plot_per_pixel_combinations(
        high_rate_pixels, all_combinations, all_counts_list, all_vectors_list,
        layer_j=layer_j, layer_k=layer_k,
        output_dir=os.path.join(output_dir, "per_pixel_combinations")
    )
    save_combination_table(
        high_rate_pixels, all_combinations, all_counts_list, all_vectors_list,
        layer_trigger=trigger, layer_j=layer_j, layer_k=layer_k,
        output_path=os.path.join(output_dir, "combination_table")
    )

    print(f"\n{'='*50}")
    if angles:
        print(f"Median angle : {np.median(angles):.1f} deg")
        print(f"Std dev      : {np.std(angles):.1f} deg")
        print(f"Range        : [{np.min(angles):.1f}, {np.max(angles):.1f}] deg")
    print(f"All outputs saved to: {output_dir}/")

    # -------------------------------------------------------------------------
    # Single-pixel deep-dive: all combinations + hit maps for one trigger pixel
    # -------------------------------------------------------------------------
    if args.skip_pixel_analysis:
        print(f"\n{'='*50}")
        print("Skipping single-pixel deep-dive (--skip-pixel-analysis).")
    else:
      px_r, px_c = args.pixel_row, args.pixel_col
      print(f"\n{'='*50}")
      print(f"Single-pixel analysis: trigger pixel ({px_r}, {px_c})")

      px_comb_dict = combination_counts.get((px_r, px_c), {})
      if not px_comb_dict:
          print(f"  No combinations found for pixel ({px_r},{px_c})")
      else:
          px_dir = os.path.join(output_dir, f"pixel_{px_r:02d}_{px_c:02d}")
          os.makedirs(px_dir, exist_ok=True)

          sorted_combs = sorted(px_comb_dict.items(), key=lambda x: x[1], reverse=True)

          # --- txt with all combinations ---
          txt_path = os.path.join(px_dir, "all_combinations.txt")
          with open(txt_path, 'w') as f:
              f.write(f"All combinations for trigger pixel ({px_r},{px_c}) "
                      f"— layer {trigger}\n")
              f.write(f"Trigger layer: {trigger} | j={layer_j} | k={layer_k}\n")
              f.write(f"Total unique combinations: {len(sorted_combs)}\n")
              f.write("=" * 60 + "\n")
              f.write(f"{'Rank':>5} | {'(row_j,col_j)':>14} | "
                      f"{'(row_k,col_k)':>14} | {'Rate':>7}\n")
              f.write("-" * 60 + "\n")
              for rank, ((rj, cj, rk, ck), count) in enumerate(sorted_combs):
                  f.write(f"{rank+1:>5} | "
                          f"  ({rj:2d},{cj:2d})       | "
                          f"  ({rk:2d},{ck:2d})       | {count:>7}\n")
          print(f"  Saved: {txt_path}")

          # --- hit maps reconstructed from combination counts ---
          hmap_j = np.zeros((N_PIX, N_PIX), dtype=np.int64)
          hmap_k = np.zeros((N_PIX, N_PIX), dtype=np.int64)
          hmap_trg = np.zeros((N_PIX, N_PIX), dtype=np.int64)
          hmap_trg[px_r, px_c] = sum(c for _, c in sorted_combs)
          for (rj, cj, rk, ck), count in sorted_combs:
              hmap_j[rj, cj] += count
              hmap_k[rk, ck] += count

          # Hit maps are filled properly during the second pass (see below)
          # Placeholder until second pass completes — overwritten after
          hit_maps_px = [hmap_trg, hmap_j, hmap_k]  # temporary, replaced below
          print(f"  Total combinations: {len(sorted_combs)}")
          print(f"  Top combination: j=({sorted_combs[0][0][0]},{sorted_combs[0][0][1]}) "
                f"k=({sorted_combs[0][0][2]},{sorted_combs[0][0][3]}) "
                f"rate={sorted_combs[0][1]}")

          # --- Pass 2a: compute cal_code mode per (layer, row, col) ---
          # Only uses events where trigger layer has exactly 1 hit at (px_r, px_c)
          print(f"\n  Pass 2a: computing CAL_CODE mode maps for pixel ({px_r},{px_c})...")
          # Accumulate cal_code histograms in shape (N_PIX,N_PIX,256) per layer (256 bins for codes 0..255)
          cal_hists = [np.zeros((N_PIX, N_PIX, 256), dtype=np.int64) for _ in range(3)]

          iter0_2a = uproot.iterate([f"{run_files[0]}:events"], BRANCHES, step_size=step_size)
          iter1_2a = uproot.iterate([f"{run_files[1]}:events"], BRANCHES, step_size=step_size)
          iter2_2a = uproot.iterate([f"{run_files[2]}:events"], BRANCHES, step_size=step_size)

          for batch0, batch1, batch2 in tqdm(itertools.islice(zip(iter0_2a, iter1_2a, iter2_2a), total_batches),
                                             desc="  CAL pass", total=total_batches):
              b = [_apply_coordinate_choice(x) for x in (batch0, batch1, batch2)]

              mask_1hit = ak.num(b[trigger].row) == 1
              b = [x[mask_1hit] for x in b]
              if len(b[0]) == 0:
                  continue

              rows_trg = ak.to_numpy(ak.flatten(b[trigger].row)).astype(int)
              cols_trg = ak.to_numpy(ak.flatten(b[trigger].col)).astype(int)
              mask_px  = (rows_trg == px_r) & (cols_trg == px_c)
              b = [x[mask_px] for x in b]
              if len(b[0]) == 0:
                  continue

              for i in range(3):
                  rows = ak.to_numpy(ak.flatten(b[i].row, axis=None)).astype(int)
                  cols = ak.to_numpy(ak.flatten(b[i].col, axis=None)).astype(int)
                  cals = ak.to_numpy(ak.flatten(b[i].cal_code, axis=None)).astype(int)
                  cals_clipped = np.clip(cals, 0, 255)
                  np.add.at(cal_hists[i], (rows, cols, cals_clipped), 1)

          # Mode = argmax over the 256 cal bins for each (row, col)
          cal_mode_maps = [np.argmax(cal_hists[i], axis=2) for i in range(3)]
          print("  CAL_CODE mode maps computed.")

          # --- Pass 2b: collect TOA values applying |cal_code - mode| < 5 cut ---
          print(f"\n  Pass 2b: reading TOA values and hit maps for trigger pixel ({px_r},{px_c})...")
          toa_trg_for_j_list, toa_trg_for_k_list = [], []
          toa_j_list, toa_k_list = [], []
          toa_trg_no_j_list, toa_trg_no_k_list = [], []
          # event/l1counter/bcid: trigger side (broadcast to hit level) and other-layer side
          event_trg_for_j_list, event_j_list = [], []
          event_trg_for_k_list, event_k_list = [], []
          l1cnt_trg_for_j_list, l1cnt_j_list = [], []
          l1cnt_trg_for_k_list, l1cnt_k_list = [], []
          bcid_trg_for_j_list,  bcid_j_list  = [], []
          bcid_trg_for_k_list,  bcid_k_list  = [], []
          hmap_pass2      = [np.zeros((N_PIX, N_PIX), dtype=np.int64) for _ in range(3)]
          hmap_bcid_align = [np.zeros((N_PIX, N_PIX), dtype=np.int64) for _ in range(3)]

          iter0_2 = uproot.iterate([f"{run_files[0]}:events"], BRANCHES, step_size=step_size)
          iter1_2 = uproot.iterate([f"{run_files[1]}:events"], BRANCHES, step_size=step_size)
          iter2_2 = uproot.iterate([f"{run_files[2]}:events"], BRANCHES, step_size=step_size)

          def to_toa_ns(toa_code, cal_code):
              cal_safe = ak.where(cal_code > 0, cal_code, 1)
              toa = 12.5 - 3.125 / cal_safe * toa_code
              return ak.where(cal_code > 0, toa, float('nan'))

          def to_tot_ns(tot_code, cal_code):
              cal_safe = ak.where(cal_code > 0, cal_code, 1)
              tot = ((2*tot_code - np.floor(tot_code_/32))*3.125 / cal_code)
              return ak.where(cal_code > 0, tot, float('nan'))

          def apply_cal_cut(layer_batch, mode_map, tol=5):
              """Keep only hits where |cal_code - mode_map[row, col]| < tol.
              Only processes per-hit (jagged) branches: row, col, cal_code, toa_code, tot_code.
              Scalar-per-event branches (event, l1counter, bcid, elink, chipid, nhits)
              must be saved before calling this function and re-masked separately."""
              orig_counts = ak.to_numpy(ak.num(layer_batch.row)).astype(np.int64)
              rows_flat   = ak.to_numpy(ak.flatten(layer_batch.row)).astype(int)
              cols_flat   = ak.to_numpy(ak.flatten(layer_batch.col)).astype(int)
              cals_flat   = ak.to_numpy(ak.flatten(layer_batch.cal_code)).astype(int)
              toas_flat   = ak.to_numpy(ak.flatten(layer_batch.toa_code))
              tots_flat   = ak.to_numpy(ak.flatten(layer_batch.tot_code))
              modes_flat  = mode_map[rows_flat, cols_flat]
              good        = np.abs(cals_flat - modes_flat) < tol
              event_idx   = np.repeat(np.arange(len(orig_counts)), orig_counts)
              new_counts  = np.zeros(len(orig_counts), dtype=np.int64)
              if good.any():
                  new_counts = np.bincount(event_idx[good], minlength=len(orig_counts)).astype(np.int64)
              return ak.zip({
                  "row":      ak.unflatten(rows_flat[good], new_counts),
                  "col":      ak.unflatten(cols_flat[good], new_counts),
                  "cal_code": ak.unflatten(cals_flat[good], new_counts),
                  "toa_code": ak.unflatten(toas_flat[good], new_counts),
                  "tot_code": ak.unflatten(tots_flat[good], new_counts),
              })

          for batch0, batch1, batch2 in tqdm(itertools.islice(zip(iter0_2, iter1_2, iter2_2), total_batches),
                                             desc="  TOA pass", total=total_batches):
              b = [_apply_coordinate_choice(x) for x in (batch0, batch1, batch2)]

              # 1-hit in trigger layer and must be the target pixel
              mask_1hit = ak.num(b[trigger].row) == 1
              b = [x[mask_1hit] for x in b]
              if len(b[0]) == 0:
                  continue

              rows_trg = ak.to_numpy(ak.flatten(b[trigger].row)).astype(int)
              cols_trg = ak.to_numpy(ak.flatten(b[trigger].col)).astype(int)
              mask_px  = (rows_trg == px_r) & (cols_trg == px_c)
              b = [x[mask_px] for x in b]
              if len(b[0]) == 0:
                  continue

              # Save scalar-per-event quantities before cal cut (dropped by apply_cal_cut)
              event_pre    = np.asarray(b[trigger]["event"]).astype(np.int64)
              l1cnt_pre    = np.asarray(b[trigger]["l1counter"]).astype(np.int64)
              bcid_pre     = np.asarray(b[trigger]["bcid"]).astype(np.int64)
              event_pre_j  = np.asarray(b[layer_j]["event"]).astype(np.int64)
              l1cnt_pre_j  = np.asarray(b[layer_j]["l1counter"]).astype(np.int64)
              bcid_pre_j   = np.asarray(b[layer_j]["bcid"]).astype(np.int64)
              event_pre_k  = np.asarray(b[layer_k]["event"]).astype(np.int64)
              l1cnt_pre_k  = np.asarray(b[layer_k]["l1counter"]).astype(np.int64)
              bcid_pre_k   = np.asarray(b[layer_k]["bcid"]).astype(np.int64)

              # Apply CAL_CODE cut to all layers
              b = [apply_cal_cut(b[i], cal_mode_maps[i]) for i in range(3)]

              # After cal cut the trigger layer may have lost its 1 hit — re-apply guard
              mask_1hit_after    = ak.num(b[trigger].row) == 1
              mask_1hit_after_np = ak.to_numpy(mask_1hit_after)
              b = [x[mask_1hit_after] for x in b]
              if len(b[0]) == 0:
                  continue

              # Apply same event mask to all pre-saved scalar quantities
              event_ev   = event_pre[mask_1hit_after_np]
              l1cnt_ev   = l1cnt_pre[mask_1hit_after_np]
              bcid_ev    = bcid_pre[mask_1hit_after_np]
              event_ev_j = event_pre_j[mask_1hit_after_np]
              l1cnt_ev_j = l1cnt_pre_j[mask_1hit_after_np]
              bcid_ev_j  = bcid_pre_j[mask_1hit_after_np]
              event_ev_k = event_pre_k[mask_1hit_after_np]
              l1cnt_ev_k = l1cnt_pre_k[mask_1hit_after_np]
              bcid_ev_k  = bcid_pre_k[mask_1hit_after_np]

              # Accumulate hit maps with ALL hits per event (after CAL cut)
              for i in range(3):
                  rows = ak.to_numpy(ak.flatten(b[i].row, axis=None)).astype(int)
                  cols = ak.to_numpy(ak.flatten(b[i].col, axis=None)).astype(int)
                  np.add.at(hmap_pass2[i], (rows, cols), 1)

              # Hit map for bcid-aligned events: all three layers share the same bcid
              mask_bcid_align = (bcid_ev == bcid_ev_j) & (bcid_ev == bcid_ev_k)
              b_align = [x[mask_bcid_align] for x in b]
              for i in range(3):
                  rows = ak.to_numpy(ak.flatten(b_align[i].row, axis=None)).astype(int)
                  cols = ak.to_numpy(ak.flatten(b_align[i].col, axis=None)).astype(int)
                  np.add.at(hmap_bcid_align[i], (rows, cols), 1)

              # Trigger TOA in ns: exactly 1 hit per event
              trg_toa_code = ak.firsts(b[trigger].toa_code, axis=1)
              trg_cal_code = ak.firsts(b[trigger].cal_code, axis=1)
              toa_trg_per_event = to_toa_ns(trg_toa_code, trg_cal_code)

              # All hits in j and k converted to ns
              toa_j_hits = to_toa_ns(b[layer_j].toa_code, b[layer_j].cal_code)
              toa_k_hits = to_toa_ns(b[layer_k].toa_code, b[layer_k].cal_code)

              trg_j, toa_j_bc = ak.broadcast_arrays(toa_trg_per_event, toa_j_hits)
              trg_k, toa_k_bc = ak.broadcast_arrays(toa_trg_per_event, toa_k_hits)

              toa_trg_for_j_list.append(ak.to_numpy(ak.flatten(trg_j)).astype(float))
              toa_trg_for_k_list.append(ak.to_numpy(ak.flatten(trg_k)).astype(float))
              toa_j_list.append(ak.to_numpy(ak.flatten(toa_j_bc)).astype(float))
              toa_k_list.append(ak.to_numpy(ak.flatten(toa_k_bc)).astype(float))

              # Trigger TOA for events with no hit in layer_j or layer_k
              toa_trg_np = ak.to_numpy(toa_trg_per_event).astype(float)
              mask_no_j  = ak.to_numpy(ak.num(b[layer_j].row) == 0)
              mask_no_k  = ak.to_numpy(ak.num(b[layer_k].row) == 0)
              toa_trg_no_j_list.append(toa_trg_np[mask_no_j])
              toa_trg_no_k_list.append(toa_trg_np[mask_no_k])

              # All quantities are scalar per event — accumulate directly (no broadcast)
              event_trg_for_j_list.append(event_ev)
              event_trg_for_k_list.append(event_ev)
              l1cnt_trg_for_j_list.append(l1cnt_ev)
              l1cnt_trg_for_k_list.append(l1cnt_ev)
              bcid_trg_for_j_list.append(bcid_ev)
              bcid_trg_for_k_list.append(bcid_ev)
              event_j_list.append(event_ev_j)
              event_k_list.append(event_ev_k)
              l1cnt_j_list.append(l1cnt_ev_j)
              l1cnt_k_list.append(l1cnt_ev_k)
              bcid_j_list.append(bcid_ev_j)
              bcid_k_list.append(bcid_ev_k)

          # Hit maps from second pass: all hits, same events as trigger pixel
          plot_hit_map(os.path.join(px_dir, "hitmap"), hmap_pass2)
          plot_hit_map(os.path.join(px_dir, "hitmap_bcid_aligned"), hmap_bcid_align)

          if toa_j_list:
              toa_trg_for_j = np.concatenate(toa_trg_for_j_list)
              toa_trg_for_k = np.concatenate(toa_trg_for_k_list)
              toa_j_all     = np.concatenate(toa_j_list)
              toa_k_all     = np.concatenate(toa_k_list)
              plot_toa_comparison(
                  toa_trg_for_j, toa_trg_for_k, toa_j_all, toa_k_all,
                  layer_trigger=trigger, layer_j=layer_j, layer_k=layer_k,
                  label=f"pixel ({px_r},{px_c})",
                  output_path=os.path.join(px_dir, "toa")
              )
          else:
              print(f"  No TOA data found for pixel ({px_r},{px_c})")

          # --- Plot trigger TOA for events with no hit in each non-trigger layer ---
          toa_trg_no_j = np.concatenate(toa_trg_no_j_list) if toa_trg_no_j_list else np.array([])
          toa_trg_no_k = np.concatenate(toa_trg_no_k_list) if toa_trg_no_k_list else np.array([])
          toa_trg_no_j = toa_trg_no_j[np.isfinite(toa_trg_no_j)]
          toa_trg_no_k = toa_trg_no_k[np.isfinite(toa_trg_no_k)]
          print(f"  Events with no hit in layer {layer_j}: {len(toa_trg_no_j)}")
          print(f"  Events with no hit in layer {layer_k}: {len(toa_trg_no_k)}")

          fig, axes = plt.subplots(1, 2, figsize=(14, 6))
          for ax, toa_vals, absent_layer in zip(
                  axes,
                  [toa_trg_no_j, toa_trg_no_k],
                  [layer_j, layer_k]):
              if len(toa_vals) == 0:
                  ax.set_title(f"No events without hit in layer {absent_layer}")
                  continue
              mean, std = np.mean(toa_vals), np.std(toa_vals)
              ax.hist(toa_vals, bins=100,
                      range=(mean - 4*std, mean + 4*std),
                      color='#3f90da', edgecolor='black', linewidth=0.5)
              ax.set_xlabel(f"TOA trigger layer {trigger} (ns)")
              ax.set_ylabel("Entries")
              ax.set_title(f"Trigger TOA — no hit in layer {absent_layer}\n"
                           f"pixel ({px_r},{px_c})  N={len(toa_vals):,}")
          fig.tight_layout()
          out_no_hit = os.path.join(px_dir, "toa_trg_no_hit")
          fig.savefig(out_no_hit + ".png", dpi=150)
          fig.savefig(out_no_hit + ".pdf")
          plt.close(fig)
          print(f"  Saved: {out_no_hit}.png")

          # --- 2D plots: trigger event/l1counter/bcid vs other layer hit values ---
          def _cat(lst): return np.concatenate(lst) if lst else np.array([], dtype=np.int64)
          plot_event_quantities_comparison(
              event_trg_j=_cat(event_trg_for_j_list), event_j=_cat(event_j_list),
              l1cnt_trg_j=_cat(l1cnt_trg_for_j_list), l1cnt_j=_cat(l1cnt_j_list),
              bcid_trg_j=_cat(bcid_trg_for_j_list),   bcid_j=_cat(bcid_j_list),
              event_trg_k=_cat(event_trg_for_k_list), event_k=_cat(event_k_list),
              l1cnt_trg_k=_cat(l1cnt_trg_for_k_list), l1cnt_k=_cat(l1cnt_k_list),
              bcid_trg_k=_cat(bcid_trg_for_k_list),   bcid_k=_cat(bcid_k_list),
              layer_trigger=trigger, layer_j=layer_j, layer_k=layer_k,
              label=f"pixel ({px_r},{px_c})",
              output_path=os.path.join(px_dir, "event_quantities")
          )

    # -------------------------------------------------------------------------
    # Global deep-dive: TOA/event comparison for ALL events with exactly
    # 1 hit in every layer (not restricted to a single trigger pixel)
    # -------------------------------------------------------------------------
    analyze_global_one_hit_per_layer(
        run_files, trigger=trigger, layer_j=layer_j, layer_k=layer_k,
        step_size=step_size, total_batches=total_batches, output_dir=output_dir
    )


if __name__ == "__main__":
    main()
