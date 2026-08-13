"""
preselection.py - Filter telescope data and save preselected events
"""

import numpy as np
import uproot
import awkward as ak
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
import mplhep as hep
from tqdm import tqdm
import math
import sys
from scipy.optimize import curve_fit
hep.style.use("CMS")

# Your BRANCHES definition
BRANCHES = ["event", "row", "col", "row_global", "col_global", "toa_code", "tot_code", "cal_code", "nhits"]  # adjust as needed

# Set from --doGlobal in main(). When True, "row"/"col" are overwritten with
# "row_global"/"col_global" (range 0..32) right after each batch is read, so
# the rest of the pipeline is unaffected. N_PIX is the corresponding grid size.
DO_GLOBAL_COORDINATES = False
N_PIX                  = 16


def _apply_coordinate_choice(batch):
    """Overwrite row/col with row_global/col_global when --doGlobal is set.
    No-op otherwise, so default behavior is unchanged."""
    if not DO_GLOBAL_COORDINATES:
        return batch
    return ak.with_field(ak.with_field(batch, batch["row_global"], "row"),
                         batch["col_global"], "col")


def to_toa_ns(toa_code, cal_code):
    cal_safe = ak.where(cal_code > 0, cal_code, 1)
    toa      = 12.5 - 3.125 / cal_safe * toa_code
    return ak.where(cal_code > 0, toa, float('nan'))


def _gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_toa_diff(toa_ns_ref, toa_ns_trg):
    """Returns (mode, sigma) of toa_ns_ref - toa_ns_trg.

    Mode from histogram peak. Sigma from a Gaussian fit to the histogram,
    restricted to the window [mode - sigma_np, mode + sigma_np], where
    sigma_np is the RMS around the mode (used as fit window and seed).
    Falls back to sigma_np if the fit fails.
    """
    diff = ak.to_numpy(ak.flatten(toa_ns_ref - toa_ns_trg, axis=None)).astype(float)
    diff = diff[np.isfinite(diff)]
    counts, edges = np.histogram(diff, bins=200, range=(-10, 10))
    centers = (edges[:-1] + edges[1:]) / 2
    mode = float(centers[np.argmax(counts)])
    sigma_np = float(np.sqrt(np.mean((diff - mode) ** 2)))

    mask = (centers > mode - sigma_np) & (centers < mode + sigma_np)
    if mask.sum() < 3:
        return mode, sigma_np

    try:
        p0 = [counts[mask].max(), mode, sigma_np]
        popt, _ = curve_fit(_gauss, centers[mask], counts[mask], p0=p0)
        sigma = abs(popt[2])
    except RuntimeError:
        sigma = sigma_np

    return mode, sigma


def preselect_and_save_single(run_files_dict, output_dir, step_size=100000, concat_every=10, do1HitUniform = True, trigger=2):
    """
    Load events from three layers (single file per layer), apply 1-hit selection, and save filtered data

    Args:
        run_files_dict: Dictionary with keys 0,1,2 containing a single file path (str) for each layer
        output_dir: Directory to save preselected data
        step_size: Step size for uproot.iterate (number of events)
        concat_every: Concatenate and save every N batches
    """

    os.makedirs(output_dir, exist_ok=True)

    print("Starting preselection process (single file mode)...")
    print(f"Output directory: {output_dir}")

    # Check file sizes and warn if different
    n0 = uproot.open(f"{run_files_dict[1]}:events").num_entries
    n1 = uproot.open(f"{run_files_dict[2]}:events").num_entries
    n2 = uproot.open(f"{run_files_dict[3]}:events").num_entries
    print(f"File sizes: layer0={n0}, layer1={n1}, layer2={n2}")
    if not (n0 == n1 == n2):
        print(f"WARNING: files have different number of events ({n0}, {n1}, {n2}). "
              f"Will truncate each batch to the minimum length.")

    # Create iterators for all three layers (wrap single file in list)
    iter0 = uproot.iterate([f"{run_files_dict[1]}:events"], BRANCHES, step_size=step_size)
    iter1 = uproot.iterate([f"{run_files_dict[2]}:events"], BRANCHES, step_size=step_size)
    iter2 = uproot.iterate([f"{run_files_dict[3]}:events"], BRANCHES, step_size=step_size)


    # Temporary storage for batches
    temp_batches0 = []
    temp_batches1 = []
    temp_batches2 = []

    chunk_counter = 0
    total_events_in = 0
    total_any_hit_all_layers = 0
    total_after_trigger_1hit = 0
    total_trigger_ambiguous = 0
    total_events_out = 0

    # Row/col tolerance (in pixels) used to spatially match a trigger-layer hit
    # against hits in the other two layers when picking the trigger sel hit.
    NEIGHBOR_WINDOW = 2

    toa_bins        = np.linspace(0, 1000, 201)
    tot_bins        = np.linspace(0, 1000, 201)
    cal_bins        = np.linspace(0, 1000, 201)
    toa_pass_counts = np.zeros(200, dtype=np.int64)
    toa_fail_counts = np.zeros(200, dtype=np.int64)
    tot_pass_counts = np.zeros(200, dtype=np.int64)
    tot_fail_counts = np.zeros(200, dtype=np.int64)
    cal_pass_counts = np.zeros(200, dtype=np.int64)
    cal_fail_counts = np.zeros(200, dtype=np.int64)

    ref_layers      = [l for l in [0, 1, 2] if l != trigger]
    toa_ns_bins     = np.linspace(0, 15, 101)
    diff_bins       = np.linspace(-10, 10, 201)
    toa_2d_counts   = {r: np.zeros((100, 100), dtype=np.int64) for r in ref_layers}
    toa_diff_counts = {r: np.zeros(200,        dtype=np.int64) for r in ref_layers}

    # mode/sigma per ref layer, refit each time a chunk is saved (not per batch).
    # Used for the final toa_diff_layer*_vs_trg.png plot.
    plot_means  = {r: 0.0 for r in ref_layers}
    plot_sigmas = {r: 1.0 for r in ref_layers}

    nhits_counts     = {r: np.zeros(10, dtype=np.int64) for r in ref_layers}
    toa_trg_0hits    = {r: np.zeros(100, dtype=np.int64) for r in ref_layers}

    # Total number:
    total_events = min(n0, n1, n2)
    total_batches = math.ceil(total_events / step_size)
    print("Total events: ", total_events)
    print("Total batches: ", total_batches)

    # Process batches
    for i, (batch0, batch1, batch2) in enumerate(tqdm(zip(iter0, iter1, iter2), desc="Processing batches", total=total_batches)):
        batch0 = _apply_coordinate_choice(batch0)
        batch1 = _apply_coordinate_choice(batch1)
        batch2 = _apply_coordinate_choice(batch2)
        min_len = min(len(batch0), len(batch1), len(batch2))
        batch0 = batch0[:min_len]
        batch1 = batch1[:min_len]
        batch2 = batch2[:min_len]
        n_events = min_len
        total_events_in += n_events

        # Diagnostic: events with >=1 hit in every layer, before any other selection
        mask_any_hit_all_layers = (ak.num(batch0.row) >= 1) & \
                                   (ak.num(batch1.row) >= 1) & \
                                   (ak.num(batch2.row) >= 1)
        total_any_hit_all_layers += int(ak.sum(mask_any_hit_all_layers))

        if do1HitUniform:

            # Apply 1-hit mask across all three layers
            mask_1hit = (ak.num(batch0.row) == 1) & \
                        (ak.num(batch1.row) == 1) & \
                        (ak.num(batch2.row) == 1)
            total_after_trigger_1hit += int(ak.sum(mask_1hit))

            # Apply mask to each layer
            presel0 = batch0[mask_1hit]
            presel1 = batch1[mask_1hit]
            presel2 = batch2[mask_1hit]

        else:

            # Identify trigger and reference layers
            if trigger == 0:    batch_trg, batch_refA, batch_refB = batch0, batch1, batch2
            elif trigger == 1:  batch_trg, batch_refA, batch_refB = batch1, batch0, batch2
            else:                batch_trg, batch_refA, batch_refB = batch2, batch0, batch1

            # For each trigger-layer hit, check whether both reference layers have a hit
            # within +-NEIGHBOR_WINDOW in row and col. The sel hit is the trigger hit that
            # satisfies this in both reference layers; if more than one does, the event is
            # ambiguous and gets dropped (counted in total_trigger_ambiguous).
            def has_near_hit(trg_row, trg_col, other_row, other_col, window=NEIGHBOR_WINDOW):
                row_pairs = ak.cartesian({"t": trg_row, "o": other_row}, axis=1, nested=True)
                col_pairs = ak.cartesian({"t": trg_col, "o": other_col}, axis=1, nested=True)
                dr = row_pairs["t"] - row_pairs["o"]
                dc = col_pairs["t"] - col_pairs["o"]
                return ak.any((np.abs(dr) <= window) & (np.abs(dc) <= window), axis=2)

            near_A = has_near_hit(batch_trg.row, batch_trg.col, batch_refA.row, batch_refA.col)
            near_B = has_near_hit(batch_trg.row, batch_trg.col, batch_refB.row, batch_refB.col)
            # Only consider trigger hits with TOA_CODE and TOT_CODE above noise threshold
            # as sel candidates, to cut down on ambiguous (>1 candidate) events.
            trg_quality = (batch_trg.toa_code > 20) & (batch_trg.tot_code > 20)
            good_hit = near_A & near_B & trg_quality

            n_good = ak.sum(good_hit, axis=1)
            mask_1hit = n_good == 1
            total_after_trigger_1hit += int(ak.sum(mask_1hit))
            total_trigger_ambiguous += int(ak.sum(n_good > 1))

            # Accumulate TOA of trigger layer hits, split by pass/fail
            toa_pass = ak.to_numpy(ak.flatten(batch_trg[mask_1hit].toa_code,  axis=None))
            toa_fail = ak.to_numpy(ak.flatten(batch_trg[~mask_1hit].toa_code, axis=None))
            tot_pass = ak.to_numpy(ak.flatten(batch_trg[mask_1hit].tot_code,  axis=None))
            tot_fail = ak.to_numpy(ak.flatten(batch_trg[~mask_1hit].tot_code, axis=None))
            cal_pass = ak.to_numpy(ak.flatten(batch_trg[mask_1hit].cal_code,  axis=None))
            cal_fail = ak.to_numpy(ak.flatten(batch_trg[~mask_1hit].cal_code, axis=None))
            toa_pass_counts += np.histogram(toa_pass, bins=toa_bins)[0]
            toa_fail_counts += np.histogram(toa_fail, bins=toa_bins)[0]
            tot_pass_counts += np.histogram(tot_pass, bins=tot_bins)[0]
            tot_fail_counts += np.histogram(tot_fail, bins=tot_bins)[0]
            cal_pass_counts += np.histogram(cal_pass, bins=cal_bins)[0]
            cal_fail_counts += np.histogram(cal_fail, bins=cal_bins)[0]

            # Apply mask to all layers
            good_hit = good_hit[mask_1hit]
            presel0 = batch0[mask_1hit]
            presel1 = batch1[mask_1hit]
            presel2 = batch2[mask_1hit]

            # Identify the sel hit in the trigger layer: the single spatially-matched hit
            if trigger==0:
                presel0 = ak.with_field(presel0, ak.firsts(presel0["toa_code"][good_hit]), "toa_code_sel")
                presel0 = ak.with_field(presel0, ak.firsts(presel0["tot_code"][good_hit]), "tot_code_sel")
                presel0 = ak.with_field(presel0, ak.firsts(presel0["cal_code"][good_hit]), "cal_code_sel")
                presel0 = ak.with_field(presel0, ak.firsts(presel0["row"][good_hit]), "row_sel")
                presel0 = ak.with_field(presel0, ak.firsts(presel0["col"][good_hit]), "col_sel")
                presel_trg = presel0
            elif trigger==1:
                presel1 = ak.with_field(presel1, ak.firsts(presel1["toa_code"][good_hit]), "toa_code_sel")
                presel1 = ak.with_field(presel1, ak.firsts(presel1["tot_code"][good_hit]), "tot_code_sel")
                presel1 = ak.with_field(presel1, ak.firsts(presel1["cal_code"][good_hit]), "cal_code_sel")
                presel1 = ak.with_field(presel1, ak.firsts(presel1["row"][good_hit]), "row_sel")
                presel1 = ak.with_field(presel1, ak.firsts(presel1["col"][good_hit]), "col_sel")
                presel_trg = presel1
            elif trigger==2:
                presel2 = ak.with_field(presel2, ak.firsts(presel2["toa_code"][good_hit]), "toa_code_sel")
                presel2 = ak.with_field(presel2, ak.firsts(presel2["tot_code"][good_hit]), "tot_code_sel")
                presel2 = ak.with_field(presel2, ak.firsts(presel2["cal_code"][good_hit]), "cal_code_sel")
                presel2 = ak.with_field(presel2, ak.firsts(presel2["row"][good_hit]), "row_sel")
                presel2 = ak.with_field(presel2, ak.firsts(presel2["col"][good_hit]), "col_sel")
                presel_trg = presel2

            # Find the TOA acceptance with trigger (calibrated TOA, mean ± 3 sigma)
            presels = {0: presel0, 1: presel1, 2: presel2}
            means   = {}
            sigmas  = {}
            if trigger==0:
                mask = (ak.num(presel1.toa_code) == 1) & (ak.num(presel2.toa_code) == 1)
            elif trigger==1:
                mask = (ak.num(presel0.toa_code) == 1) & (ak.num(presel2.toa_code) == 1)
            elif trigger==2:
                mask = (ak.num(presel0.toa_code) == 1) & (ak.num(presel1.toa_code) == 1)
            toa_ns_trg_fit = to_toa_ns(presel_trg["toa_code_sel"][mask], presel_trg["cal_code_sel"][mask])
            trg_flat = ak.to_numpy(ak.flatten(toa_ns_trg_fit, axis=None)).astype(float)
            for r in ref_layers:
                toa_ns_ref_fit = to_toa_ns(presels[r].toa_code[mask], presels[r].cal_code[mask])
                means[r], sigmas[r] = fit_toa_diff(toa_ns_ref_fit, toa_ns_trg_fit)
                ref_flat  = ak.to_numpy(ak.flatten(toa_ns_ref_fit, axis=None)).astype(float)
                diff_flat = ref_flat - trg_flat
                finite    = np.isfinite(ref_flat) & np.isfinite(trg_flat)
                toa_2d_counts[r]   += np.histogram2d(trg_flat[finite], ref_flat[finite],
                                                      bins=[toa_ns_bins, toa_ns_bins])[0].astype(np.int64)
                toa_diff_counts[r] += np.histogram(diff_flat[np.isfinite(diff_flat)], bins=diff_bins)[0]
            mean0, sigma0 = means.get(0, 0), sigmas.get(0, 1)
            mean1, sigma1 = means.get(1, 0), sigmas.get(1, 1)
            mean2, sigma2 = means.get(2, 0), sigmas.get(2, 1)



            toa_ns_trg = to_toa_ns(presel_trg["toa_code_sel"], presel_trg["cal_code_sel"])

            if trigger==1 or trigger==2:
                # Select the hits that are compatible in layer 0
                toa_ns_ref0 = to_toa_ns(presel0["toa_code"], presel0["cal_code"])
                near0 = (np.abs(presel0["row"] - presel_trg["row_sel"]) <= NEIGHBOR_WINDOW) & \
                        (np.abs(presel0["col"] - presel_trg["col_sel"]) <= NEIGHBOR_WINDOW)
                mask_toa = (ak.fill_none(np.abs(toa_ns_ref0 - toa_ns_trg - mean0) < 5*sigma0, False)) & (presel0["toa_code"] > 20) & (presel0["tot_code"] > 20) & ak.fill_none(near0, False)
                presel0 = ak.with_field(presel0, presel0["toa_code"][mask_toa], "toa_code_sel")
                presel0 = ak.with_field(presel0, presel0["tot_code"][mask_toa], "tot_code_sel")
                presel0 = ak.with_field(presel0, presel0["cal_code"][mask_toa], "cal_code_sel")
                presel0 = ak.with_field(presel0, presel0["row"][mask_toa], "row_sel")
                presel0 = ak.with_field(presel0, presel0["col"][mask_toa], "col_sel")

            if trigger==0 or trigger==2:
                # Select the hits that are compatible in layer 1
                toa_ns_ref1 = to_toa_ns(presel1["toa_code"], presel1["cal_code"])
                near1 = (np.abs(presel1["row"] - presel_trg["row_sel"]) <= NEIGHBOR_WINDOW) & \
                        (np.abs(presel1["col"] - presel_trg["col_sel"]) <= NEIGHBOR_WINDOW)
                mask_toa = (ak.fill_none(np.abs(toa_ns_ref1 - toa_ns_trg - mean1) < 5*sigma1, False)) & (presel1["toa_code"] > 20) & (presel1["tot_code"] > 20) & ak.fill_none(near1, False)
                presel1 = ak.with_field(presel1, presel1["toa_code"][mask_toa], "toa_code_sel")
                presel1 = ak.with_field(presel1, presel1["tot_code"][mask_toa], "tot_code_sel")
                presel1 = ak.with_field(presel1, presel1["cal_code"][mask_toa], "cal_code_sel")
                presel1 = ak.with_field(presel1, presel1["row"][mask_toa], "row_sel")
                presel1 = ak.with_field(presel1, presel1["col"][mask_toa], "col_sel")

            if trigger==0 or trigger==1:
                # Select the hits that are compatible in layer 2
                toa_ns_ref2 = to_toa_ns(presel2["toa_code"], presel2["cal_code"])
                near2 = (np.abs(presel2["row"] - presel_trg["row_sel"]) <= NEIGHBOR_WINDOW) & \
                        (np.abs(presel2["col"] - presel_trg["col_sel"]) <= NEIGHBOR_WINDOW)
                mask_toa = (ak.fill_none(np.abs(toa_ns_ref2 - toa_ns_trg - mean2) < 5*sigma2, False)) & (presel2["toa_code"] > 20) & (presel2["tot_code"] > 20) & ak.fill_none(near2, False)
                presel2 = ak.with_field(presel2, presel2["toa_code"][mask_toa], "toa_code_sel")
                presel2 = ak.with_field(presel2, presel2["tot_code"][mask_toa], "tot_code_sel")
                presel2 = ak.with_field(presel2, presel2["cal_code"][mask_toa], "cal_code_sel")
                presel2 = ak.with_field(presel2, presel2["row"][mask_toa], "row_sel")
                presel2 = ak.with_field(presel2, presel2["col"][mask_toa], "col_sel")

            # Accumulate nhits-in-window and trigger TOA for 0-hit events (before final cut)
            presels_all = {0: presel0, 1: presel1, 2: presel2}
            toa_ns_trg_flat = ak.to_numpy(ak.flatten(toa_ns_trg, axis=None)).astype(float)
            for r in ref_layers:
                nhits = ak.to_numpy(ak.num(presels_all[r]["toa_code_sel"])).astype(int)
                nhits_clipped = np.clip(nhits, 0, 9)
                nhits_counts[r] += np.bincount(nhits_clipped, minlength=10)
                mask_0 = nhits == 0
                toa_trg_0hits[r] += np.histogram(
                    toa_ns_trg_flat[mask_0], bins=toa_ns_bins)[0].astype(np.int64)

            if trigger==0:
                mask_1hit = (ak.num(presel1["toa_code_sel"]) == 1) & (ak.num(presel2["toa_code_sel"]) == 1)
            elif trigger==1:
                mask_1hit = (ak.num(presel0["toa_code_sel"]) == 1) & (ak.num(presel2["toa_code_sel"]) == 1)
            elif trigger==2:
                mask_1hit = (ak.num(presel0["toa_code_sel"]) == 1) & (ak.num(presel1["toa_code_sel"]) == 1)

            presel0 = presel0[mask_1hit]
            presel1 = presel1[mask_1hit]
            presel2 = presel2[mask_1hit]
            

        n_selected = len(presel0)
        total_events_out += n_selected

        #print(f"  → Selected: {n_selected} events ({n_selected/n_events*100:.1f}%)")

        if n_selected > 0:
            temp_batches0.append(presel0)
            temp_batches1.append(presel1)
            temp_batches2.append(presel2)

        # Save to disk every N batches
        if (i + 1) % concat_every == 0 and len(temp_batches0) > 0:
            print(f"  Saving chunk {chunk_counter}...")

            # Concatenate temp batches
            chunk0 = ak.concatenate(temp_batches0)
            chunk1 = ak.concatenate(temp_batches1)
            chunk2 = ak.concatenate(temp_batches2)

            if not do1HitUniform:
                chunks = {0: chunk0, 1: chunk1, 2: chunk2}
                toa_ns_trg_chunk = to_toa_ns(chunks[trigger]["toa_code_sel"], chunks[trigger]["cal_code_sel"])
                for ref in [l for l in [0, 1, 2] if l != trigger]:
                    m, s = fit_toa_diff(to_toa_ns(chunks[ref]["toa_code_sel"], chunks[ref]["cal_code_sel"]), toa_ns_trg_chunk)
                    plot_means[ref], plot_sigmas[ref] = m, s
                    print(f"  TOA layer{ref}-trg: mean={m:.2f} ns  sigma={s:.2f} ns  5sigma_window=[{m-5*s:.2f}, {m+5*s:.2f}]")

            # Save to parquet (efficient format)
            ak.to_parquet(chunk0, f"{output_dir}/layer0_chunk_{chunk_counter}.parquet")
            ak.to_parquet(chunk1, f"{output_dir}/layer1_chunk_{chunk_counter}.parquet")
            ak.to_parquet(chunk2, f"{output_dir}/layer2_chunk_{chunk_counter}.parquet")

            print(f"  Chunk {chunk_counter} saved: {len(chunk0)} events")

            chunk_counter += 1

            # Clear temp storage
            temp_batches0 = []
            temp_batches1 = []
            temp_batches2 = []

    # Save remaining batches
    if len(temp_batches0) > 0:
        print(f"Saving final chunk {chunk_counter}...")

        chunk0 = ak.concatenate(temp_batches0)
        chunk1 = ak.concatenate(temp_batches1)
        chunk2 = ak.concatenate(temp_batches2)

        if not do1HitUniform:
            chunks = {0: chunk0, 1: chunk1, 2: chunk2}
            toa_ns_trg_chunk = to_toa_ns(chunks[trigger]["toa_code_sel"], chunks[trigger]["cal_code_sel"])
            for ref in [l for l in [0, 1, 2] if l != trigger]:
                m, s = fit_toa_diff(to_toa_ns(chunks[ref]["toa_code_sel"], chunks[ref]["cal_code_sel"]), toa_ns_trg_chunk)
                plot_means[ref], plot_sigmas[ref] = m, s
                print(f"  TOA layer{ref}-trg: mean={m:.2f} ns  sigma={s:.2f} ns  5sigma_window=[{m-5*s:.2f}, {m+5*s:.2f}]")

        ak.to_parquet(chunk0, f"{output_dir}/layer0_chunk_{chunk_counter}.parquet")
        ak.to_parquet(chunk1, f"{output_dir}/layer1_chunk_{chunk_counter}.parquet")
        ak.to_parquet(chunk2, f"{output_dir}/layer2_chunk_{chunk_counter}.parquet")

        print(f"Final chunk {chunk_counter} saved: {len(chunk0)} events")
        chunk_counter += 1

    print(f"\n=== Preselection Summary ===")
    print(f"  {'Step':<35} {'Events':>10}   {'Step eff':>9}   {'Cumul eff':>9}")
    print(f"  {'Input':<35} {total_events_in:>10,}   {'':>9}   {'100.00%':>9}")
    eff_any_hit = total_any_hit_all_layers / total_events_in * 100 if total_events_in else 0
    print(f"  {'>=1 hit in every layer':<35} {total_any_hit_all_layers:>10,}   {eff_any_hit:>8.2f}%   {eff_any_hit:>8.2f}%")
    if do1HitUniform:
        eff_1hit = total_after_trigger_1hit / total_events_in * 100
        print(f"  {'1-hit all layers':<35} {total_after_trigger_1hit:>10,}   {eff_1hit:>8.2f}%   {eff_1hit:>8.2f}%")
    else:
        eff_trg   = total_after_trigger_1hit / total_events_in * 100
        eff_toa   = total_events_out / total_after_trigger_1hit * 100 if total_after_trigger_1hit else 0
        eff_final = total_events_out / total_events_in * 100
        print(f"  {'1-hit trigger layer':<35} {total_after_trigger_1hit:>10,}   {eff_trg:>8.2f}%   {eff_trg:>8.2f}%")
        print(f"  {'TOA window + 1-hit ref layers':<35} {total_events_out:>10,}   {eff_toa:>8.2f}%   {eff_final:>8.2f}%")
        print(f"  {'Discarded (ambiguous trigger hit)':<35} {total_trigger_ambiguous:>10,}")
    print(f"  {'Output':<35} {total_events_out:>10,}   {'':>9}   {total_events_out/total_events_in*100:>8.2f}%")
    print(f"\nTotal chunks saved: {chunk_counter}")
    print(f"Output directory: {output_dir}")

    if not do1HitUniform:
        # 2D and 1D TOA plots per reference layer
        toa_centers  = (toa_ns_bins[:-1] + toa_ns_bins[1:]) / 2
        diff_centers = (diff_bins[:-1]   + diff_bins[1:])   / 2
        for r in ref_layers:
            m, s = plot_means[r], plot_sigmas[r]

            fig, ax = plt.subplots(figsize=(9, 8))
            ax.pcolormesh(toa_ns_bins, toa_ns_bins, toa_2d_counts[r].T,
                          cmap='viridis', rasterized=True)
            x = toa_centers
            ax.plot(x, x + m,       color='red', ls='-',  linewidth=1.2,
                    label=rf"$\Delta={m:.2f}$ ns")
            ax.plot(x, x + m - 5*s, color='red', ls='--', linewidth=0.8)
            ax.plot(x, x + m + 5*s, color='red', ls='--', linewidth=0.8,
                    label=rf"$\pm5\sigma={5*s:.2f}$ ns")
            ax.set_xlabel(f"TOA trigger layer {trigger} (ns)")
            ax.set_ylabel(f"TOA layer {r} (ns)")
            ax.legend()
            ax.set_xlim(0, 15)
            ax.set_ylim(0, 15)
            fig.tight_layout()
            out = os.path.join(output_dir, f"toa_2d_layer{r}_vs_trg")
            fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf"); plt.close(fig)
            print(f"Saved: {out}.png")

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.step(diff_centers, toa_diff_counts[r], where='mid', color='#3f90da', linewidth=1.2)
            ax.axvline(m,       color='red', ls='-',  linewidth=1.2, label=rf"mode={m:.2f} ns")
            ax.axvline(m - 5*s, color='red', ls='--', linewidth=0.8, label=rf"$\pm5\sigma={5*s:.2f}$ ns")
            ax.axvline(m + 5*s, color='red', ls='--', linewidth=0.8)
            ax.set_xlabel(f"TOA layer{r} - TOA trigger (ns)")
            ax.set_ylabel("Events")
            ax.legend()
            fig.tight_layout()
            out = os.path.join(output_dir, f"toa_diff_layer{r}_vs_trg")
            fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf"); plt.close(fig)
            print(f"Saved: {out}.png")

        # nhits-in-window per ref layer
        for r in ref_layers:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.bar(np.arange(10), nhits_counts[r], color='#3f90da', edgecolor='black')
            ax.set_xlabel("Hits in TOA window per event")
            ax.set_ylabel("Events")
            ax.set_xticks(np.arange(10))
            ax.set_xticklabels([str(n) if n < 9 else '9+' for n in range(10)])
            fig.tight_layout()
            out = os.path.join(output_dir, f"nhits_window_layer{r}")
            fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf"); plt.close(fig)
            print(f"Saved: {out}.png")

        # trigger TOA (ns) for events with 0 hits in ref layer
        toa_ns_centers = (toa_ns_bins[:-1] + toa_ns_bins[1:]) / 2
        for r in ref_layers:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.step(toa_ns_centers, toa_trg_0hits[r], where='mid',
                    color='#e42536', linewidth=1.2)
            ax.set_xlabel(f"TOA trigger layer {trigger} (ns)")
            ax.set_ylabel("Events")
            fig.tight_layout()
            out = os.path.join(output_dir, f"toa_trg_0hits_layer{r}")
            fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf"); plt.close(fig)
            print(f"Saved: {out}.png")

        n_pass = total_after_trigger_1hit
        n_fail = total_events_in - total_after_trigger_1hit
        for centers, pass_c, fail_c, label, fname in [
            ((toa_bins[:-1] + toa_bins[1:]) / 2, toa_pass_counts, toa_fail_counts, "TOA code", "toa_trigger_1hit"),
            ((tot_bins[:-1] + tot_bins[1:]) / 2, tot_pass_counts, tot_fail_counts, "TOT code", "tot_trigger_1hit"),
            ((cal_bins[:-1] + cal_bins[1:]) / 2, cal_pass_counts, cal_fail_counts, "CAL code", "cal_trigger_1hit"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.step(centers, pass_c, where='mid', color='green',
                    label=f"Pass 1-hit ({n_pass:,})", linewidth=1.2)
            ax.step(centers, fail_c, where='mid', color='red',
                    label=f"Fail 1-hit ({n_fail:,})", linewidth=1.2)
            ax.set_xlabel(f"{label} (trigger layer)")
            ax.set_ylabel("Hits")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=150)
            fig.savefig(os.path.join(output_dir, f"{fname}.pdf"))
            plt.close(fig)
            print(f"Saved: {output_dir}/{fname}.png")

    return chunk_counter


def preselect_and_save(run_files_dict, output_dir, step_size="200 MB", concat_every=10, target=0):
    """
    Load events from three layers, apply 1-hit selection, and save filtered data
    
    Args:
        run_files_dict: Dictionary with keys 0,1,2 containing file lists for each layer
        output_dir: Directory to save preselected data
        step_size: Step size for uproot.iterate
        concat_every: Concatenate and save every N batches
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting preselection process...")
    print(f"Output directory: {output_dir}")
    
    # Create iterators for all three layers
    iter0 = uproot.iterate(run_files_dict[0], BRANCHES, step_size=step_size)
    iter1 = uproot.iterate(run_files_dict[1], BRANCHES, step_size=step_size)
    iter2 = uproot.iterate(run_files_dict[2], BRANCHES, step_size=step_size)
    
    # Temporary storage for batches
    temp_batches0 = []
    temp_batches1 = []
    temp_batches2 = []
    
    chunk_counter = 0
    total_events_in = 0
    total_events_out = 0
    
    # Process batches
    for i, (batch0, batch1, batch2) in enumerate(zip(iter0, iter1, iter2)):
        batch0 = _apply_coordinate_choice(batch0)
        batch1 = _apply_coordinate_choice(batch1)
        batch2 = _apply_coordinate_choice(batch2)
        n_events = len(batch0)
        total_events_in += n_events
        
        print(f"Processing batch {i+1}, events: {n_events}")
        
        # Apply 1-hit mask across all three layers
        if target==0:
            mask_1hit = (ak.num(batch0.row) == 1)
        elif target==1:
            mask_1hit = (ak.num(batch1.row) == 1)
        elif target==2:
            mask_1hit = (ak.num(batch2.row) == 1)
        else:
            mask_1hit = (ak.num(batch0.row) == 1) & \
                        (ak.num(batch1.row) == 1) & \
                        (ak.num(batch2.row) == 1)

        # Apply mask to each layer
        presel0 = batch0[mask_1hit]
        presel1 = batch1[mask_1hit]
        presel2 = batch2[mask_1hit]
        
        n_selected = len(presel0)
        total_events_out += n_selected
        
        #print(f"  → Selected: {n_selected} events ({n_selected/n_events*100:.1f}%)")
        
        if n_selected > 0:
            temp_batches0.append(presel0)
            temp_batches1.append(presel1)
            temp_batches2.append(presel2)
        
        # Save to disk every N batches
        if (i + 1) % concat_every == 0 and len(temp_batches0) > 0:
            print(f"  Saving chunk {chunk_counter}...")
            
            # Concatenate temp batches
            chunk0 = ak.concatenate(temp_batches0)
            chunk1 = ak.concatenate(temp_batches1)
            chunk2 = ak.concatenate(temp_batches2)
            
            # Save to parquet (efficient format)
            ak.to_parquet(chunk0, f"{output_dir}/layer0_chunk_{chunk_counter}.parquet")
            ak.to_parquet(chunk1, f"{output_dir}/layer1_chunk_{chunk_counter}.parquet")
            ak.to_parquet(chunk2, f"{output_dir}/layer2_chunk_{chunk_counter}.parquet")
            
            print(f"  Chunk {chunk_counter} saved: {len(chunk0)} events")
            
            chunk_counter += 1
            
            # Clear temp storage
            temp_batches0 = []
            temp_batches1 = []
            temp_batches2 = []
    
    # Save remaining batches
    if len(temp_batches0) > 0:
        print(f"Saving final chunk {chunk_counter}...")
        
        chunk0 = ak.concatenate(temp_batches0)
        chunk1 = ak.concatenate(temp_batches1)
        chunk2 = ak.concatenate(temp_batches2)
        
        ak.to_parquet(chunk0, f"{output_dir}/layer0_chunk_{chunk_counter}.parquet")
        ak.to_parquet(chunk1, f"{output_dir}/layer1_chunk_{chunk_counter}.parquet")
        ak.to_parquet(chunk2, f"{output_dir}/layer2_chunk_{chunk_counter}.parquet")
        
        print(f"Final chunk {chunk_counter} saved: {len(chunk0)} events")
        chunk_counter += 1
    
    print(f"\n=== Preselection Summary ===")
    print(f"Total input events: {total_events_in}")
    print(f"Total selected events: {total_events_out}")
    print(f"Selection efficiency: {total_events_out/total_events_in*100:.2f}%")
    print(f"Total chunks saved: {chunk_counter}")
    print(f"Output directory: {output_dir}")
    
    return chunk_counter

def main():
    # Configuration
    is_final = True
    if is_final:
        data_path = "/eos/project/m/mtd-etl-system-test/public/DesyNovember2025/Final/"
        data_path = "/eos/project/m/mtd-etl-system-test/public/SPS_April_2026/Final/"
        data_path = "/eos/project/m/mtd-etl-system-test/public/SPS_April_2026/Final/"
        data_path = "/eos/project/m/mtd-etl-system-test/public/Test Beam Data/SPS_April_2026/Final/"
        data_path = "/eos/project/m/mtd-etl-system-test/public/Test Beam Data/SPS_April_2026/rereco_bcid_matching/"
        data_path = "/eos/project/m/mtd-etl-system-test/public/Test Beam Data/SPS_April_2026/rereco_2/"
        data_path = "/eos/project/m/mtd-etl-system-test/public/Test Beam Data/SPS_April_2026/rereco_bcid_matching_2/"
        data_path = "/eos/user/f/fernance/ETL/Bootstrap/July2026/Data/"
    else:
        #data_path = "/eos/user/f/fernance/ETL/TEST-Bootstrap/Nov2025-TEST/Test-0T"
        data_path = "/eos/project/m/mtd-etl-system-test/public/DesyNovember2025/merged/"
    argv = sys.argv[1:]
    global DO_GLOBAL_COORDINATES, N_PIX
    DO_GLOBAL_COORDINATES = "--doGlobal" in argv
    N_PIX                 = 33 if DO_GLOBAL_COORDINATES else 16
    argv = [a for a in argv if a != "--doGlobal"]

    if not DO_GLOBAL_COORDINATES:
        BRANCHES.remove('row_global')
        BRANCHES.remove('col_global')

    run_start = int(argv[0]) # 133082
    run_stop = int(argv[1]) # 133181
    trigger = int(argv[2]) # 0
    target = f"trigger-{trigger}"
    output_dir = f"/eos/user/f/fernance/ETL/Bootstrap/July2026/Test-bootstrap_{run_start}_{run_stop}_target-{target}_OPTSEL"

    ### Selection options
    do1HitUniform = False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get run files for each layer
    print("Collecting run files...")
    run_files = {}
    for layer in range(1, 3+1):
        if is_final:
            #run_files[layer] = data_path + f'run_{run_start}_{run_stop}_rb{layer}.root'
            run_files[layer] = data_path + f'final_run_{run_start}_{run_stop}_rb{layer}.root'
        else:
            run_files[layer] = get_run_files(
                data_path,
                run_start,
                run_stop,
                r"run_(\d+)_rb%i.root" % layer,
                verbose=True
            )
            print(f"Layer {layer}: {len(run_files[layer])} files")

    # Clean: keep only runs that have all 3 layers
    if not is_final:
        import re
        print("\nCleaning: keeping only runs with all 3 layers...")

        # Extract run numbers from each layer
        def get_run_number(filepath):
            match = re.search(r'run_(\d+)_rb\d\.root', os.path.basename(filepath))
            return int(match.group(1)) if match else None

        runs_per_layer = {}
        for layer in range(3):
            runs_per_layer[layer] = set(get_run_number(f) for f in run_files[layer])

        # Find common runs
        common_runs = runs_per_layer[0] & runs_per_layer[1] & runs_per_layer[2]
        print(f"  Common runs: {len(common_runs)}")

        # Filter file lists
        for layer in range(3):
            original_count = len(run_files[layer])
            run_files[layer] = [f for f in run_files[layer] if get_run_number(f) in common_runs]
            print(f"  Layer {layer}: {original_count} → {len(run_files[layer])} files")

    # Run preselection
    if is_final:
        preselect_and_save_single(
            run_files,
            output_dir,
            step_size=500000,
            concat_every=10,
            do1HitUniform=do1HitUniform,
            trigger=trigger
        )
    else:
        preselect_and_save(
            run_files,
            output_dir,
            step_size="200 MB",
            concat_every=10,
            target=target
        )
    
    print("\nPreselection complete!")

if __name__ == "__main__":
    main()