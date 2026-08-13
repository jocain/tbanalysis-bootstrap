"""
do_bcid_alignment.py - Check BCID alignment across ETL telescope modules.

Plots BCID vs sequential event number for each layer.
With --align: synchronises the three layers by dropping extra events,
reads all data branches and saves aligned parquet files.

Usage:
    python do_bcid_alignment.py <run_start> <run_stop> [--data-path ...] [--output-dir ...]
                                [--nevents N] [--align]
"""

import sys
import os
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm

hep.style.use("CMS")

CHUNK_SIZE = 5000
BRANCHES   = ["event", "l1counter", "row", "col", "toa_code", "tot_code",
               "cal_code", "elink", "chipid", "bcid", "nhits"]


class BCIDBuffer:
    """Reads BCID from a ROOT file in chunks, maintaining a rolling buffer."""

    def __init__(self, path, chunk_size):
        self.f          = uproot.open(f"{path}:events")
        self.chunk_size = chunk_size
        self.total      = self.f.num_entries
        self.file_pos   = 0
        self.base       = 0   # global index of buf[0]
        self.buf        = np.array([], dtype=np.int32)
        self.ptr        = 0
        self._load()

    def _load(self):
        if self.file_pos >= self.total:
            return
        stop          = min(self.file_pos + self.chunk_size, self.total)
        arr           = self.f["bcid"].array(entry_start=self.file_pos, entry_stop=stop)
        tail          = self.buf[self.ptr:]
        self.base    += self.ptr
        self.buf      = np.concatenate([tail, ak.to_numpy(arr).astype(np.int32)])
        self.ptr      = 0
        self.file_pos = stop

    @property
    def done(self):
        return self.ptr >= len(self.buf) and self.file_pos >= self.total

    @property
    def available(self):
        return len(self.buf) - self.ptr

    def ensure(self):
        if self.available < self.chunk_size // 2 and self.file_pos < self.total:
            self._load()

    def peek(self, n):
        return self.buf[self.ptr:self.ptr + n]

    def global_idx(self):
        return self.base + self.ptr

    def current(self):
        return int(self.buf[self.ptr])

    def advance(self, n):
        self.ptr += n
        self.ensure()


def align_bcids_chunked(run_files, n0, n1, n2, chunk_size=CHUNK_SIZE):
    """
    3-pointer BCID alignment reading chunk_size events at a time per layer.
    Returns aligned global index arrays for each layer.
    """
    bufs      = [BCIDBuffer(run_files[i], chunk_size) for i in range(3)]
    idx_lists = [[], [], []]

    with tqdm(total=min(n0, n1, n2), desc="Aligning BCIDs", unit="ev") as pbar:
        while not any(b.done for b in bufs):
            avail = min(b.available for b in bufs)
            if avail == 0:
                break

            b = [bufs[i].peek(avail) for i in range(3)]
            mismatch = np.where((b[0] != b[1]) | (b[0] != b[2]))[0]
            k = int(mismatch[0]) if len(mismatch) > 0 else avail

            if k > 0:
                for i, buf in enumerate(bufs):
                    idx_lists[i].append(
                        np.arange(buf.global_idx(), buf.global_idx() + k, dtype=np.int64)
                    )
                    buf.advance(k)
                pbar.update(k)

            if any(b.done for b in bufs):
                break
            if not all(b.available > 0 for b in bufs):
                break

            vals  = [b.current() for b in bufs]
            min_b = min(vals)
            for buf, v in zip(bufs, vals):
                if v == min_b:
                    buf.advance(1)

    return tuple(
        np.concatenate(lst) if lst else np.array([], dtype=np.int64)
        for lst in idx_lists
    )


def save_aligned_data(run_files, indices, output_dir, run_start, run_stop, chunk_size=CHUNK_SIZE):
    """Read all branches and save aligned events as ROOT files, chunk_size events at a time."""
    rb_map = {0: 1, 1: 2, 2: 3}
    for layer, idx in zip([0, 1, 2], indices):
        out = os.path.join(output_dir,
                           f"final_run_{run_start}_{run_stop}_rb{rb_map[layer]}.root")
        print(f"  Writing layer {layer} ({len(idx):,} events) -> {out}")
        f = uproot.open(f"{run_files[layer]}:events")

        with uproot.recreate(out) as fout:
            for i in tqdm(range(0, len(idx), chunk_size), desc=f"  Layer {layer}", unit="chunk"):
                idx_chunk = idx[i:i + chunk_size]
                start     = int(idx_chunk[0])
                stop      = int(idx_chunk[-1]) + 1
                batch     = f["events"].arrays(BRANCHES, entry_start=start, entry_stop=stop,
                                               library="ak")
                local_idx = idx_chunk - start
                chunk     = batch[local_idx]
                if i == 0:
                    fout["events"] = {b: chunk[b] for b in BRANCHES}
                else:
                    fout["events"].extend({b: chunk[b] for b in BRANCHES})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_start', type=int)
    parser.add_argument('run_stop',  type=int)
    parser.add_argument('--data-path', type=str,
                        default="/eos/project/m/mtd-etl-system-test/public/Test Beam Data/SPS_April_2026/rereco_bcid_matching/")
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--nevents', type=int, default=10000,
                        help='Number of events to plot (default: 10000)')
    parser.add_argument('--align', action='store_true',
                        help='Align layers by BCID, save data and indices')
    args = parser.parse_args()

    output_dir = args.output_dir or f"bcid_alignment_{args.run_start}_{args.run_stop}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    rb_map = {0: 1, 1: 2, 2: 3}
    run_files = {}
    for layer in [0, 1, 2]:
        path = os.path.join(
            args.data_path,
            f"final_run_{args.run_start}_{args.run_stop}_rb{rb_map[layer]}.root"
        )
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)
        run_files[layer] = path

    n0 = uproot.open(f"{run_files[0]}:events").num_entries
    n1 = uproot.open(f"{run_files[1]}:events").num_entries
    n2 = uproot.open(f"{run_files[2]}:events").num_entries
    print(f"File sizes: layer0={n0}, layer1={n1}, layer2={n2}")
    if not (n0 == n1 == n2):
        print(f"WARNING: files have different number of events ({n0}, {n1}, {n2}).")

    # -------------------------------------------------------------------------
    # Alignment mode
    # -------------------------------------------------------------------------
    if args.align:
        print(f"\nStep 1 — Aligning BCIDs in chunks of {CHUNK_SIZE}...")
        idx0, idx1, idx2 = align_bcids_chunked(run_files, n0, n1, n2, CHUNK_SIZE)
        indices   = [idx0, idx1, idx2]
        n_aligned = len(idx0)

        print(f"\n{'':>20} {'Before':>12} {'After':>12} {'Dropped':>12}")
        print(f"  {'Layer 0':<18} {n0:>12,} {n_aligned:>12,} {n0-n_aligned:>11,}  ({100*(n0-n_aligned)/n0:.2f}%)")
        print(f"  {'Layer 1':<18} {n1:>12,} {n_aligned:>12,} {n1-n_aligned:>11,}  ({100*(n1-n_aligned)/n1:.2f}%)")
        print(f"  {'Layer 2':<18} {n2:>12,} {n_aligned:>12,} {n2-n_aligned:>11,}  ({100*(n2-n_aligned)/n2:.2f}%)")

        np.save(os.path.join(output_dir, "aligned_idx_layer0.npy"), idx0)
        np.save(os.path.join(output_dir, "aligned_idx_layer1.npy"), idx1)
        np.save(os.path.join(output_dir, "aligned_idx_layer2.npy"), idx2)

        print(f"\nStep 2 — Saving aligned data...")
        save_aligned_data(run_files, indices, output_dir, args.run_start, args.run_stop, CHUNK_SIZE)

        # BCIDs for plot: take from the aligned index arrays directly
        nevents_plot = min(args.nevents, n_aligned)
        bcid_plot = []
        for layer, idx in zip([0, 1, 2], indices):
            with uproot.open(f"{run_files[layer]}:events") as f:
                arr = f["bcid"].array(entry_stop=int(idx[nevents_plot - 1]) + 1)
                bcid_plot.append(ak.to_numpy(arr).astype(np.int32)[idx[:nevents_plot]])
        suffix = "_aligned"

    else:
        nevents_plot = min(args.nevents, n0, n1, n2)
        print(f"Reading first {nevents_plot} events...")
        bcid_plot = []
        for layer in [0, 1, 2]:
            with uproot.open(f"{run_files[layer]}:events") as f:
                arr = f["bcid"].array(entry_stop=nevents_plot)
                bcid_plot.append(ak.to_numpy(arr).astype(np.int32))
        suffix = ""

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    colors     = ['#3f90da', '#e42536', '#f89c20']
    runs_label = f"runs {args.run_start}–{args.run_stop}"

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, c in enumerate(colors):
        ax.plot(np.arange(len(bcid_plot[i])), bcid_plot[i],
                color=c, label=f"Layer {i}", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Event number")
    ax.set_ylabel("BCID")
    ax.set_title(f"BCID vs event number{' (aligned)' if args.align else ''} — {runs_label}")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(output_dir, f"bcid_vs_event{suffix}")
    fig.savefig(out + ".png", dpi=150)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print(f"Saved: {out}.png")


if __name__ == "__main__":
    main()
