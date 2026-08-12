# Bootstrap analysis

Standalone bootstrap pipeline (preselection -> step1 -> step2 -> step3 -> step4).

## Environment

Create a local venv in this directory (this is also what the condor jobs expect
via `$WORK_DIR/venv`):

```
python3 -m venv venv
source venv/bin/activate
pip install numpy==1.26.1 awkward==2.6.4 scipy==1.13.1 hist==2.7.2 mplhep==0.3.48 \
            matplotlib==3.8.4 uproot==5.3.7 scikit-learn==1.5.0 lmfit==1.3.1 \
            pyarrow==15.0 tqdm pandas
```

Activate it (`source venv/bin/activate`) before running any script by hand.
Condor jobs (`condor/executable_*.sh`) source it automatically through
`$WORK_DIR/venv/bin/activate`, so nothing else needs to be exported.

## Pipeline

1. **Preselection** — `do_bootstrap_preselection.py` (via `condor/launch_preselection.sh`
   + `condor/submit_preselection.sub`)
2. **Step 1** — `do_bootstrap_multiple_step1.py` (via `condor/launch_step1.sh`
   + `condor/submit_step1.sub`)
3. **Step 2** — `do_bootstrap_multiple_step2.py` (via `condor/launch_step2.sh`
   + `condor/submit_step2.sub`)
4. **Step 3** — `do_bootstrap_multiple_step3.py`, run locally (no condor submit file)
5. **Step 4** — `do_bootstrap_multiple_step4.py`, run locally (no condor submit file)

Three layers are involved throughout (`i`, `j`, `k` — mapped to the physical
telescope layers 0/1/2 via `--layer-i/-j/-k`). TOA/TOT are always derived from
the raw TDC codes the same way:

```
toa_ns = 12.5 - 3.125/cal_code * toa_code      # valid only for cal_code > 0
tot_ns = (2*tot_code - floor(tot_code/32)) * 3.125 / cal_code
```

`--doGlobal` (preselection, step1, step3) switches row/col for row_global/col_global,
i.e. the 32x32 telescope-wide pixel grid (stitching the module's 4 ETROCs) instead of
the default 16x16 single-ETROC grid. It must be set consistently across every step
that reads a given run's output.

### 1. Preselection — what it selects

Reads one ROOT file per layer (`rb1/rb2/rb3`) in chunks and, per chunk:

1. Picks a **trigger layer** (`--trigger 0/1/2` from the CLI). Candidate trigger
   hits must pass a basic noise cut: `toa_code > 20` and `tot_code > 20`.
2. A trigger hit is kept only if there is **at least one hit within +-2 pixels**
   (row and col) in **both** other ("reference") layers. Events where more than
   one trigger hit satisfies this are **ambiguous and dropped** (counted
   separately as "Discarded (ambiguous trigger hit)" in the summary printout).
   Of course, this assumed that the layers are resonably aligned. Looseing this did
   not probe to increase the efficiency.
3. From the surviving events (exactly 1 hit in trigger + both reference layers),
   fits the TOA(reference) - TOA(trigger) distribution per reference layer to get
   a per-layer `(mean, sigma)` timing calibration (mode of the histogram, then a
   Gaussian fit in a +-1 sigma window around it).
4. Final per-reference-layer hit selection: a hit is kept if it is within +-2
   pixels of the trigger's selected pixel, has `toa_code > 20` and `tot_code > 20`,
   **and** its TOA falls within `mean +- 5*sigma` of the trigger's TOA (the
   calibration from step 3).
5. Requires **exactly one** surviving hit per reference layer; ambiguous/zero-hit
   events are dropped.

Output: `layer{0,1,2}_chunk_N.parquet` with the selected hit's fields as
`row_sel/col_sel/toa_code_sel/tot_code_sel/cal_code_sel`, plus diagnostic plots
(2D TOA correlation and TOA-difference per reference layer, hits-in-window
histograms, TOA/TOT/CAL pass/fail distributions for the trigger-hit selection).

### 2. Step 1 — pixel selection, combinations, timewalk correction

Loads the `*_sel` columns from all preselection chunks, then narrows further to
build the analysis sample:

- **Trigger TOA window**: `50 < toa_code_sel < 1000` on whichever layer is
  `--trigger` (`i`/`j`/`k`, default `k`).
- **TOT window per layer**: centered on the mode of `tot_code_sel` (histogram
  peak), using the [0.5, 99.5] percentile within a +-50-code window around it,
  capped at 200.
- **CAL_CODE window per layer**: per-pixel, `|cal_code - mode(cal_code @ pixel)| < 2`.
- A `100 < toa_code_sel < 550` window is computed and plotted for diagnostics but
  is **not applied** as a cut (commented out in the code).

For each pixel `(row_i, col_i)` in layer i with more than `--min-hits` (default
500) hits:

1. Find all `(row_j, col_j, row_k, col_k)` combinations sharing >=3 events with
   that pixel; keep those with rate > 200 hits (or only the single best one with
   `--use-best 1`).
2. For each kept combination, run `--iterations` (default 4) of the iterative
   timewalk correction: compute each layer's `dTOA` (average of the other two
   minus this one), fit `dTOA` vs `TOT` — linear (`--tw-fit-type linear`) or a
   degree-2 polynomial (`--tw-fit-type twc`, default, matching `twc.py`) — correct
   the TOA with it, and repeat.
3. After the last iteration, write the corrected (and raw, pre-correction) event-
   by-event `T_ij = TOA_i - TOA_j`, `T_jk`, `T_ki` to
   `<output-dir>/step1/corrected_deltaT_<rowcol_i><rowcol_j><rowcol_k>.root`.

The real output of step 1 is the `corrected_deltaT_*.root` files it writes per
pixel/combination — `run_bootstrap_analysis()` doesn't return a results dict (its
only `return` is an early `return None` when no combination is found), so no
per-run JSON summary is produced. `--do-light` does work (skips the per-iteration
diagnostic plots).

Example command for running locally:
```
python do_bootstrap_multiple_step1.py \
  --input-dir "/eos/user/f/fernance/ETL/Bootstrap/July2026/Test-bootstrap_149132_149203_target-trigger-2_OPTSEL/" \
  --output-dir "test_results/" \
  --layer-i 0 --layer-j 1 --layer-k 2 \
  --trigger k \
  --do-light 1 --use-best 1 \
  --doGlobal --do-limit 3 \
```

But recomended way is to run through condor **after modifying condor/submit_step1.sub**:
```
sh condor/launch_step1.sh step1_runstart_runstop
```

### 3. Step 2 — pairwise timing resolution

Reads `corrected_deltaT_*.root` from a step1 output dir (`--input-dir`, typically
`.../step1`) and, for each pixel-pair (`ij`, `jk`, `ki` — `--pair` restricts to
one), concatenates the corresponding `T_ij`/`T_jk`/`T_ki` branch (or the `_raw`
branch with `--do-raw 1`) across every third-pixel combination sharing that pair.
Fits Gaussian-mixture models with 1, 2 and 3 components, picks the one with the
best Kolmogorov-Smirnov goodness-of-fit, and converts its FWHM to a sigma:
`sigma_ps = FWHM / 2.355 * 1000`.

Output: `sigma_ij.json` / `sigma_jk.json` / `sigma_ki.json` (and `_raw` variants),
each `{pixel_a: {pixel_b: {fwhm, sigma, n, n_gaussians}}}`.

These are the ingredients to compute the resolution in step 3.

Running locally all:
```
python do_bootstrap_multiple_step2.py --input-dir test_results/step1/ --output-dir test_results/step2/
```

Restricting to a given pair:
```
python do_bootstrap_multiple_step2.py --input-dir test_results/step1/ --output-dir test_results/step2/ --pair ij
python do_bootstrap_multiple_step2.py --input-dir test_results/step1/ --output-dir test_results/step2/ --pair jk
python do_bootstrap_multiple_step2.py --input-dir test_results/step1/ --output-dir test_results/step2/ --pair ik
```

Running in condor (recommended, after modifying `condor/submit_step2.sub` — it
queues one job per pair, so `ij`/`jk`/`ki` run in parallel):
```
sh condor/launch_step2.sh step2_runstart_runstop
```

There is also an option to run on raw uncorrected deltaT - used in condor too by default.

### 4. Step 3 — single-layer resolution per pixel

Reads the three `sigma_*.json` from step2 plus the run's threshold log
(`baseline`/`noise_width` per pixel). For each layer and pixel, picks the
best-statistics `(i, j, k)` triplet involving it and solves the per-layer
resolution from the three pairwise sigmas via the standard three-technique
decomposition (cyclic i->j->k->i, N/P = next/previous layer):

```
sigma_layer = sqrt( 0.5 * (sigma_TN^2 + sigma_PT^2 - sigma_NP^2) )
```

Only kept if all three pairwise sigmas have more than `min_n=200` entries (and
the result isn't complex, i.e. the quantity under the sqrt is non-negative).

Builds a per-pixel resolution heatmap plus a resolution histogram fit with a
single Gaussian (range 0-140 ps) over all valid pixels, giving the module's
mean +- std resolution. A hardcoded `MASK_I`/`MASK_J`/`MASK_K` exists to restrict
the fit to a sub-square of pixels, but it's currently disabled (set to `None`
right after being defined). If step2 also produced `_raw` json's, the same is
done for the uncorrected resolution and both are overlaid for comparison.

Output: `resolution_summary.json` (per-module mean/std resolution in ps, plus
bias voltage/temperature from the run log), heatmaps and histograms.

Example command for running locally (no condor submit file for this step):
```
python do_bootstrap_multiple_step3.py \
  --input-dir "test_results/step2/" \
  --output-dir "test_results/step3/" \
  --run-start 148847
```

### 5. Step 4 — combine runs

Takes several step3 `resolution_summary.json` (`--input-dirs`, e.g. different
bias-voltage or temperature points of the same modules) and plots each module's
mean resolution +- std vs bias voltage (always) and vs temperature (if present
in the run logs).

Example command for running locally (no condor submit file for this step):
```
python do_bootstrap_multiple_step4.py \
  --input-dirs test_results/step3/ test_results_run2/step3/ \
  --output-dir test_results/step4/
```
