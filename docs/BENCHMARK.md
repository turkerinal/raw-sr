# CuPy/CUDA GPU Backend Benchmark (`feat/gpu`, 2026-11-24)

Measured on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM, CUDA 13.5) with CuPy 14.2.0
(`cupy-cuda13x[ctk]`), median of 3 warm runs per stage. Data stays resident in VRAM
across each stage (upload once, download once).

| Config | Drizzle CPU/GPU | IBP CPU/GPU | Enhance WP2 CPU/GPU | Enhance Adaptive CPU/GPU | SR e2e CPU/GPU | e2e PSNR | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| own9 2× (9f, 2048²) | 3.79 / 0.10 s (**39.5×**) | 0.84 / 0.31 s (**2.8×**) | 0.95 / 0.30 s (**3.2×**) | 1.00 / 0.68 s (**1.5×**) | 8.94 / 5.77 s | 144.7 dB | 1320 MB |
| own9 4× (9f, 4096²) | 20.77 / 0.50 s (**41.7×**) | 6.04 / 0.73 s (**8.2×**) | 16.23 / 1.42 s (**11.5×**) | 5.65 / 3.12 s (**1.8×**) | 34.11 / 18.63 s | 141.5 dB | 2332 MB |
| HDR+ 0919 2× (8f, 2048²) | 3.19 / 0.09 s (**34.7×**) | 0.76 / 0.27 s (**2.8×**) | 0.89 / 0.29 s (**3.1×**) | 0.99 / 0.66 s (**1.5×**) | 6.15 / 5.86 s | 132.6 dB | 1322 MB |
| Pixel 4a 20f 2× (20f, 2048²) | 7.15 / 0.17 s (**41.6×**) | 1.67 / 0.64 s (**2.6×**) | 0.91 / 0.30 s (**3.0×**) | 1.00 / 0.66 s (**1.5×**) | 15.36 / 13.22 s | 141.8 dB | 1322 MB |

See `docs/GPU_BACKEND.md` for architecture, VRAM guards, and stages kept on CPU.

---

# Before/after benchmark: `826bc65` vs `ec7cb6b` (`integ/C1`, 2026-11-22)

Comparison of pre-`integ/C1` `master` (`826bc65`) and current `master`
(`ec7cb6b`: VST edge-preserving MAP-IBP prox ratio-scaled with `alpha=0.75`,
`drizzle_low_cov_thresh=0.0`, in-place VST + freed IBP temporaries).

- **Real burst (9-DNG handheld, ROI `(1650, 1240, 1024, 1024)`, 2×/3×/4×,
  all 9 frames held out):** numbers from `docs/agent-reports/2026-11-22-C5-final.md`
  (`_C5final` vs `826bc65`). Note: `scripts/compare_versions.py` could not refresh
  `output/compare/` in the C6 sandboxed session because `test_data` symlinks
  outside the workspace (`test_data`); the
  `output/compare/` panels remain from the 2026-11-20 run below.
- **Synthetic GT (`scripts/exp_cfa_sr.py --synthetic --ratios 2 3 4 --seeds 3 4 5`):**
  measured fresh in C6 on `826bc65` (worktree `/tmp/rawsr/wt-old`, removed) vs
  `ec7cb6b`.
- **Root `pytest -q`:** **504 passed, 0 failed** in 720.30 s (`C5-final.md`).

| metric | 2× `826bc65` | 2× `ec7cb6b` | 3× `826bc65` | 3× `ec7cb6b` | 4× `826bc65` | 4× `ec7cb6b` |
|---|---|---|---|---|---|---|
| held-out gain, min (dB, P1) | −0.006 (FAIL) | **+0.0266 (PASS)** | −0.046 (FAIL) | **+0.0167 (PASS)** | −0.051 (FAIL) | **+0.0045 (PASS)** |
| held-out gain, mean (dB) | +0.013 | **+0.0519** | −0.014 | **+0.0460** | −0.022 | **+0.0380** |
| HF gain vs single (dB) | 5.091 | 4.934 (−0.157) | 14.737 | 14.574 (−0.163) | 18.581 | 18.418 (−0.163) |
| S2 min HF vs any frame (dB) | +4.87 | **+4.65 (PASS)** | +14.55 | **+14.33 (PASS)** | +18.34 | **+18.12 (PASS)** |
| sharpness gain (×) | 2.34 | 2.09 | 13.88 | 11.86 | 32.41 | 27.65 |
| SR wall (s, acceptance) | 6.67 | 6.96 | 9.65 | 10.16 | 13.32 | 19.18 (standalone 13.94) |
| peak RSS (MiB, SR run) | 1485 | **1093** | 1543 | 1575 | 2294 | **2281** |
| synthetic GT PSNR (dB, seeds 3/4/5) | 34.60 | **34.72 (+0.12)** | 35.15 | **35.52 (+0.37)** | 35.39 | **35.85 (+0.46)** |
| synthetic GT HP-PSNR (dB, seeds 3/4/5) | 39.80 | **40.25 (+0.45)** | 41.65 | **43.79 (+2.14)** | 42.16 | **44.74 (+2.58)** |
| synthetic GT SSIM (seeds 3/4/5) | 0.923 | 0.920 | 0.926 | 0.925 | 0.930 | **0.936** |

All 27/27 held-out frames are strictly positive (**P1 PASS** at 2×/3×/4×),
`hf_gain_db_vs_single` stays within the `<= 0.3 dB` budget at every ratio
(−0.157 / −0.163 / −0.163 dB), and synthetic GT PSNR and HP-PSNR improve at
every ratio (+0.12/+0.37/+0.46 dB PSNR, +0.45/+2.14/+2.58 dB HP-PSNR).

---

# Historical benchmark: `master` (`e6b5a3f`) vs `integ/B3` (`ee44c4b`, 2026-11-20)

Reproducible, side-by-side measurement of the original project state (`master`)
and the integration branch that carries every fix (`integ/B3`). **No source
file under `raw_sr/` is modified by this benchmark**: both versions run their
own core, and all comparison arithmetic lives in
`scripts/compare_versions.py`, which is copied unchanged into both checkouts.

- baseline: `master` of `.`
  (commit `e6b5a3f`; worktree `/tmp/rawsr/wt-master2`, removed after the run)
- new: `integ/B3` at **`ee44c4b`** — the numbers below are the final measurement of
  2026-11-20 against that tip (detached worktree `/tmp/rawsr/wt-final2`, removed after
  the run). `26c2cab` (the next tip) adds only docstrings and
  `docs/REGISTRATION_RESIDUAL.md`, so the table describes it too. Earlier passes of
  this report were measured at `31eb1ed` and `213b784`; those timing/RSS columns are
  superseded, and the image columns moved because the HR grid became pixel-centre
  aligned (see "Why the image columns moved" below).
- data: the same 9-DNG handheld burst, ROI `(1650, 1240, 1024, 1024)`, 2x/3x/4x
- raw numbers: `output/compare/comparison.json` (dumps in `/tmp/rawsr/cmp2`);
  full report and clean-environment check `internal audit report`
- root `pytest -q` on that worktree: **450 passed, 1 xfailed in 655.08 s**, 0 failed,
  0 skipped

## Method

**Identical evaluator for both versions.** The honest held-out metric is the
integration branch's coverage-normalised, interior-masked evaluator, copied
verbatim into `scripts/compare_versions.py` and applied to *both* versions' HR
outputs, so the metric cannot favour either side:

- for each held-out frame `h` (all 9 frames are held out, no cherry-picking):
  reconstruct the burst without `h` with that version's own core
  (`blur_threshold_ratio=0.0`, i.e. every frame fused, the evaluator's
  documented default), then predict `h`'s **measured LR samples** from the HR
  estimate through the sensor forward model (Gaussian per-LR-pixel integration
  window, sigma = 0.289·ratio HR px, bilinear sampling);
- the baseline is the coverage-normalised (per-pixel weighted mean) denoised LR
  stack of *exactly the frames the run fused*, warped into the held-out frame
  with a fresh ref→held-out homography (so the metric never grades the run's own
  registration; it cancels in the gain);
- both sides are scored only where **both** are determined: the prediction's
  geometric support, a fixed 24 LR px interior margin (1.7x the worst measured
  shift), and baseline coverage ≥ 0.999·Σw — 90.8 % of the ROI;
- `heldout_gain_db` = PSNR(SR prediction) − PSNR(stack) on that mask;
  `heldout_gain_db_legacy` is the pre-2026-11 whole-frame number, kept only so
  the old (rim-inflated) value can be compared field by field.

**Grid alignment.** `to_drizzle_matrix` on the branch adds the pixel-centre offset
`(ratio-1)/2`, master's does not. The evaluator's LR-sampling mask therefore adds
the same offset for the prediction, and each version's image metrics are computed
against a bicubic baseline sampled on *that version's own* lattice. This is the
only way both sides measure the same thing; it also means per-pixel differences
between the two HR images are lattice-dominated (see "Why the image columns
moved").

**Image metrics.** `hf_gain_db_vs_single` and `sharpness_gain` are recomputed by
the collector from the two HR TIFFs with one code path (band energy above the
single-frame Nyquist in the display-transformed view; Laplacian-variance ratio),
both against a **bicubic-upsampled single frame** — the `scripts/sr_acceptance.py`
convention. `hf_gain_db` (vs the denoised stack) is the number each version
emits itself and is shown only as a cross-check: master's own stack baseline is
global-weight normalised (dark rim), the branch's is coverage-normalised.

**Per-version core metrics** (`align_residual_*`, `ghost_energy`,
`robust_active`/`robust_reason`) are copied verbatim from that version's
`SRResult.metrics`; they describe the run, not a re-derivation.

**Resources.** Wall clock from `perf_counter` around `SRResult`'s phases: the
pre-drizzle bucket (sharpness weights + reference selection + registration +
auto ghost gate), the drizzle deposit, and IBP; `seconds_sr_wall` is the whole
`run()`. Peak RSS is sampled every 50 ms over the process tree (parent +
spawned registration workers) during `run()` only; held-out reconstructions run
afterwards and are not counted.

**Images.** `compare_<r>x_crop.png` is the protocol panel: the same 256 HR px
region for all three panels (in that ratio's HR grid), left = single frame
nearest-neighbour upscaled to HR size, middle = master SR, right = new SR. The
crop centre is the same *fraction* of the frame (16 % / 70 %) at every ratio, so
the panel shows the same scene region — the roof-ridge and eave tile texture. `compare_<r>x_full.png` is the whole
frame at 460 px per panel. `diff_<r>x_amp4.png` is new − master at literally
`128 + 4·(new − master)` per channel (mid grey = no change, red = new brighter,
blue = new darker; one display level of difference = 32 grey levels), and
`diff_<r>x_amp4_stretch.png` is the same signed field additionally stretched so
its own 99.5th percentile is full scale (per-image scale; for reading structure
that sits below one display level).

## Results

All rows: 9/9 frames kept, reference index 2, gate **OFF** in both versions
(`robust_reason` = "static burst — plain fusion").

| metric | 2x master | 2x new | 3x master | 3x new | 4x master | 4x new |
|---|---|---|---|---|---|---|
| held-out gain, mean (dB) | −0.003 | **+0.013** | −0.036 | **−0.014** | −0.047 | **−0.022** |
| held-out gain, min (dB) | −0.061 | **−0.006** | −0.077 | **−0.046** | −0.077 | **−0.051** |
| held-out gain, legacy mean (dB) | +2.879 | +3.693 | +2.356 | +3.603 | +2.133 | +3.564 |
| HF gain vs single (dB) | 5.190 | 5.091 | 14.799 | 14.737 | 18.916 | 18.581 |
| HF gain vs denoised stack (dB)* | 8.304 | 8.204 | 15.328 | 15.266 | 19.262 | 18.927 |
| sharpness gain (×) | 3.25 | 2.34 | 16.51 | 13.88 | 64.61 | 32.41 |
| registration residual, mean (px) | 0.0337 | 0.0337 | 0.0337 | 0.0337 | 0.0337 | 0.0337 |
| registration residual, max (px) | 0.0646 | 0.0646 | 0.0646 | 0.0646 | 0.0646 | 0.0646 |
| ghost energy | 4.985e−4 | 4.985e−4 | 4.972e−4 | 4.972e−4 | 4.968e−4 | 4.968e−4 |
| gate | OFF | OFF | OFF | OFF | OFF | OFF |
| registration wall (s) | 9.45 | **5.31** | 11.97 | **7.00** | 23.80 | **9.17** |
| pre-drizzle wall, incl. registration (s) | 11.80 | **5.73** | 16.90 | **7.91** | 38.96 | **10.72** |
| drizzle deposit (s) | 0.61 | 0.45 | 0.60 | 0.66 | 0.74 | 0.79 |
| IBP (s) | 1.38 | **0.48** | 2.93 | **1.09** | 10.87 | **1.82** |
| SR wall, whole run (s) | 13.80 | **6.67** | 20.44 | **9.65** | 50.57 | **13.32** |
| peak RSS (MiB) | 1627 | **1485** | 2987 | **1543** | 4912 | **2294** |
| crop noise σ, display (Immerkaer) | 0.771 | 0.712 | 0.764 | 0.762 | 0.594 | 0.616 |
| absolute output >Nyquist band energy, vs master | — | −0.10 dB | — | −0.06 dB | — | −0.33 dB |
| new−master RMS, linear (crop) | — | 2.96e−3 | — | 4.47e−3 | — | 5.49e−3 |
| new−master p99.5 abs, linear (crop) | — | 1.49e−2 | — | 2.13e−2 | — | 2.52e−2 |

\* version-emitted, different baseline normalisation (see Method); cross-check
only.

**What the deltas are.** The two version defaults differ, so part of every
difference is configuration, not only code: master runs `ibp_iterations=3` and
`drizzle_subsamples=3` at every ratio, the branch runs `ibp_iterations=1` and
ratio-aware `subsamples` 2/3/3 (both validated separately in
`docs/IBP_AND_HELDOUT.md` and the B1 pixfrac report: 3 IBP sweeps cost 0.02–0.03 dB
of held-out gain and add acutance, which is exactly the shape seen here). The
rest is the branch's own work: coverage-normalised held-out baseline, the
ghost-gate hardening, the memory pass (subsampled lattice allocation, streamed
float32 deposit, chunked medians, HR guard) and the sharpness-per-noise fusion
weights.

Read together: **honest held-out gain improves at all three ratios**
(−0.003 → +0.013, −0.036 → −0.014, −0.047 → −0.022 dB mean; the min improves by
0.026–0.055 dB everywhere, and 2x goes from −0.061 to −0.006), **the SR call is
2.1x / 2.1x / 3.8x faster** and **peak RSS drops 9 / 48 / 53 %**. Every bucket is
cheaper than master's: IBP 2.9x/2.7x/6.0x (1 vs 3 sweeps), registration
1.8x/1.7x/2.6x (affine warm-start; the residuals are unchanged to the 4th decimal
because both versions fit the same transform, so the residual column shows the
fit did not move, not that it is better), pre-drizzle including registration
2.1x/2.1x/3.6x.

**Why the image columns moved (2026-11-20, `fix/grid15`).** The HR grid is now
pixel-centre aligned, so master's and the branch's HR images are samples of
*different* lattices, offset by (r−1)/2 px (0.5 / 1.0 / 1.5). Consequences, all
verified rather than assumed:

- the amplified `diff_*` panels are now an edge map instead of flat grey, and the
  raw new−master RMS is 2.96e−3/4.47e−3/5.49e−3 linear instead of ~1e−3. That is
  the half-pixel shift, not a structural change: Fourier-shifting master onto the
  new lattice removes it (best offset exactly (r−1)/2 at every ratio) and the
  interior RMS returns to the previous round's size.
- the 4x sharpness ratio falls 64.6 → 32.4, and putting both images on one lattice
  confirms a real part of it: master (and the pre-grid `213b784`) keeps Laplacian
  variance 7.47e−5 after a Fourier shift where this tip has 5.53e−5; at 2x the two
  are equal (7.68 vs 7.66e−5). So the 4x output is genuinely smoother, and in
  exchange its prediction is better (held-out 4x −0.033 → −0.022 dB mean) while
  its absolute above-Nyquist band energy is unchanged (−0.33 dB). The extra
  acutance the old 4x grid had was partly aliasing.
- at 2x/3x the HF column reads slightly *negative* versus master (−0.10 / −0.06 dB)
  for the same reason plus the 1-vs-3 IBP trade; the absolute band energy in the
  HR output is flat, which is the honest way to read "no resolution lost".

The legacy held-out column is shown only to make the point that its +2.1…+3.7 dB
is an artefact: it is the branch's *own* fix that removes it, and it does so on
master's images too.

## Visual inspection (the author looked at the PNGs)

Panel = single frame NN-upscaled | master | new; crops from
`output/compare/compare_*_crop.png`, plus 3x nearest zooms of the same regions
(`/tmp/rawsr/zoom_f3_*x.png`), the amplified differences, and a master
reprojected onto the new lattice (`/tmp/rawsr/aligned_cmp_*x.png`) because of the
half-pixel grid change.

- **Sharpness.** At 2x the single-frame panel is blockier than both SR panels
  (expected: NN upscale); master's SR panel is the crisper of the two, new's
  slightly softer on the tile ridges and the roof/eave line, edges still clean and
  not blurred into each other. At 3x the same, and the ~1.5 HR-px ridge pattern is
  resolved in both. At 4x new is **visibly softer** on ridge/tile micro-contrast
  even after reprojecting master onto the new lattice — consistent with
  sharpness 64.6 → 32.4, and with the analysis above (part lattice, part a real
  reduction of aliasing-driven acutance, with a better prediction in exchange).
  No structure is lost: absolute above-Nyquist energy is flat (−0.33 dB).
- **Ringing / halos.** None visible in either version at any ratio. Both keep the
  roof/white-wall edge monotone (checked at 3x nearest zoom); no light or dark
  fringe along the eaves, no double edge on the railing, no dark rim around the
  over-exposed wall. A 2-D local-extremum overshoot test on the crop returned
  exactly 0 for both at all three ratios, which agrees with the eye. The new
  pixel-centre grid also removed the half-pixel rim the old 3x output carried.
- **Ghosting.** No ghost/trail in either: the foliage and roof textures are
  single-valued, no displaced copy of the railing at 2x/3x/4x. Both versions report
  the same `ghost_energy` (4.985/4.972/4.968e−4) and gate OFF — the expected
  agreement on a static burst. It also shows the new defaults (`min_align_score`
  0.1, the content/size-aware align gate, D4) do not fire spuriously here: 9/9
  frames kept, `mean_align_score` 0.9979 in both.
- **Colour artifacts.** No colour fringing at edges, no magenta/green speckle on
  the tile pattern, no hue shift between master and new. Flat-region chroma noise
  is equal to marginally lower in new (Lab a/b std 1.87/1.33 → 1.85/1.32 at 2x,
  1.99/1.52 → 1.99/1.52 at 3x, 2.01/1.53 → 2.01/1.54 at 4x).
- **Noise.** Grass, wall and sky textures are equally clean; crop Immerkaer σ is
  0.135 → 0.122 display levels at 2x (new slightly cleaner in that patch — the
  `crop_noise_sigma` field reads 0.771 vs 0.712 because the crop box lands half a
  pixel over), 0.130 → 0.129 at 3x, 0.093 → 0.096 at 4x. Nothing that reads as
  amplified sensor noise, and no plastic/smeared look.
- **The ×4 difference image is no longer flat grey.** It is a full-strength copy of
  every ridge and the eave line (RMS 2.85/4.46/5.30 display levels), which is what a
  half-pixel lattice shift looks like: *the whole image moved*, so every edge
  shows. Confirmed by reprojecting master onto the new grid — the structure
  disappears and only a faint per-pixel gain/phase residue plus noise remains,
  with no displaced edge and no large-area level shift. So read the panels for
  quality, and read `diff_*` as "lattice shift + a small change".

## Reproducing

```bash
# master worktree (temporary), and the branch tip under test
git -C . worktree add --detach /tmp/rawsr/wt-master master
git -C . worktree add --detach /tmp/rawsr/wt-final2 ee44c4b
cp scripts/compare_versions.py /tmp/rawsr/wt-master/scripts/
cp scripts/compare_versions.py /tmp/rawsr/wt-final2/scripts/

export RAW_SR_TEST_DATA=test_data

# one heavy job at a time
for r in 2 3 4; do
  timeout 3000 flock /tmp/rawsr/heavy.lock systemd-run --user --scope -q \
    -p MemoryMax=6G -p MemorySwapMax=0 bash -lc \
    "cd /tmp/rawsr/wt-master2 && RAW_SR_TEST_DATA=$RAW_SR_TEST_DATA \
     venv/bin/python \
     scripts/compare_versions.py run --label master --ratio $r --dump-dir /tmp/rawsr/cmp2"
  timeout 3000 flock /tmp/rawsr/heavy.lock systemd-run --user --scope -q \
    -p MemoryMax=6G -p MemorySwapMax=0 bash -lc \
    "cd /tmp/rawsr/wt-final2 && RAW_SR_TEST_DATA=$RAW_SR_TEST_DATA \
     venv/bin/python \
     scripts/compare_versions.py run --label new --ratio $r --dump-dir /tmp/rawsr/cmp2"
done

# images + comparison.json (re-reads the dumps; no reconstruction)
venv/bin/python \
    scripts/compare_versions.py collect --dump-dir /tmp/rawsr/cmp2 \
    --out-dir output/compare --ratios 2 3 4 --crop-frac 0.16,0.70
```

`run` uses `--heldout-indices ""` (default) = every frame; a single index makes
a quick smoke run. Total cost of the full sweep: ~2 min for master (4x peaks at
≈4.8 GiB, so 6G is the ceiling to use) and ~1 min for the branch; every step is
deterministic apart from wall clock and RSS. On the same machine the root suite is
`pytest -q` = 450 passed, 1 xfailed in 655 s.

Note for 4x: the branch's N-aware memory guard refuses an HR 4096² output when
`MemAvailable` is low (it estimated 2.5 GiB against a 15.2 Mpx limit while another
job was running). That is the guard working; free memory or raise
`memory_budget_bytes` before rerunning.

## Caveats

- Held-out gain ≈0 dB is the honest answer for **this** burst (it has no
  genuine above-single-frame-Nyquist content the 9-frame stack cannot already
  reach); the value of the table is the *delta* between versions on one
  identical instrument, not the absolute number.
- Timing is single-run on a shared machine that also ran a review job under the
  same lock; master's 4x wall read 40.7 s in the previous round and 50.6 s here
  (the `srspeed10` median/FFT work landed in between), so treat the wall columns
  as ±20 % and the *ratios* as the signal. The largest structural differences are
  IBP (1 vs 3 sweeps) and the registration warm-start; residuals are essentially identical between versions
  (0.0337 px mean), so the residual column does not measure registration
  *quality*, only that it did not move. The residual is also ROI-composition
  dependent: on the same burst it reads 0.593 px at ROI 512 vs 0.034 px at ROI
  1024 because the 512 crop is ~96 % structureless sky; see
  `docs/REGISTRATION_RESIDUAL.md`.
- Peak RSS at master/4x (≈4.8 GiB) is the memory problem the branch's HR guard
  and streaming deposit remove; the branch's 2.2 GiB is why `MemoryMax=5G` no
  longer kills a 4x run.
- **The two versions no longer share an HR lattice** (pixel-centre vs corner;
  offset (r−1)/2 px). Per-pixel differences, the amplified `diff_*` panels and
  the sharpness ratio are all sensitive to that; the held-out gain, the residuals
  and the RSS are not. See "Why the image columns moved".
- The visual notes are a single-scene judgement on one burst; they are evidence
  about *this* content (roof tiles, eaves, foliage), not a general image-quality
  ranking.
