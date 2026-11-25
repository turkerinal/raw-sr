# CFA-domain multi-frame drizzle — EXPERIMENTAL, not a production path

**Status: research prototype, rejected as a default.** Last revised
2026-11-20 after the independent audit `internal audit report`.
`raw_sr/core/cfa_drizzle.py` is not called by `superres.py`, the UI, the desktop
app or any acceptance default. Do not quote it as a resolution improvement.

Owner area: `raw_sr/core/cfa_drizzle.py`, `raw_sr/core/cfa_heldout.py`,
`scripts/exp_cfa_sr.py`, `scripts/sr_acceptance.py` (`--cfa-heldout`),
`tests/test_cfa_drizzle.py`, `tests/test_cfa_heldout.py`. `superres.py` is not
modified.

---

## 1. What was claimed, what is true

The first report (`internal audit report`) claimed **+2.25 dB** ground-truth
PSNR over the shipped demosaic→drizzle pipeline at 2x. That claim is
**withdrawn**: the synthetic harness had two defects.

1. **Wrong sign on the ground-truth transforms.** Frame *k* samples the scene at
   `(x + offs[k])`, and the cv2/drizzle convention is `dst(x) = src(M⁻¹x)`, so
   the ref→frame transform is `T(-offs[k])`. The harness passed `T(+offs[k])`,
   depositing every frame with a **2× shift error** (up to 8 LR px). The core's
   own `estimate_homography` was always right; only the synthetic `aligns` were
   wrong.  Fixed: `synthetic_burst()`.
2. **Red/blue swap in the demosaic helper.** `_demosaic` demosaiced the BGGR
   mosaic with `COLOR_BayerBG2BGR_EA` (then swapped BGR→RGB), which puts **B in
   the R slot**. OpenCV's Bayer names are additionally offset by one pixel for
   a mosaic whose (0,0) is the first letter: a BGGR mosaic must be demosaiced
   as `COLOR_BayerRG2RGB_EA`.  Both the "pipeline" and the "stack" columns were
   rebuilding the wrong colour image, which cost them ~10 dB.  The CFA prototype
   never demosaics, so the *difference* looked like a CFA win.  Fixed in
   `scripts/exp_cfa_sr.py` and `tests/test_cfa_drizzle.py`, pinned by
   `test_demosaic_test_helper_preserves_channel_identity` (a red-only BGGR
   mosaic must demosaic to red).

Independent verification of the swap (constant-colour mosaic, centre pixel):

| code | value at centre (R,G,B) |
|---|---|
| `COLOR_BayerBG2BGR_EA` (as shipped) | 0.2, 0.5, **0.8** |
| `COLOR_BayerRG2BGR_EA` (correct) | **0.8**, 0.5, 0.2 |

The rest of B5 that does **not** depend on the harness — the real-burst
diagnostics (noise floor, demosaic ambiguity, DCB-space metric lock, the
>Nyquist band being noise-dominated) — is unaffected: neither bug touches a DNG.

---

## 2. Corrected ground-truth result

`scripts/exp_cfa_sr.py --synthetic` with the fixed harness; 9 frames, known
sub-pixel shifts, Bayer CFA, shot+read noise, 10-bit, lens σ = 0.55 LR px,
per-LR-pixel integration.  Mean of seeds 3/4/5, HR PSNR (dB) against the
sensor-integrated HR scene (the signal the frames actually measure):

| ratio | LR stack | demosaic pipeline | cfa pf=0.5 | cfa pf=1.0 |
|---|---|---|---|---|
| 2x | 32.97 | 34.55 | 34.87 | 36.65 |
| 3x | 33.10 | 35.01 | 34.43 | 36.34 |
| 4x | 33.15 | 35.15 | 34.31 | 36.00 |

Paired difference versus the pipeline (same seeds, same target, same mask):

| ratio | cfa pf=0.5 | cfa pf=1.0 |
|---|---|---|
| 2x | **+0.32** | +2.10 |
| 3x | **−0.57** | +1.33 |
| 4x | **−0.84** | +0.84 |

The auditor's independent harness (different target: un-integrated lens image,
5 seeds) measures **−1.18 ± 0.28 dB at 2x**, **−1.48 at 3x**, **−1.19 at 4x**
for `pixfrac=0.5`.  Taken together: **at the module default the CFA path is at
parity at best and 1-1.5 dB worse in the auditor's harness**.  The honest
summary of the 2x number is therefore **−1.2 … +0.3 dB**, not a win.

### 2.1 Why the `pixfrac=1.0` gain is not a resolution gain

With a footprint as wide as one LR pixel the CFA path wins *total* PSNR at every
ratio, but the extra energy is not scene detail:

| ratio | HP-PSNR pipeline / cfa05 / cfa10 | chroma-consistency PSNR pipeline / cfa05 / cfa10 |
|---|---|---|
| 2x | 39.63 / 36.81 / 39.99 | 44.55 / 38.11 / 41.60 |
| 3x | 41.54 / 37.33 / 40.71 | 43.63 / 38.56 / 41.91 |
| 4x | 42.19 / 38.38 / 41.14 | 43.53 / 39.51 / 42.36 |

The CFA path is **3-6 dB worse on chroma consistency at every ratio and
footprint**: its per-channel deposit reconstructs R/B independently, so colour
is not tied to green and a false-colour zipper appears along edges.  The audit
reproduced this visually on the real burst (`audit12_real_panel2.png`: diagonal
rainbow zipper on a roof/wall edge; chroma high-pass energy 4.5× the pipeline's,
chroma-luma correlation 0.17-0.43 versus 0.02-0.26) and objectively in the
synthetic HP-PSNR (pipeline 34.9 dB vs CFA 28.6 dB at 2x).

Consequence: `hf_gain_db` and Laplacian sharpness must **not** be read as
resolution for this path.

---

## 3. Real 9-DNG burst, ROI 512 (truth-free sanity only)

With the level calibration of §4 applied (2x; 3x/4x figures were taken before
it and are indicative only):

| ratio | demosaic HF / sharpness | cfa HF / sharpness |
|---|---|---|
| 2x | +10.5 dB / 2.8× | +13.9 dB / 4.4× |

That difference is **partly false colour**, not detail (§2.1).  The 4x
demosaic "sharpness" of ~455× is acutance/noise; the CFA path reports ~41× for
the same band energy, which is the one property that checked out: a per-channel
deposit adds less pixel-level noise for the same HF magnitude.

---

## 4. Level calibration (new, required for any comparison)

`load_cfa_frame` normalises by per-site black/white and applies
`camera_whitebalance / green`, but LibRaw's colour scaling is not exactly that.
Measured on the project DNG at the same sites: demosaiced/mosaic level ratio is
**R 0.965, G 0.973, B 1.057** (the audit measured a 1.0265 green ratio with a
slightly different ROI/normalisation).  A CFA-vs-LibRaw comparison that ignores
this reads a constant per-channel bias.

`calibrate_cfa_levels(frame, reference_rgb)` returns least-squares per-channel
gains (`<ref·mosaic>/<mosaic²>`), and `apply_level_gains(frame, gains)` applies
them.  On the real burst the gains are R 0.9757, G 0.9769, B 1.0504 and
re-calibrating afterwards returns exactly 1.0 (pinned by
`test_level_calibration_recovers_a_known_per_channel_scale`).  Both
`scripts/exp_cfa_sr.py` and `scripts/sr_acceptance.py --cfa-heldout` now
calibrate before comparing.

---

## 5. Measurement-space held-out metric (`--cfa-heldout`) — experimental

Kept as opt-in and **off by default** (`raw_sr/core/cfa_heldout.py`), because
the DCB-space metric cannot rank deposition strategies: its truth is a
DCB-demosaiced image, so a demosaic-free output is charged for not reproducing
DCB's interpolation (B5 §2.4).

Design: the held-out frame's own Bayer sites are predicted from an HR estimate
(sensor-PSF forward model, then sampling at `ratio·W⁻¹p`, read only in the
site's channel) and compared with a coverage-normalised **per-channel CFA stack**
of the fused frames.  Both sides are scored on identical sites: interior margin
(24 LR px), prediction support (bilinear + 3σ PSF), and baseline coverage
≥ `0.5 · sum(w) · density_c` (density is 1/2 green, 1/4 R/B, so the rule is
parity-independent).  The held-out frame is excluded from the HR estimate and
from the baseline (structural: it is not an argument of the baseline).

Validation on the synthetic burst (frame 4 held out, 5 seeds × 3 ratios, CFA at
`pixfrac=1.0`): Spearman correlation between the CFA-space gain and GT HR PSNR
is **0.88**; the ordering `stack < pipeline < cfa < oracle` held in 14 of 15
runs (the exception is a 0.4 dB GT tie).  The metric **compresses magnitudes**
(GT 1.5-12 dB → 0.2-1.9 dB) because one noisy held-out frame dominates PSNR;
treat it as a ranking/regression instrument, not a magnitude.

### 5.1 It is level-calibration sensitive — do not use it as a gate

Measured on the real burst, 2x, held-out frames 0/2/4/6/8 at ROI 1024 **without**
the §4 calibration versus ROI 512 **with** it (pipeline / cfa, mean gain dB):

| run | pipeline | cfa |
|---|---|---|
| ROI 1024, uncalibrated | −3.55 | −0.04 |
| ROI 512, calibrated | −1.09…−1.62 | −0.03…−0.24 |
| ROI 512 smoke, 256 px, calibrated | +0.18 | −0.26 |

The *ranking* flips with the calibration and with the crop.  That is the whole
finding: the metric is monotone in a per-channel level offset of a few percent,
so until the loader's scale is exact (and the baseline band-limited like the
prediction) it cannot decide between methods.  It stays an off-by-default
diagnostic; the DCB-space metric remains the accepted number, and neither is
sensitive enough to rank deposition strategies on this burst.

Known limitations (why it is not a gate):

* the baseline is a direct bilinear splat of measurements while the prediction
  is PSF-band-limited, so the two have different effective bandwidths
  (`baseline_psf=True` is provided as an A/B; it moves the gain by <0.01 dB on
  the real burst);
* it depends on the level calibration of §4;
* its "oracle" comparison uses noise-free RGB frames, so the oracle's absolute
  gain is not a bound for a noisy pipeline (labelled `oracle_noiseless` in the
  synthetic output).

---

## 6. What still holds from B5 (not affected by the harness bugs)

* The pipeline deposits **demosaiced RGB** (LibRaw/DCB first, then drizzle), so
  2/3 of every deposited value at a pixel is interpolated and the CFA sampling
  pattern is gone before fusion.  That remains a true statement about the code.
  It is **not** shown to cost accuracy by this harness: with a correct
  demosaic, a DCB/EA demosaic + RGB drizzle + IBP already matches the
  noise-free pre-CFA oracle within 0.07 dB (audit §2.1), so the "3.5 dB of
  headroom" does not exist in this experiment.
* The real burst's above-Nyquist content is noise-dominated (single-frame PSD
  0.25-0.35 cyc/px 1.63 vs temporal-difference noise 2.27); the honest
  DCB-space held-out gain is ≈0 dB at 2x/3x/4x.
* The honest held-out metric is **representation-limited**, not purely
  noise-limited: flat-region σ² ≈ 8e-6 versus a baseline MSE ≈ 3.9e-5, and
  DCB-vs-AHD/VNG ambiguity alone is ~17 % of it.
* Registration residual (0.03-0.04 px) is far below the 0.5 LR-px deposit
  footprint and is not the limiting factor. The 0.59 px quoted for ROI 512
  elsewhere in the report is the same burst on a crop that is ~96 %
  structureless sky, where the whole-frame phase correlation has no peak in
  half the frame; see `docs/REGISTRATION_RESIDUAL.md`.

---

## 7. Verdict

* **Do not integrate** the CFA path as a default at any ratio.  It is at parity
  to 1.5 dB worse on ground-truth PSNR, and its extra HF is partly false colour.
* The module itself is sound where the audit could test it (coordinates, phase,
  ROI alignment, per-site calibration, per-channel normalisation, hole policy).
  Keep it as a research prototype behind the experimental label.
* Before any re-submission: add rotation/perspective and a 13-14 bit noise level
  to the synthetic burst (the current one is translation-only and ~3× noisier
  than the real burst), add a chroma-consistency term to the acceptance
  criteria, and re-check whether a green-guided chroma stage removes the zipper.
* Never present `hf_gain_db` or Laplacian sharpness for this path as resolution.

---

## 8. Reproducing

```bash
cd <worktree>
V=venv/bin/python
$V scripts/exp_cfa_sr.py --synthetic --ratios 2 3 4 --seeds 3 4 5 --lr 384
$V scripts/exp_cfa_sr.py --synthetic --ratios 2 3 4 --seeds 3 4 5 --lr 384 \
   --pixfrac 1.0 --synthetic-metric
export RAW_SR_TEST_DATA=test_data
flock /tmp/rawsr/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  $V scripts/exp_cfa_sr.py --real --ratio 2 --roi-size 512 --real-heldout 0,2,4
flock /tmp/rawsr/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  $V scripts/sr_acceptance.py --ratio 2 --roi-size 1024 --heldout-sweep \
     --cfa-heldout --out-suffix _B6cfa --out-dir output
$V -m pytest tests/test_cfa_drizzle.py tests/test_cfa_heldout.py -q
```
