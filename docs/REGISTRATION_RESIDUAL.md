# Registration residual vs ROI size

The acceptance run reports `align_residual_mean_px` = **0.593 px** at ROI 512 and
**0.034 px** at ROI 1024 on the same 9-frame real burst. This note explains the
gap. Short version: the *fit* is not eighteen times worse at 512. The residual
is a whole-frame phase correlation, and on this scene the ROI-512 crop is ~96 %
structureless sky/field, so the statistic reports the support quality of the
crop more than the misalignment. That is content, plus the already-documented
small-support ECC limitation — not a sign, mask or normalisation bug.

Reproduction (branch `fix/roi18`, `RAW_SR_TEST_DATA` set, default core options —
`registration_robust=False`, `ecc_scale=0.5`):

| ROI | `align_residual_mean_px` | `align_residual_max_px` | frames |
|---|---|---|---|
| 1024 at (1650,1240) | 0.0337 | 0.0646 | 9/9, ref 2 |
| 512 at (1650,1240) | 0.5929 | 0.8553 | 9/9, ref 2 |

Both numbers are stable to the last digit across reruns; the ROI-512 value is the
one the D4 ghost-gate threshold (`_GHOST_GATE_RESIDUAL_MAX = 2.0`) was calibrated
against. Probes: `/tmp/rawsr/b9_{final,robust,ramp,resp,content,synth,masked}.py`,
data `/tmp/rawsr/b9_pipeline.json`.

## 1. The residual *measurement* does not depend on ROI size

Known-shift synthetic at every size and at two ROI locations. Reference = real
frame 2, moved frame = the same crop resampled by an exact sub-pixel translation,
warped back with the exact inverse matrix, so the true residual is zero:

| crop | informative px | residual (px) | score |
|---|---|---|---|
| 256 TL | 0.71 | 0.0004 | 0.9961 |
| 256 centre | 0.09 | 0.0017 | 0.9999 |
| 384 TL | 0.18 | 0.0010 | 0.9998 |
| 384 centre | 0.13 | 0.0010 | 0.9999 |
| 512 TL | 0.04 | 0.0016 | 0.9999 |
| 512 centre | 0.12 | 0.0006 | 0.9999 |
| 768 TL | 0.07 | 0.0008 | 0.9999 |
| 1024 | 0.07 | 0.0004 | 0.9999 |

"Informative px" is the fraction of pixels with gradient magnitude above 10 % of
the crop maximum. Same for a constant-shift sweep `(0.35,0.35)`, `(3.1,2.2)`,
`(6.2,13.9)`, `(20,16)` px and for content-only crops (sky, distant field,
foliage, roof): all 0.000-0.004 px. So the residual implementation — border
zeroing, `valid > 0.99` mask, mean subtraction, the Hanning-windowed phase
correlation — is size- and location-independent **when the content determines the
shift**. There is nothing to fix in the metric's arithmetic.

## 2. The same transform scores 3x differently depending on *where* it is measured

The ROI-1024 homography (a good fit: see §3) applied unchanged inside five
512x512 windows of the same frames:

| 512 window | mean residual (px) |
|---|---|
| TL (0,0) | 0.167 |
| TR (0,512) | 0.160 |
| centre (256,256) | 0.057 |
| BL (512,0) | 0.057 |
| BR (512,512) | 0.083 |

One transform, no refit, 2.9x spread. The residual is therefore a property of the
ROI's content mix, not only of the registration.

## 3. Where the ROI-512 number comes from

`_alignment_residual_px` runs one more phase correlation over the whole warped
frame. Splitting that measurement into bands (frame 0 vs reference, ROI 512 TL,
fitted W):

| rows | phase-correlation response | residual (dx, dy) |
|---|---|---|
| 0-128 (sky) | **0.047** | (−5.7, −10.4) → 11.9 px (meaningless) |
| 128-256 (sky/field) | **0.049** | (−4.2, −9.4) → 10.3 px (meaningless) |
| 256-384 (tree line, roof) | 1.065 | (−1.17, +0.11) → 1.17 px |
| 384-512 (roof/ground) | 1.080 | (−0.18, −0.08) → 0.19 px |

A response near zero means there is no peak: the top half of the crop is a smooth
brightness gradient with no structure, so its "residual" is noise. A full-frame
response of 0.86-0.91 is a lock on the lower half, but the uninformative half
still biases the single global estimate towards ~0.7-0.85 px (and it is almost
purely an x-offset, which is the axis a horizontal sky gradient cannot
constrain). The gradient mask used by `_alignment_score` covers **6.0 %** of this
crop (only **4.2 %** of its pixels carry a gradient above 10 % of the crop
maximum), so the fit and the score are effectively computed on a few percent of
the pixels.

The residual is still directionally right: in the ROI-1024 fit the same textured
rows measure 0.05/0.05 px, while the ROI-512 fit leaves 1.17 px in the 256-384
band. So the 512 fit really is worse *there*:

| ROI | mean \|aligned − ref\| (linear) | max corner perspective |
|---|---|---|
| 1024 | 0.00319 | 3.2 px |
| 512 TL | 0.00405 | 8.7 px |
| 512 centre | 0.00308 | 0.5 px |
| 384 TL | 0.00401 | 6.3 px |
| 256 TL | 0.00417 | 23.2 px |

Actual pixel disagreement rises only 1.3x from 1024 to 512 TL (and is at the
sensor-noise floor: for two independent noise realisations the expected
`mean|diff|` is 1.13 σ), while the reported residual rises 18x. The mechanism for
the real part is the one the `estimate_homography` / `_corner_perspective_px`
docstrings already describe for perspective terms: with 96 % of the support structureless, the homography's
perspective coefficients (8.7 px corner excursion at 512 TL, 23 px at 256 TL) are
determined by a few percent of the pixels and extrapolate across the sky — the
failure mode `_corner_perspective_px` and the `estimate_homography` docstring
describe, and what the affine fallback (`prefer_affine`, used only when
`registration_robust=True`) exists to prevent. A 512 crop centred on the building
(11.6 % informative) selects an affine fit (0.46 px corner excursion) and reports
0.039 px — the same as ROI 1024.

## 4. Consequence and what was *not* changed

* ROI 512 vs 1024 is **not** evidence of a registration regression; it is the
  statistic responding to a low-information crop. Read `align_residual_*`
  together with the informative-pixel fraction (or with `align_scores`) before
  treating it as a quality signal.
* The residual cannot *rank* fits against true pixel error: at ROI 512 TL the
  default path reports 0.593 px with `mean|diff|` 0.00405, while
  `registration_robust=True` reports 0.357 px with `mean|diff|` 0.00437 — the
  lower residual belongs to the slightly *worse* alignment.
* The D4 threshold (`_GHOST_GATE_RESIDUAL_MAX = 2.0` px) is unaffected in
  practice: the real burst stays OFF at both ROIs, and the gate is a
  gross-failure detector (one 20 px outlier among 8 frames), not a sub-pixel
  quality gate. Its margin at ROI 512 is 3.4x, and §3 explains why that margin
  is not a defect of the burst.
* No registration code was changed. Masking the residual to structured pixels
  would make the number look like 0.02-0.25 px, but it would also hide the real
  1.17 px band error above, and it would move the D4 calibration that another
  branch owns. The honest fix, if one is wanted later, is at the *fit* level
  (reject perspective terms when the informative fraction is low), and it must
  be measured on held-out gain, not on this statistic.

## 5. Known-shift check at the failing size

To be explicit that a good fit is recognisable at 512 TL: applying the ROI-1024
homography to the ROI-512 crop gives 0.17 px mean there (0.04-0.39 per frame),
versus 0.59 px mean for the independent 512 fit. The crop size is not the
limitation; the amount of structure in the crop is.
