# WP7: Joint Registration Refinement Against the HR Estimate

Opt-in via `SuperResolution(reg_refine=N)` (`N >= 1`, default `0`).

## 1. Implementation (`raw_sr/core/reg_refine.py`)

Alternating outer loop around fusion in `SuperResolution._run_inner` (up to `reg_refine` iterations, stopping early when max corner displacement across all frames is `< 0.005` LR px):

1. **Forward-model Gauss-Newton / Lucas-Kanade (`refine_burst_registration`):**
   - Blurs the current HR estimate by the sensor integration PSF (`forward_sigma = 0.289 * ratio` HR px).
   - When all frames carry calibrated Bayer mosaics (`RawFrame.mosaic` and `.pattern`), fits a global 3×3 + offset linear color map from blurred HR to the reference frame's mosaic (identity on synthetic/`cfa_kr`; bridges LibRaw sRGB to camera-RGB on real `demosaic_drizzle`), then refines directly against raw Bayer CFA samples. Falls back to the green channel otherwise.
   - Solves jointly for the 8 normalized homography parameters (`ref -> mov`) and per-frame photometric gain/offset `(a, b)` via Huber-weighted IRLS Gauss-Newton with MAD scale estimation and Levenberg-Marquardt damping on affine/perspective terms so flat corners do not drift.
   - Accepts an update only when the forward-model RMS residual does not worsen and total corner shift stays within `3.0` LR px.
2. **Photometric normalization & re-fusion:** Applies refined `(a, b)` to shallow-copied frames and re-fuses (`demosaic_drizzle` or `cfa_kr`) with the refined homographies.
3. **Default path:** When `reg_refine=0` (default), the code path is unchanged and bit-identical to master (`max abs diff = 0.0` on `own9` 2×).

---

## 2. Synthetic Evaluation (`scripts/eval_wp7_synth.py`, `lr=256`, `seed=7`)

Comparison of `reg_refine=0` (`rr0`) vs `reg_refine=2` (`rr2`) for both `demosaic_drizzle` and `cfa_kr`:

| Case | Fusion | PSNR `rr0 -> rr2` (dB) | MTF50 `rr0 -> rr2` | Reg Err mean/max `rr0 -> rr2` (LR px) | Time `rr0 -> rr2` (s) |
|---|---|---:|---:|---:|---:|
| 2× n=9 | `demosaic_drizzle` | 30.331 -> 30.353 (+0.022) | 0.3617 -> 0.3617 | 0.0221/0.0458 -> 0.1184/0.3028 | 3.52 -> 5.45 |
| 2× n=9 | `cfa_kr` | 33.956 -> 33.913 (-0.043) | 0.3784 -> 0.3789 | 0.0221/0.0458 -> 0.0708/0.1532 | 1.62 -> 6.15 |
| 2× n=20 | `demosaic_drizzle` | 30.439 -> 30.481 (+0.042) | 0.3566 -> 0.3575 | 0.0273/0.1095 -> 0.1594/0.3553 | 4.10 -> 11.73 |
| 2× n=20 | `cfa_kr` | 34.113 -> 34.121 (+0.008) | 0.3729 -> 0.3728 | 0.0273/0.1095 -> 0.0631/0.1804 | 5.49 -> 13.00 |
| 3× n=9 | `demosaic_drizzle` | 30.534 -> 30.511 (-0.023) | 0.3760 -> 0.3725 | 0.0281/0.1130 -> 0.1155/0.2827 | 6.68 -> 11.29 |
| 3× n=9 | `cfa_kr` | 34.545 -> 34.551 (+0.006) | 0.3872 -> 0.3866 | 0.0281/0.1130 -> 0.0654/0.1350 | 3.44 -> 9.49 |
| 3× n=20 | `demosaic_drizzle` | 30.841 -> 30.856 (+0.015) | 0.3785 -> 0.3799 | 0.0292/0.1086 -> 0.1541/0.3178 | 9.52 -> 23.74 |
| 3× n=20 | `cfa_kr` | 34.798 -> 34.820 (+0.022) | 0.3870 -> 0.3863 | 0.0292/0.1086 -> 0.0495/0.1037 | 9.40 -> 20.99 |

**Key finding on synthetic:** `raw_sr`'s initial ECC homography (`transforms`) already achieves `0.022–0.029` LR px mean corner error on `synthesize_target_burst`. (The `+2.9 dB` gap in `docs/SIRIL_COMPARISON.md` came from injecting phase-correlation `shifts` rather than `transforms` into Siril.) On synthetic bursts with a genuine `>0.3` LR px homography + exposure perturbation (`tests/test_wp7_reg_refine.py`), `refine_burst_registration` recovers the transform to `< 0.05` LR px and exposure gain/offset to `< 0.01 / 0.005`.

---

## 3. Real Bursts, 2×, 1024×1024 ROI (`scripts/eval_wp7_real.py`)

5 evenly spaced held-out frames per burst. Both DCB-space and CFA-space held-out gains (dB, mean; min in brackets for DCB) are reported:

| Burst (frames) | DCB `master` | DCB `reg_refine=2` | CFA `m_def -> r_def` | CFA `m_cfa -> r_cfa` | Time `m -> rr2` (s) | Max $\Delta$ (LR px) |
|---|---:|---:|---:|---:|---:|---:|
| `own9` (9) | +0.057 (+0.029) | +0.058 (+0.029) | -2.847 -> -2.844 | -0.035 -> -0.034 | 9.58 -> 45.12 | 0.0907 |
| `HDR+ 0919` (8) | +1.386 (+0.932) | +1.398 (+0.934) | -3.096 -> -3.095 | +0.952 -> +0.951 | 8.56 -> 27.44 | 0.0168 |
| `HDR+ 0382` (8) | +0.748 (+0.670) | +0.748 (+0.667) | -7.142 -> -7.141 | +0.248 -> +0.256 | 8.42 -> 30.83 | 0.0112 |
| `Samsung 14` (14) | +1.021 (+0.619) | +0.965 (+0.696) | -8.258 -> -8.275 | -5.271 -> -5.295 | 14.61 -> 107.44 | 0.1718 |
| `Pixel 4a 20` (20) | +1.581 (+0.645) | +1.580 (+0.642) | +0.788 -> +0.781 | +1.043 -> +1.035 | 20.81 -> 110.06 | 0.0164 |

Peak RSS stayed within the 6 GiB scope budget across all bursts (~1.8–2.1 GiB); runtime overhead is `3.2×–7.4×` on CPU.

---

## 4. Visual Panels (`docs/wp7/*.png`, pixel-aligned 2×2, all `< 1.2 MB`)

- [`own9_2x.png`](wp7/own9_2x.png) (347 KB): `master` and `reg_refine=2` are visually indistinguishable; `|diff| x10` is nearly black.
- [`hdrplus_0919_2x.png`](wp7/hdrplus_0919_2x.png) (358 KB): Both eliminate single-frame diagonal moiré and resolve clean louvers; visually identical.
- [`hdrplus_0382_2x.png`](wp7/hdrplus_0382_2x.png) (450 KB): Glassware highlights and amber stem are crisp and visually identical.
- `samsung14_2x.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only): Scaffolding against sky is equally sharp with no double edges; `|diff| x10` shows faint edge outlines.
- `Pixel 4a 20` (internal only, not committed): Perforated facade looks visually identical between `master` and `reg_refine=2`.

---

## 5. Default Decision

**Keep `reg_refine=0` (opt-in).** While the Gauss-Newton solver accurately recovers genuine homography + exposure perturbations (`< 0.05` LR px on perturbed synthetic bursts) and does not regress any real burst by `> 0.1` dB (`-0.056` to `+0.012` dB DCB, `-0.024` to `+0.008` dB CFA), `raw_sr`'s initial ECC homography is already within `0.02–0.03` LR px on the synthetic benchmark, so `reg_refine=2` does not achieve the `>= +0.5 dB` synthetic threshold and adds `3.2×–7.4×` CPU runtime.

Image licenses: `hdrplus_*` panels derive from the HDR+ Burst Photography Dataset (Hasinoff et al., 2016), CC-BY-SA; `samsung14_*` panels are not included in this repository (source burst: Bhat et al. 2021, deep-rep, CC BY-NC-SA 4.0, non-commercial); both shared under the same license. `own9_*` panels are the author's own burst.
