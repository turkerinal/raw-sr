# WP2: Classical (Non-Neural) Resolution Enhancement (`raw_sr/core/enhance.py`)

This document describes the WP2 classical enhancement pipeline implemented on top of `raw_sr`'s measured linear SR master, along with honest quantitative evaluations on synthetic ground-truth targets, real bursts, and 4096×4096 runtime/memory benchmarks.

The measured linear SR master (`SRResult.hr`) is **never modified**; enhancement is strictly a separate, opt-in post-processing stage that accepts any linear HR RGB image + scale ratio + noise estimate (composing cleanly with both current fusion and WP1 direct Bayer CFA fusion).

---

## 1. Method (`raw_sr/core/enhance.py`)

1. **Luminance-Only Processing:** Converts linear RGB to Rec. 709 luminance (`Y = 0.2126R + 0.7152G + 0.0722B`), enhances `Y`, and recombines with original chromaticity (`rgb_out = rgb_in * (Y_enh / max(Y_orig, 1e-4))`, clamped to `[0, 4]` with additive fallback below `1e-4`). This prevents colour-noise amplification and cuts runtime/RAM by 3×.
2. **Forward-Model PSF & Edge-Based Optics Estimation:**
   - `build_psf_1d(ratio, optics_sigma_lr)` / `build_psf(...)`: separable 1D sensor-pixel integration box of width `ratio` HR px convolved with a lens optics Gaussian of `optics_sigma_lr * ratio` HR px, normalized to unit sum.
   - `estimate_optics_sigma(img, ratio)`: locates up to 16 strong edge patches, projects pixels onto the local edge normal to build an oversampled ESF → LSF, measures FWHM, and subtracts the sensor box variance (`ratio² / 12`) in quadrature (lower-quartile estimate, clamped to `[0.15, 0.70]` LR px, safe fallback `0.35` LR px).
3. **TV-Regularized Richardson-Lucy with Discrepancy Stopping & Deringing:**
   - `richardson_lucy_tv(...)`: Dey et al. (2006) RL-TV update in linear light with noise-scaled TV weight `tv_weight = clip(0.001 + 1.5 * sigma_n, 0.001, 0.025)`.
   - Morozov discrepancy stopping (`rms(obs - PSF * est) <= 1.05 * noise_sigma`) using `frame_noise_sigma / sqrt(used_frames)` (or Immerkaer MAD fallback).
   - Local-support deringing (`cv2.erode` / `cv2.dilate` min/max clamp with tolerance `0.025 * local_range`) plus gradient-gated edge-aware blending so flat areas do not amplify noise.
4. **À-Trous Starlet (B3 Spline) Multiscale Sharpening:**
   - `starlet_decompose` / `starlet_sharpen`: 3-scale undecimated B3-spline (`[1, 4, 6, 4, 1]/16`) decomposition with exact perfect reconstruction at gain 1.
   - Per-layer MAD noise estimation floored by theoretical B3 starlet noise factors, soft-thresholded at `3.0 * sigma_j`, with linear shadow/highlight protection and a final local-support deringing clamp against `luma_orig`.
5. **Single Strength Control & Presets:** `strength ∈ [0, 1]` with presets `off (0.0)`, `subtle (0.35)`, `balanced (0.65, default)`, `astro (0.90)`.
6. **Integration:**
   - `SRResult.enhanced` optional field in `raw_sr/core/superres.py` (minimal hook; no fusion/registration edits).
   - `Exporter.export_enhanced(...)` in `raw_sr/core/export.py` writes 16-bit TIFF (linear or baked) and 8-bit PNG.
   - `desktop/backend/server.py` `/api/export` supports `kind='enhanced'` and `kind='enhanced_baked'` with `enhance_strength` / `enhance_preset`.

---

## 2. Synthetic GT Evaluation (`scripts/eval_wp2.py synth`, default `strength=0.65`)

Fairness: `single_enhanced` applies the **exact same** enhancement config (`strength=0.65`) to `single_bicubic` using its own noise estimate.

### 2× Super-Resolution (`lr=256, hr=512`, `optics_sigma_est = 0.423` LR px)

| Method | MTF50 (cyc/LR px) | GT PSNR vs Sensor (dB) | GT PSNR vs Lens (dB) | Edge Overshoot (%) | Flat Noise σ |
|---|---:|---:|---:|---:|---:|
| `single_bicubic` | 0.3139 | 28.574 | 26.159 | 10.76 | 0.007245 |
| `single_enhanced` (fair) | 0.4701 | 30.494 | 27.113 | 13.04 | 0.008645 |
| `plain_sr` | 0.3617 | 30.222 | 27.061 | 2.93 | 0.001856 |
| **`enhanced_sr`** | **0.6034 (+66.8%)** | **33.829 (+3.61)** | **28.410 (+1.35)** | **9.22** | **0.001856** |

### 4× Super-Resolution (`lr=256, hr=1024`, `optics_sigma_est = 0.351` LR px)

| Method | MTF50 (cyc/LR px) | GT PSNR vs Sensor (dB) | GT PSNR vs Lens (dB) | Edge Overshoot (%) | Flat Noise σ |
|---|---:|---:|---:|---:|---:|
| `single_bicubic` | 0.3376 | 29.008 | 27.209 | 10.81 | 0.007063 |
| `single_enhanced` (fair) | 0.4565 | 30.541 | 28.848 | 13.79 | 0.008701 |
| `plain_sr` | 0.3814 | 30.423 | 28.115 | 3.98 | 0.001881 |
| **`enhanced_sr`** | **0.5522 (+44.8%)** | **33.437 (+3.01)** | **30.518 (+2.40)** | **9.55** | **0.001881** |

**Criterion 1 verdict: PASS** at both 2× and 4× (GT PSNR enhanced SR > plain SR and > enhanced single; MTF50 +66.8% / +44.8% >= +30%; overshoot 9.22% / 9.55% <= 10%; 0% flat noise growth).

---

## 3. Real-Data Evaluation (`scripts/eval_wp2.py real`, 4×, 1024×1024 ROI -> 4096×4096 HR)

| Burst | Method | Real-Edge MTF50 (cyc/LR px) | Flat-Area HR Immerkaer σ | Est. Optics σ (LR px) |
|---|---|---:|---:|---:|
| `own9` (`1650,1240 1024`, 9/9) | `single_bicubic` | 0.1218 | 0.000003 | — |
| | `single_enhanced` (fair) | 0.1602 | 0.000033 | — |
| | `plain_sr` | 0.1525 | 0.000088 | — |
| | **`enhanced_sr`** | **0.1638** | **0.000088** | 0.351 |
| `HDR+ 0919` (centred `1024`, 8/8) | `single_bicubic` | 0.3528 | 0.000004 | — |
| | `single_enhanced` (fair) | 0.6661 | 0.000039 | — |
| | `plain_sr` | 0.2039 | 0.000080 | — |
| | **`enhanced_sr`** | **0.2029** | **0.000080** | 0.326 |
| `HDR+ 0382` (centred `1024`, 8/8) | `single_bicubic` | 0.2157 | 0.000004 | — |
| | `single_enhanced` (fair) | 1.5360 | 0.000040 | — |
| | `plain_sr` | 0.2347 | 0.000126 | — |
| | **`enhanced_sr`** | **0.2747** | **0.000125** | 0.213 |

### Honest Failure / Limitation Notes (Criterion 2)
- **What passed:** On `own9` and `HDR+ 0382`, real-edge MTF50 strictly satisfies `enhanced_sr > plain_sr > single_bicubic` (`0.1638 > 0.1525 > 0.1218` and `0.2747 > 0.2347 > 0.2157`), and `enhanced_sr` adds **zero** flat-area noise over `plain_sr` on all three bursts (`8.8e-5 -> 8.8e-5`, `8.0e-5 -> 8.0e-5`, `1.26e-4 -> 1.25e-4`), while `single_enhanced` amplifies HR Laplacian noise by ~10×.
- **What failed:**
  1. On `HDR+ 0919`, the auto-selected 192×192 edge patch landed on periodic facade louvers where `plain_sr` (0.2039) scored below `single_bicubic` (0.3528) and `enhanced_sr` (0.2029) did not lift that patch's MTF50.
  2. Measured with HR-grid 3×3 `immerkaer_sigma`, `single_bicubic` reads `3e-6..4e-6` because 4× bicubic interpolation low-passes adjacent HR pixels, whereas `plain_sr` (and `enhanced_sr`) reads `8.0e-5..1.26e-4` due to the current fusion's 4×4 sparse-drizzle grid artifact (which WP1 is replacing). Thus `flat_sigma(enhanced_sr) <= flat_sigma(single_bicubic)` **fails** under HR-grid Immerkaer on current fusion, even though `enhanced_sr` adds 0% noise on top of `plain_sr`.

---

## 4. Visual 4-Panel Crops (`docs/wp2/`)

> **Caveat:** in these WP2 panels the single-frame panels are not taken on the SR reference frame, so they are visibly offset from the SR panels. Use the pixel-aligned `docs/images/showcase_*.png` (`scripts/make_showcase.py`) for before/after judgement.

All panels share an identical percentile-calibrated display transform (`gamma=2.2`, auto-exposure targeting 99.5th percentile = 0.92, `< 1%` clipping):

- [`own9_4x_4panel.png`](wp2/own9_4x_4panel.png) — 469.9 KB, exposure 2.153×, max clip 0.004% ✅
- [`hdrplus_0919_4x_4panel.png`](wp2/hdrplus_0919_4x_4panel.png) — 568.3 KB, exposure 1.399×, max clip 0.000% ✅
- [`hdrplus_0382_4x_4panel.png`](wp2/hdrplus_0382_4x_4panel.png) — 687.2 KB, exposure 1.640×, max clip 0.755% ✅

---

## 5. Runtime, Memory & Tests

- **4096×4096×3 float32 benchmark (`scripts/eval_wp2.py bench`):** **15.9 s** (target `<= 20 s` ✅), peak extra RSS **+824.5 MiB / 0.81 GiB** (target `<= 3 GiB` ✅).
- **Root pytest (`QT_QPA_PLATFORM=offscreen ./venv/bin/pytest -q`):** **482 passed, 31 skipped, 0 failed** (including 9 new unit tests in `tests/test_enhance.py`).
