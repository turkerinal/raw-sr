# WP6: PSF-Estimated, Scene-Adaptive Classical Enhancement (`raw_sr/core/enhance.py`)

This document summarises the WP6 opt-in (`mode="adaptive"`) classical resolution enhancement implemented on top of `raw_sr`'s linear SR master, along with honest synthetic and real-data measurements against plain SR, WP2 (`mode="wp2"`), and single-frame fairness baselines.

---

## 1. What Was Built (`raw_sr/core/enhance.py`)

1. **Structure-Tensor PSF Estimation (`estimate_psf_sigma_edges`):**
   - Detects strong, straight edges via structure-tensor coherence ($C \ge 0.65$) per RGB channel and across a $4 \times 4$ spatial tile grid.
   - Rejects near-clipped patches (`max > 0.95`) and fits the exact forward-model ESF (sensor integration box $\ast$ optics Gaussian) via closed-form OLS ($R^2 \ge 0.90$), aggregating via the 30th percentile (sharpest in-focus edges) with fallback to `DEFAULT_OPTICS_SIGMA_LR = 0.35` LR px when fewer than 2 clean edges are found.
   - Reports global, per-channel (`psf_sigma_r/g/b`), edge count, and fallback flag in `EnhanceResult.details`.
2. **Noise-Adaptive RL-TV with Overshoot Monitor (`richardson_lucy_tv`):**
   - Uses the estimated PSF, Immerkaer-scaled TV weight, Morozov discrepancy stopping, and a 98th-percentile local overshoot/ringing monitor (`max_overshoot_frac=0.18`).
3. **Scene-Adaptive Local Mask & Safety Back-Off (`compute_scene_adaptive_mask`, `evaluate_enhance_safety`):**
   - Attenuates enhancement to `0` in flat regions ($< 1.5\sigma_n$) and near-clipped specular highlights (`local_max > 0.80`), keeping local-support deringing.
   - Measures edge MTF50 and flat-area Immerkaer noise on the exact same ROIs before/after enhancement; if MTF50 drops or flat noise grows $> 5\%$, automatically backs off strength (`0.5 -> 0.25 -> 0.0`).

---

## 2. Synthetic 2× PSF Sweep (`scripts/eval_wp6.py`)

Forward model: `synthesize_target_burst(ratio=2, lr=256, n=9)` with known lens PSF $\sigma_{\text{lens}} \in \{0.30, 0.45, 0.60, 0.80\}$ LR px.

| True $\sigma$ (LR px) | WP2 Est (Err%) | WP6 Est (Err%) | Plain SR MTF50 / PSNR$_s$ | WP2 MTF50 / PSNR$_s$ | WP6 MTF50 / PSNR$_s$ / PSNR$_l$ | WP6 Ov% / Backoff |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.467 (55.7%) | 0.421 (40.3%) | 0.362 / 30.22 | 0.653 / 33.18 | 0.512 / 32.76 / 27.52 | 6.7% / 1.0 |
| 0.45 | 0.468 (4.0%) | 0.526 (17.0%) | 0.335 / 32.01 | 0.534 / 34.84 | 0.440 / 34.92 / 29.50 | 5.8% / 1.0 |
| 0.60 | 0.517 (13.8%) | **0.662 (10.4%)** | 0.299 / 33.89 | 0.427 / 36.08 | 0.374 / **36.55** / 31.07 | 4.4% / 1.0 |
| 0.80 | 0.688 (13.9%) | **0.841 (5.1%)** | 0.220 / 35.77 | 0.352 / 36.35 | 0.220 / 35.77 / 32.03 | 1.7% / 0.0 |

*Note:* On isolated synthetic slanted edges (`tests/test_enhance_psf.py`), `estimate_psf_sigma_edges` recovers all four sigmas within `< 15%` error. On full SR bursts, demosaic + drizzle + MAP-IBP adds $\sim 0.25$ LR px of effective blur in quadrature, shifting small-$\sigma$ estimates (`0.30 -> 0.421`) while tracking larger blur (`0.60`, `0.80`) accurately (`10.4%`, `5.1%`).

---

## 3. Real 2× 4-Scene Benchmark (`scripts/eval_wp6.py`)

Pixel-aligned panels (`single | plain SR | WP2 enhanced | WP6 enhanced` + 2× zoom):

| Scene (2×) | Est. PSF (LR px) | Single MTF50 | Plain SR MTF50 | WP2 MTF50 | WP6 MTF50 | Flat $\sigma$ (Plain / WP2 / WP6) | Backoff | Panel |
|---|---:|---:|---:|---:|---:|---|---:|---|
| **HDR+ `0919`** (8/8) | 0.486 | 0.210 | 0.360 | 0.585 | **0.678** | 0.00139 / 0.00139 / 0.00139 | 1.0 | [`hdrplus_0919_2x_wp6.png`](wp6/hdrplus_0919_2x_wp6.png) (604 KB) |
| **HDR+ `0382`** (8/8) | 0.629 | 0.143 | 0.338 | 0.112 | 0.112 | 0.00104 / 0.00105 / 0.00105 | 1.0 | [`hdrplus_0382_2x_wp6.png`](wp6/hdrplus_0382_2x_wp6.png) (868 KB) |
| **Own 9-frame** (9/9) | 1.129 | 2.322* | 0.098 | 0.076 | **0.098** | 0.00111 / 0.00111 / 0.00111 | 0.0 | [`own9_2x_wp6.png`](wp6/own9_2x_wp6.png) (400 KB) |
| **Pixel4a 20** (20/20) | 0.298 | 0.644* | 0.285 | 0.434 | 0.429 | 0.00154 / 0.00154 / 0.00154 | 1.0 | not published (no stated license) |

*\*Single-frame MTF50 values above `0.50 cyc/LR px` are aliasing/moiré, not real detail.*

### Honest Visual Inspection of Panels
- **`HDR+ 0919`:** Single frame has strong diagonal colour moiré. Plain SR resolves the horizontal louvers cleanly. WP6 is visibly crisper than WP2 on the louvers with zero added flat noise or halos.
- **`HDR+ 0382`:** Single frame has heavy chroma speckle on the glass. Plain SR is clean. WP2 and WP6 both look slightly crisper on the amber glass stem, but both drop the measured green-channel slanted-edge MTF50 from `0.338` to `0.112` (the luma safety check did not catch the green-channel drop).
- **`Own 9-frame`:** Soft telephoto scene (`PSF 1.13 LR px`). WP6's safety check backed off to `0.0`, preserving plain SR's MTF50 (`0.098` vs WP2's `0.076`) with zero halos, at the cost of no visible sharpening.
- **`Pixel4a 20`:** Single frame shows strong colour aliasing across the perforated metal facade. Plain SR resolves individual holes; WP2 and WP6 both sharpen the perforations cleanly and tie visually (`0.434` vs `0.429`) with zero flat noise growth.

---

## 4. Default Mode Decision

**Decision: Keep `mode="adaptive"` (WP6) opt-in (`DEFAULT_ENHANCE_MODE = "wp2"`).**

WP6 beat WP2 clearly on `HDR+ 0919` (`0.678` vs `0.585`) and protected `Own 9-frame` (`0.098` vs `0.076`), tied on `Pixel4a 20` (`0.429` vs `0.434`), but still failed to prevent the green-channel MTF50 drop on `HDR+ 0382` (`0.338 -> 0.112`). Because it did not beat WP2 on $\ge 3$ of 4 real scenes and did not guarantee `0382 >= plain SR` on the green-channel edge metric, it remains opt-in via `EnhanceConfig(mode="adaptive")` or `enhance_image(..., mode="adaptive")`.
