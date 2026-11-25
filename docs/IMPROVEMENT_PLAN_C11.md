# Improvement Plan C11: Visible Classical (Non-Neural) Resolution Enhancement for `raw_sr`

This document proposes a classical, strictly non-neural engineering plan to achieve a **genuinely visible resolution increase** on top of `raw_sr`'s measured multi-frame RAW reconstruction, addressing the root causes diagnosed in [AUDIT_C11.md](AUDIT_C11.md).

---

## 1. Survey of Classical (Non-Neural) State of the Art

Each method below is evaluated specifically for **handheld 8–9 frame RAW bursts** (both DSLR-class and smartphone 10-bit RAW) with limited sub-pixel diversity (5–7 / 16 bins occupied at 4×) and 3.3×–4.6× stacking noise reduction.

### 1.1 Astrophotography Stacking (Bayer/CFA Drizzle, Rejection, Normalization)
- **Bayer/CFA Drizzle** (Fruchter & Hook 2002; DeepSkyStacker, Siril, PixInsight): Deposits raw single-channel Bayer samples directly onto the HR grid without per-frame demosaicing.
  - **What it buys on this data:** Eliminates demosaic-before-fusion colour moiré and recovers aliased high frequencies above single-frame Nyquist. On our synthetic resolution target, direct CFA drizzle (`pixfrac=0.5`, exact shifts) improves GT PSNR by **+4.23 dB at 2×** (30.33 → 34.56 dB) and **+2.34 dB at 4×** (30.66 → 33.00 dB) over the shipped demosaic-first pipeline, and resolves Siemens-star spokes substantially closer to the centre with zero false-colour zippering.
  - **Limitation:** At 4× with only 8–9 frames (5–7 / 16 subpixel bins occupied), a small box `pixfrac=0.5` leaves sparse holes and a visible 4×4 dot/grid pattern unless paired with adaptive kernel support or hole-filling regularization.
- **Sigma-Clipping / Winsorized Sigma-Clipping Rejection** (Siril, PixInsight): Rejects per-pixel outliers (hot pixels, cosmic rays, transient motion/ghosts) using robust median + MAD statistics.
  - **What it buys:** Clean outlier rejection without global ghost-gate fallback; prevents moving objects from leaving 4×4 dot trails.
- **Local Normalization**: Matches per-frame exposure/vignetting gradients before stacking.
  - **What it buys:** Fixes the mixed-exposure bug seen on HDR+ bursts `33TJ` and `6G7M` (where a 3.8× brighter frame `N009` caused reference selection to collapse to 1/10 frames).

### 1.2 Google Handheld Multi-Frame SR (Wronski et al. 2019)
- **Direct Bayer Kernel Regression with Anisotropic Structure-Tensor Kernels + Robustness Model**: Fuses raw Bayer frames directly onto the HR grid using spatially varying 2D Gaussian kernels steered along local image edges (structure tensor: narrow across edges, elongated along edges), weighted by a per-frame robustness model.
  - **What it buys on this data:** Directly solves **both** major weaknesses of current `raw_sr`:
    1. Avoids per-frame demosaicing (eliminating moiré and recovering aliased HF).
    2. Eliminates the 4×4 sparse-drizzle dot/grid artifact at 3×–4× by adapting kernel support to local sample density and edge orientation, so edges stay crisp while flat areas stay smooth and noise-free.
  - **Fit:** Highest-impact architectural upgrade for the measured master itself.

### 1.3 Classical Deconvolution (Richardson-Lucy + TV/Regularization, Wiener, Blind PSF)
- **Regularized Richardson-Lucy (RL-TV / Damped RL)** (Lucy 1974; Richardson 1972; Dey et al. 2006): Iteratively inverts the combined optical + sensor-aperture PSF (`sigma ≈ 0.35–0.55` LR px) with Total Variation (TV) or wavelet/VST damping to suppress noise amplification and ringing.
  - **What it buys on this data:** Because 8–9 frame stacking reduces linear noise by **3.3× to 4.6×** (+10.4 to +13.3 dB SNR), the stacked master can sustain 6–12 RL iterations where a single frame breaks down into noise and ringing. On our synthetic target, 8-iteration RL on `raw_sr` raises 4× MTF50 from **0.381 → 0.534 cyc/LR px (+40%)** and *improves* GT PSNR (**30.42 → 30.97 dB**), whereas the exact same RL on a single bicubic frame *degrades* GT PSNR (**29.01 → 28.16 dB**) and amplifies noise.
- **Wiener Deconvolution**: Fast FFT-based linear inverse filter; prone to global ringing around high-contrast edges unless edge-tapered, less flexible than damped RL-TV.
- **PSF Estimation from Edges (Slanted-Edge / Blind)**: Estimates effective PSF width directly from high-contrast edges in the stacked master so deconvolution adapts to the lens/aperture without manual tuning.
- **Deringing**: Local overshoot clamping (clamping deconvolved values toward local min/max neighbourhood, as in PixInsight's Local Support deringing) prevents bright/dark halos on high-contrast edges.

### 1.4 Wavelet / Multiscale Sharpening (À Trous Starlet, MLT)
- **À Trous (Undecimated) Starlet / Multiscale Linear Transform (MLT)** (Starck & Murtagh; PixInsight, RegiStax): Decomposes the image into dyadic detail layers (`scale 1, 2, 3, ...`) plus residual; applies noise-thresholded soft shrinkage on fine scales and controlled contrast boost on medium-fine scales.
  - **What it buys:** Gives astrophotography-grade micro-contrast and crispness on fine textures (roof tiles, foliage, brick, facade louvers) while keeping flat sky/shadow regions noise-free. Best used as a gentle, toggleable stage after RL-TV deconvolution in the **Enhanced** output.

### 1.5 Farsiu Robust Multi-Frame SR (L1 Data Fidelity + Bilateral TV)
- **L1 + BTV MAP SR** (Farsiu et al. 2004): Minimizes `sum_k || D B M_k X - Y_k ||_1 + lambda * BTV(X)`.
  - **What it buys:** Stronger data fidelity and outlier robustness than 1-sweep L2 IBP, without the over-smoothing of Gaussian/isotropic prox. Good complement to CFA kernel regression if multi-sweep MAP refinement is retained.

### 1.6 Edge-Directed Upscaling (NEDI / EDI / Lanczos)
- **What it buys:** Marginal on multi-frame SR (single-frame interpolation trick); useful only as a fallback when `frames_used == 1` or coverage is zero.

---

## 2. Prioritized Implementation Plan (Work Packages)

### WP1 — Direct Bayer CFA Fusion with Anisotropic Structure-Tensor Kernel Regression (Measured Master)
- **Goal:** Replace per-frame demosaic + isotropic box drizzle with direct Bayer CFA fusion using edge-directed anisotropic kernels (Wronski et al. 2019 style) and per-frame exposure normalization.
- **Owner files:**
  - `raw_sr/core/cfa_drizzle.py` (promote from experimental to core; add anisotropic structure-tensor kernel regression + adaptive support radius to eliminate 4×4 grid artifacts)
  - `raw_sr/core/superres.py` (wire CFA-first path, add per-frame exposure normalization before reference selection/fusion to fix mixed-exposure bursts)
  - `tests/test_cfa_drizzle.py` (regression + synthetic resolution target tests)
- **Expected visible gain:**
  - Elimination of false-colour moiré on fine periodic textures (blinds, tiles, Siemens star).
  - Elimination of the 4×4 sparse-drizzle dot/grid artifact at 3×–4×.
  - +2.0 to +4.0 dB GT PSNR on synthetic resolution targets; cleaner, more faithful measured master.
- **Risks:**
  - Colour fringing if per-channel sample coverage is unbalanced at 4× (mitigated by adaptive kernel width + chroma-luma joint weighting).
  - Higher compute time than simple scatter (mitigated by separable/vectorized NumPy/OpenCL kernels).
- **Acceptance criteria:**
  - Synthetic 2×/4× GT PSNR beats current pipeline by `>= +1.5 dB` and eliminates Siemens-star false colour.
  - Real-data held-out min gain stays `>= 0.0 dB` across `own 9-frame burst` and HDR+ `0919`, `0382`, `6G7M` (and improves on `33TJ`).
  - Zero 4×4 grid artifact on flat/periodic regions at 4×.

### WP2 — Regularized Deconvolution & Multiscale Detail Enhancement ("Enhanced" Output)
- **Goal:** Add a dedicated, clearly labelled **Enhanced** output (`Export Enhanced` / `sr_<ratio>x_enhanced.png`) built on top of the measured linear master: edge-estimated PSF -> TV-regularized / deringed Richardson-Lucy deconvolution -> optional noise-gated à trous starlet micro-contrast boost.
- **Owner files:**
  - `raw_sr/core/deconvolve.py` (new module: edge-based PSF estimator, damped/TV-regularized RL with local-support deringing, à trous starlet layer sharpening)
  - `raw_sr/core/export.py` (support exporting both `Master` linear TIFF and `Enhanced` TIFF/PNG)
  - `scripts/sr_acceptance.py` (add fair enhanced-vs-enhanced comparison: `single_frame + same_enhancement` vs `raw_sr + same_enhancement`)
  - `tests/test_deconvolve.py` (synthetic MTF50, ringing/overshoot bound, noise amplification check)
- **Expected visible gain:**
  - **Unmistakable, immediately visible sharpness and micro-detail jump** at 2×, 3×, and 4×.
  - Synthetic MTF50 increase of **+35% to +60%** over single-frame bicubic (crossing past `0.50 cyc/LR px` LR Nyquist).
  - Honest advantage demonstrated clearly: applying the **exact same** deconvolution to a single frame amplifies noise and ringing, whereas the 3.3×–4.6× quieter stacked master stays clean and crisp.
- **Risks:**
  - Ringing/halos on high-contrast edges (mitigated by TV damping + local-support overshoot clamping + overshoot metric gate).
  - Noise amplification in dark shadows (mitigated by VST-domain noise gating before starlet boost).
- **Acceptance criteria:**
  - Synthetic 4× MTF50 `>= 0.50 cyc/LR px` (vs `0.338` single bicubic) AND GT PSNR strictly higher than both `current_sr` and `single_bicubic + same_RL`.
  - Real-edge MTF50 on `own 9-frame burst` and `HDR+ 0919` improves by `>= +25%` over single bicubic.
  - Edge overshoot (ringing) `<= 5%` of local edge step height.
  - Fair baseline check: `SNR(raw_sr + enh) - SNR(single + enh) >= +6 dB` on flat regions.

### WP3 — Sub-Pixel Registration & Exposure-Invariant Frame Gating
- **Goal:** Improve sub-pixel registration accuracy on low-contrast/aliased bursts (e.g., `33TJ` where `align_residual_mean_px = 0.304 px` exceeds the `0.125 px` 4× requirement) via full-resolution (`ecc_scale=0.5` or pyramid coarse-to-fine to `1.0`) ECC on luminance + per-frame exposure normalization + Winsorized sigma-clipping rejection.
- **Owner files:**
  - `raw_sr/core/superres.py` (pyramid ECC to full-res on ROI, exposure-invariant reference selection, sigma-clipped frame rejection)
  - `tests/test_registration.py` (mixed-exposure burst test, subpixel accuracy `< 0.08 LR px` on synthetic burst)
- **Expected visible gain:**
  - Prevents registration blur on bursts with larger residuals (`33TJ`, `6G7M`), turning negative held-out gain on `33TJ` positive.
  - Eliminates ghosting trails from transient moving objects without collapsing to single-frame output.
- **Risks:**
  - Longer registration time (mitigated by coarse-to-fine pyramid with early stopping).
- **Acceptance criteria:**
  - `align_residual_mean_px <= 0.10 LR px` on all 5 real bursts (`own 9-frame`, `0919`, `0382`, `33TJ`, `6G7M`).
  - Full 10-frame `33TJ` and `6G7M` bursts fuse `9/10` frames automatically without manual `short/` filtering.

### WP4 — Honest Visual Showcase & Auto-Exposure Preview
- **Goal:** Replace misleading/clipped README images and fix `scripts/sr_acceptance.py` / `scripts/compare_versions.py` display tone-mapping.
- **Owner files:**
  - `scripts/sr_acceptance.py` (percentile-based auto-exposure instead of hardcoded `exposure=4.0`; export 4-panel honest comparison strips)
  - `README.md` (replace `compare_4x_crop.png` and `c4_4x_tex1_sbs.png` with pixel-aligned, properly exposed single-frame vs `raw_sr` Master vs `raw_sr` Enhanced comparisons)
- **Expected visible gain:**
  - Eliminates the 70% highlight clipping on `HDR+ 6G7M` and the 39 HR px panel misalignment in the README hero.
  - Clearly communicates both the honest physical measurement (Master) and the non-neural mathematical enhancement (Enhanced) against fair baselines.

---

## 3. Recommended Execution Order

1. **WP2 (Regularized Deconvolution + Multiscale Enhancement)** *or* **WP1 (Direct Bayer CFA Anisotropic Fusion)** first:
   - **Recommended first work package: WP1** to fix the physical foundation (eliminate pre-demosaic moiré and 4×4 sparse-drizzle grid artifacts on the measured master), immediately followed by **WP2** to unlock the dramatic, astrophotography-grade visible sharpness jump on top of the clean CFA master.
   - Then **WP3** (registration & exposure invariance) and **WP4** (honest visual showcase).
