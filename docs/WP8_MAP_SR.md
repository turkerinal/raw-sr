# WP8: Full Forward-Model MAP Super-Resolution (`raw_sr/core/map_sr.py`)

Classical (non-neural) joint demosaicing, super-resolution, and deconvolution following Farsiu, Robinson, Elad & Milanfar (2004, 2006). Opt-in via `SuperResolution(fusion="map")`.

---

## 1. Formulation & Implementation

Solves the single inverse problem directly on calibrated raw Bayer samples:

$$y_k = D \cdot W_k \cdot B \cdot x + n_k$$

- **Initialisation:** `x0` is initialised from the default `demosaic_drizzle` HR output so MAP only refines.
- **Exact Forward/Adjoint Operators (`MAPForwardModel`):**
  - `B`: separable 2D PSF (default `SENSOR_PSF_SIGMA_PER_RATIO * ratio`, or pixel aperture box convolved with Gaussian lens $\sigma$, optionally estimated via WP6 `estimate_psf_sigma_edges` on `x0`), with exact replicate-boundary adjoint `B^T`.
  - `W_k`, `D`: bilinear gather from HR to frame `k`'s Bayer lattice using `to_drizzle_matrix`, paired with the exact bilinear scatter-add adjoint.
  - Verified by dot-product test: $|\langle Ax, y \rangle - \langle x, A^T y \rangle| / \max < 10^{-12}$ (float64) and $< 10^{-6}$ (float32).
- **Objective & Solver:** Huber robust L1 data term ($\delta = 2\sigma_n$, Immerkaer noise estimate) + smooth Bilateral Total Variation (BTV) on luminance $Y$ + Tikhonov/Laplacian smoothness on chroma ($R - G$, $B - G$), with regularization scaled by measured noise and effective sample density. Solved via Barzilai-Borwein projected gradient descent with Armijo backtracking line search and Morozov/relative-step stopping (35 max iterations).
- **GPU Acceleration:** GPU-resident CuPy implementation with custom CUDA `RawKernel`s for forward/adjoint CFA warping and transparent CPU fallback.

---

## 2. Synthetic GT Evaluation (`scripts/eval_wp8.py synth`)

Slanted edge + Siemens star (`lr=256`, `seed=7`), comparing estimated vs exact registration.

| Config | Method | Est PSNR (dB) | Est MTF50 | Est Ov% | Est Flat $\sigma$ | Exact PSNR (dB) | Exact MTF50 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2x, n=9 | `default` | 30.22 | 0.3617 | 1.34 | 0.00029 | 29.81 | 0.3372 |
| 2x, n=9 | `default+wp2` | 33.86 | 0.5950 | 4.30 | 0.00025 | 33.51 | 0.5305 |
| 2x, n=9 | `cfa_drizzle` | 34.64 | 0.3093 | 2.98 | 0.00189 | 34.56 | 0.3097 |
| 2x, n=9 | **`map` (WP8)** | **33.50** (+3.28) | 0.3535 | 4.15 | 0.00311 | **33.50** (+3.69) | 0.3503 |
| 2x, n=20 | `default` | 30.33 | 0.3566 | 0.89 | 0.00022 | 29.89 | 0.3355 |
| 2x, n=20 | `default+wp2` | 34.09 | 0.5947 | 4.19 | 0.00020 | 33.77 | 0.5647 |
| 2x, n=20 | `cfa_drizzle` | 35.20 | 0.3298 | 1.99 | 0.00149 | 35.11 | 0.3295 |
| 2x, n=20 | **`map` (WP8)** | **35.67** (+5.34) | 0.4621 | 2.54 | 0.00250 | **35.77** (+5.88) | 0.4655 |
| 3x, n=9 | `default` | 30.45 | 0.3760 | 1.21 | 0.00036 | 30.23 | 0.3273 |
| 3x, n=9 | `default+wp2` | 33.02 | 0.5197 | 3.06 | 0.00030 | 32.95 | 0.4879 |
| 3x, n=9 | `cfa_drizzle` | 33.59 | 0.2589 | 3.59 | 0.00183 | 33.54 | 0.2590 |
| 3x, n=9 | **`map` (WP8)** | **32.01** (+1.56) | 0.2214 | 5.17 | 0.00202 | **32.07** (+1.84) | 0.2263 |
| 3x, n=20 | `default` | 30.74 | 0.3785 | 0.84 | 0.00022 | 30.51 | 0.3577 |
| 3x, n=20 | `default+wp2` | 33.11 | 0.5148 | 2.74 | 0.00020 | 33.09 | 0.4960 |
| 3x, n=20 | `cfa_drizzle` | 35.12 | 0.3108 | 2.65 | 0.00130 | 35.17 | 0.3126 |
| 3x, n=20 | **`map` (WP8)** | **34.36** (+3.62) | 0.3648 | 3.24 | 0.00124 | **34.49** (+3.98) | 0.3676 |

---

## 3. Real Bursts, 2x, 1024x1024 ROI (`scripts/eval_wp8.py real`)

Both DCB-space and CFA-space held-out gain (dB, mean [min] over 5 evenly spaced held-out frames):

| Burst | Frames | Time def -> map | Peak RSS | DCB def -> map (dB) | CFA def -> map (dB) |
|---|---:|---|---:|---|---|
| `own9` | 9 | 8.44 -> 10.23 s | 1.88 GiB | +0.057 [+0.029] -> -2.969 [-3.239] | -2.847 [-3.067] -> **-0.569** [-0.703] |
| `hdrplus_0919` | 8 | 7.87 -> 12.13 s | 2.37 GiB | +1.386 [+0.932] -> -2.359 [-3.153] | -3.096 [-4.327] -> **+3.147** [+2.334] |
| `hdrplus_0382` | 8 | 7.07 -> 12.02 s | 2.37 GiB | +0.748 [+0.670] -> -5.313 [-5.469] | -7.142 [-7.476] -> **+0.299** [-0.275] |
| `samsung14` | 14 | 13.70 -> 15.87 s | 2.89 GiB | +1.027 [+0.619] -> -3.358 [-3.632] | -8.258 [-8.940] -> **-5.952** [-6.551] |
| `pixel4a20` | 20 | 20.03 -> 24.34 s | 3.35 GiB | +1.581 [+0.645] -> +0.976 [+0.346] | +0.788 [+0.207] -> **+2.126** [+1.733] |

---

## 4. Honest Visual Inspection (`docs/wp8/*.png`, all < 1.2 MB)

- [`own9_2x.png`](wp8/own9_2x.png): `map` matches `default` in sharpness with zero halos or staircasing, with a slight warm/desaturated cast from camera-WB CFA space.
- [`hdrplus_0919_2x.png`](wp8/hdrplus_0919_2x.png): `map` eliminates single-frame moire cleanly and resolves horizontal louvers, but is noticeably desaturated compared to `default+wp2` (which looks crisper and retains colour).
- [`hdrplus_0382_2x.png`](wp8/hdrplus_0382_2x.png): `default+wp2` is cleaner on the glass highlights; `map` shows slightly more fine grain and muted saturation.
- `samsung14_2x.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only): `map` resolves the scaffolding struts cleanly with no ghosting (unlike plain `cfa_drizzle`), with slightly more sky grain than `default+wp2`.
- `pixel4a20` (internal only, not committed): `map` renders the "HOTEL" sign and window mullions visibly crisper than `default` and competitive with `default+wp2`, without halos.

---

## 5. Default Decision

**Keep `fusion="map"` opt-in (default remains `demosaic_drizzle`).**

While WP8 MAP delivers large synthetic GT gains (+1.56 to +5.34 dB, beating all baselines at 2x n=20) and massive CFA-space held-out gains (+1.34 to +7.44 dB across all 5 real bursts, including +3.15 dB on HDR+ 0919) with fast CuPy execution (+1.8 s on 9 frames, < 2 GB RSS), it loses > 0.1 dB in DCB-space held-out on all 5 real bursts (due to the CFA-vs-sRGB colour-space mismatch) and visually trails `default+wp2` in colour saturation and grain on 3 of 4 publishable bursts.

Image licenses: `hdrplus_*` panels derive from the HDR+ Burst Photography Dataset (Hasinoff et al., 2016), CC-BY-SA; `samsung14_*` panels are not included in this repository (source burst: Bhat et al. 2021, deep-rep, CC BY-NC-SA 4.0, non-commercial); both shared under the same license. `own9_*` panels are the author's own burst.
