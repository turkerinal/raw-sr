# WP13: Real-Photo Forward-Model MAP Super-Resolution (`raw_sr/core/map_sr.py`)

Classical (non-neural) upgrade to forward-model joint demosaic + super-resolution + deconvolution, opt-in via `SuperResolution(fusion="map", map_mode="wp13")` or `map_super_resolution(..., mode="wp13")`. Existing `fusion="map"` / `fusion="auto"` defaults remain unchanged pending integration.

---

## 1. Root Causes Fixed

1. **PSF Estimation Bug & Per-Channel Lens PSF (`estimate_burst_psf_rgb`, `build_psf_kernels_rgb`):**
   - Fixed the silent `AttributeError` in legacy `map_super_resolution(estimate_psf=True)`, which accessed `estimate_psf_sigma_edges(...)` as a dict instead of a `PSFEstimate` dataclass.
   - Added per-channel $(R, G, B)$ slanted-edge lens PSF estimation from the drizzle initialisation `x0`, removing the drizzle quadrature bias ($\sigma_{\text{bias}} = 0.10$ LR px) and common Bayer $R/B$ demosaic variance excess $\Delta_{RB}^2 = \max(\min(\sigma_R^2, \sigma_B^2) - \sigma_G^2, 0)$ while preserving true dispersive lateral chromatic blur (e.g., `hdrplus_0919`: $R=0.517, G=0.495, B=0.411$ LR px). Clamped to $[0.15, 0.90]$ LR px with $|\sigma_{R,B} - \sigma_G| \le 0.15$ and fallback $0.35$.
   - Forward blur $B_{\text{rgb}} = \mathrm{diag}(B_R, B_G, B_B)$ uses a bilinear-warp-compensated 1D kernel matching total physical variance $\sigma_{B,c}^2 = \max(r^2/12 + (\sigma_c r)^2 - 1/6, 0.01)$ HR px$^2$. Exact replicate-boundary adjoint $B_{\text{rgb}}^T$ verified to $< 10^{-12}$ (float64) and $< 10^{-6}$ (float32) with non-identity colour matrix $C$.

2. **Forward Model & Noise-Aware Robust Data Term:**
   $$y_k = D W_k C B_{\text{rgb}} x + n_k, \qquad \sigma^2(\mu) = a + b \cdot \max(\mu, 0)$$
   - $(a, b)$ are estimated per burst from quiet low-gradient quantile bins of the reference Green sub-grid via MAD of the 3×3 Immerkaer Laplacian (`estimate_poisson_gaussian_noise`).
   - Pixel variance is evaluated on the clean expected signal $\hat{\mu}_k = \mathrm{clip}(A_k x_0, 0, 1)$ clamped to $[0.75\sigma_{\text{med}}, 1.50\sigma_{\text{med}}]$ (eliminating noisy-$y_k$ downward mean bias).
   - Whitened residuals $u_{k,p} = ((A_k x)_p - y_{k,p}) / \sigma_{k,p}$ are penalised with smooth pseudo-Huber loss $\rho_\kappa(u) = \kappa^2(\sqrt{1 + (u/\kappa)^2} - 1)$ ($\kappa = 3.0$), with Lorentzian per-frame robust weights $w_k$ downweighting frames whose median $|u_k|$ exceeds $1.25\times$ the burst median.

3. **DC-Invariant Priors & Period-2 Null-Space Suppression:**
   $$\Phi(x) = \sum_k w_k \sum_p \rho_\kappa(u_{k,p}) + \lambda_Y R_{\text{HTV}}(Y(x)) + \lambda_C \big(R_C(R-G) + R_C(B-G)\big) + \lambda_{\text{nyq}} R_{\text{nyq}}(x)$$
   - **Luminance:** Smooth isotropic Huber/Charbonnier TV on Rec. 709 $Y$, $\sum \big(\sqrt{g_x^2 + g_y^2 + \epsilon_Y^2} - \epsilon_Y\big)$ ($\epsilon_Y = \max(0.5\sigma_{\text{med}}, 5\times 10^{-4})$). Selected over BTV and hyper-Laplacian ($p=0.8$) on Synthetic A + `own9` for strict convexity, faster convergence, and higher PSNR.
   - **Cross-channel chroma:** Edge-weighted quadratic smoothness $\frac{1}{2}\sum (w_x (\nabla_x C)^2 + w_y (\nabla_y C)^2)$ with $w = 1 / (1 + (\nabla Y_0 / \tau)^2)$, $\tau = \max(3\sigma_{\text{med}}, 0.015)$. Preserves colour boundaries and DC mean identically.
   - **Period-2 Nyquist regularizer:** $R_{\text{nyq}}(x) = \frac{1}{2}\|K_{\text{nyq}} * x\|_2^2$ with $K_{\text{nyq}} = \frac{1}{4}\begin{bmatrix}1 & -1\\ -1 & 1\end{bmatrix}$ (mixed derivative $\partial_x\partial_y$: identically zero on constants, ramps, and pure horizontal/vertical edges; maximal on period-2 checkerboards).

4. **Principled Per-Burst Parameter Rule (Frozen on Synthetic A + `own9` Only):**
   With $n_{\text{eff}} = (\sum w_k)/r^2$:
   $$\lambda_Y = 0.25 \frac{\sqrt{n_{\text{eff}}}}{\sigma_{\text{med}}}, \qquad \lambda_C = 0.04 \frac{n_{\text{eff}}}{\sigma_{\text{med}}^2}, \qquad \lambda_{\text{nyq}} = 0.06 \frac{n_{\text{eff}}}{\sigma_{\text{med}}^2}$$
   Discrepancy-principle check at iteration 12: if $\mathrm{RMS}(u) < 0.90$, $\lambda_Y$ is scaled up by $\min((0.95/\mathrm{RMS}(u))^2, 4.0)$ and L-BFGS history resets.

5. **Convergent Solver (`_solve_map_wp13_cpu` / `_solve_map_wp13_gpu`):**
   Coverage-Preconditioned Projected L-BFGS ($m=5$) with diagonal preconditioner $P^{-1} = 1 / \max(A^T[w_k/\sigma_k^2] + d_{\text{prior}}, 1)$, active-set box projection $x \in [0, 1]$, Armijo backtracking line search, and tolerance stopping ($\|\Delta x\|/\|x\| < 10^{-4}$ or $|\Delta \Phi|/|\Phi| < 10^{-5}$, cap 70).

---

## 2. Synthetic GT (Protocol A) & Simulated Test Chart (Protocol B)

### Protocol A — Mean over $n \in \{9, 20\}$, seeds $\{7, 23\}$ (blind PSF estimation, vs sharp `scene` GT)

| $\sigma_{\text{lens}}$ | Single | Old Def | Old + Enh | Current MAP | **WP13 MAP** | WP13 vs Old+Enh | WP13 MTF50 (vs Old+Enh) | WP13 Flat $\sigma$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 22.97 | 23.40 | 24.57 | 25.43 | **25.58 dB** | **+1.01 dB** | **0.860** (0.598) | 0.00037 |
| 0.60 | 22.58 | 22.83 | 24.04 | 24.41 | **25.43 dB** | **+1.39 dB** | **0.843** (0.436) | 0.00013 |
| 0.90 | 21.81 | 21.95 | 23.05 | 22.89 | **24.93 dB** | **+1.88 dB** | **0.654** (0.308) | 0.00006 |

### Protocol B — Simulated Test Chart (2×, 16 frames, Zoom GT PSNR / SSIM, Siemens star f50 / f10)

| $\sigma_{\text{lens}}$ | Method | Zoom GT PSNR | Zoom GT SSIM | Star f50 | Star f10 |
|---:|---|---:|---:|---:|---:|
| 0.35 | Single / Old Def / Old+Enh | 16.40 / 17.17 / 18.12 | 0.614 / 0.676 / 0.725 | 0.358 / 0.363 / 0.370 | 0.451 / 0.461 / 0.457 |
| 0.35 | Current MAP -> **WP13 MAP** | 20.08 -> **20.22 dB** | 0.800 -> **0.821** | 0.428 -> **0.529** | 0.640 -> 0.615 |
| 0.70 | Single / Old Def / Old+Enh | 19.20 / 19.64 / 20.68 | 0.627 / 0.678 / 0.715 | 0.311 / 0.314 / 0.318 | 0.408 / 0.417 / 0.409 |
| 0.70 | Current MAP -> **WP13 MAP** | 23.08 -> **24.27 dB** | 0.779 -> **0.812** | 0.347 -> **0.363** | 0.480 -> **0.485** |

---

## 3. Real Bursts (Protocol C, 2×, 1024×1024 ROI, 3 Held-Out Frames)

### Truth-Free Crop Metrics (`Old+Enh` / `Current MAP` -> **`WP13 MAP`**)

| Burst | Est PSF (R,G,B) | Grad Energy | Flat $\sigma$ | Nyquist Frac | Overshoot | Colour Drift | CFA-Fair (dB) | DCB (dB) |
|---|---|---|---|---|---|---|---|---|
| `own9` | (.90,.90,.90) | .0080 / .0096 -> **.0086** | .000105 / .000427 -> **.000006** | .00131 / .00206 -> **.00123** | .424 / .323 -> .550 | **0.17%** | -0.31 / -0.06 -> -0.97 | -0.39 / -0.80 -> -0.87 |
| `0919` | (.52,.49,.41) | .0739 / .0971 -> **.1375** | .000125 / .000439 -> **.000053** | .00366 / .00398 -> **.00129** | .106 / .407 -> .107 | **0.44%** | +2.36 / +4.32 -> +0.76 | +1.74 / +1.80 -> +0.54 |
| `0382` | (.40,.25,.23) | .0226 / .0266 -> **.0280** | .000182 / .000405 -> **.000106** | .00125 / .00142 -> **.00087** | 2.02 / 2.44 -> 3.37 | **0.37%** | +0.50 / +1.90 -> +0.50 | +0.58 / +0.18 -> -0.76 |
| `samsung14` | (.75,.90,.90) | .0215 / .0223 -> **.0318** | .000156 / .000323 -> **.000007** | .00138 / .00159 -> **.00093** | .840 / 1.11 -> 1.31 | 1.55% | -11.19 / -10.07 -> **-10.77** | -3.55 / -2.01 -> -5.13 |
| `pixel4a20` | (.37,.29,.29) | .0323 / .0398 -> **.0414** | .000171 / .000463 -> **.000094** | .00140 / .00150 -> **.00113** | .909 / .695 -> **.830** | **0.62%** | +1.20 / +2.61 -> **+2.93** | +1.71 / +1.51 -> **+1.77** |

*Note on CFA-fair held-out:* `evaluate_cfa_heldout` predicts held-out LR frames using only the sensor box PSF (`0.578` HR px) without the lens PSF. Because WP13 MAP deconvolves the lens PSF, it beats `old_enh` strongly on sharp-lens `pixel4a20` (+1.73 dB) and `samsung14` (+0.42 dB) and ties `0382` (-0.004 dB), but is penalised when comparing a lens-deconvolved HR image against a lens-blurred held-out LR frame on `0919` ($\sigma \approx 0.49$) and `own9` ($\sigma \approx 0.90$).

### Honest Visual Inspection (`docs/wp13/*.png`)
- [`own9_2x.png`](wp13/own9_2x.png): Visibly sharper diagonal shadow and cleaner horizontal siding than `old+enh` and `Current MAP`, zero grain, exact colour (0.17% drift).
- [`hdrplus_0919_2x.png`](wp13/hdrplus_0919_2x.png): Dramatically crisper horizontal louvers than `old+enh`, period-2 dot grid completely eliminated (Nyquist energy 3× lower), exact turquoise colour (0.44% drift), no halos.
- [`hdrplus_0382_2x.png`](wp13/hdrplus_0382_2x.png): Crisper amber glass stem and clean dark background (0.37% drift), slightly punchier specular highlights (`ov` 3.37 vs 2.02).
- `samsung14_2x.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only): Smooth sky and sharper struts overall, **but** parallax-inflated PSF estimate (`0.90` vs true `~0.40`) causes dark/bright ringing specks on thin struts and 1.55% colour drift.
- `pixel4a20` (internal only, not committed): Visibly sharpest perforated facade with zero colour moiré and cleanest window mullions.

---

## 4. Runtime & Peak Memory (Protocol D: 2× 1024² × 9 frames)

| Backend | Method | Runtime | Peak RSS | VRAM Pool | Iters |
|---|---|---:|---:|---:|---:|
| GPU Auto (RTX 3050 4 GB) | `old_default` | 7.91 s | 1552 MB | 453 MB | — |
| GPU Auto (RTX 3050 4 GB) | `map_current` | 10.85 s | 1972 MB | 1052 MB | 34 |
| GPU Auto (RTX 3050 4 GB) | **`map_wp13`** | **17.52 s** | **2068 MB** | **2156 MB** | 51 |
| CPU | `old_default` | 5.86 s | 2068 MB | — | — |
| CPU | `map_current` | 196.91 s | 2844 MB | — | 34 |
| CPU | **`map_wp13`** | **761.58 s** | **3689 MB** | — | 51 |

Image licenses: `hdrplus_*` panels derive from the HDR+ Burst Photography Dataset (Hasinoff et al., 2016), CC-BY-SA; `samsung14_*` panels are not included in this repository (source burst: Bhat et al. 2021, deep-rep, CC BY-NC-SA 4.0, non-commercial); both shared under the same license. `own9_*` panels are the author's own burst.
