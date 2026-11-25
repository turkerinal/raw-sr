# WP14: Plug-and-Play (PnP-HQS / RED) Multi-Frame Super-Resolution (`raw_sr/core/pnp_sr.py`)

Classical (non-neural, zero learned weights) joint demosaicing, super-resolution, and deconvolution solver operating directly on calibrated raw Bayer mosaics.

---

## 1. Mathematical Formulation

Given $N$ calibrated low-resolution (LR) Bayer frames $\{y_k\}_{k=1}^{N}$ of shape $(H_{\text{LR}}, W_{\text{LR}})$, per-frame quality weights $\{w_k\}_{k=1}^{N}$, and planar homographies $\{W_k\}_{k=1}^{N}$ mapping reference LR coordinates to frame $k$, the physical observation model for the high-resolution (HR) linear sRGB image $x \in [0, 1]^{H_{\text{HR}} \times W_{\text{HR}} \times 3}$ ($H_{\text{HR}} = r H_{\text{LR}}, W_{\text{HR}} = r W_{\text{LR}}$) is:

$$y_k = D \, W_k \, B \, (C x) + n_k = A_k(x) + n_k$$

where:
- $C \in \mathbb{R}^{3 \times 3}$ is the per-burst least-squares camera-to-sRGB colour matrix estimated on unclipped interior sites of the reference frame (`estimate_colour_matrix`, LR RMS $< 10^{-5}$, $\text{cond}(C) \in [1.0, 2.9]$), preceded by $\text{Gr}/\text{Gb}$ green-phase gain equalization (`equalize_green_imbalance`).
- $B = \text{diag}(B_R, B_G, B_B)$ is the separable 2D point-spread function (PSF) combining the $r \times r$ HR pixel integration box $\text{Box}_r$ with the channel-specific optical Gaussian $G_{\sigma_{\text{lens}, c} \cdot r}$:
  $$k_{1\text{D}, c}(u) = \left( \text{Box}_r \ast G_{\sigma_{\text{lens}, c} \cdot r} \right)(u), \qquad \sum_u k_{1\text{D}, c}(u) = 1$$
- $W_k$ is the exact bilinear resampling operator from the pixel-centre-aligned HR lattice (`to_drizzle_matrix`), and $W_k^T$ is its exact scatter-add adjoint.
- $D$ is the Bayer CFA channel selector, and $D^T$ places each scalar Bayer sample into its corresponding RGB plane.

### 1.1 Blind Optical PSF Estimation (`estimate_calibrated_burst_psf`)

In `map_sr.py`, `estimate_psf=True` failed silently because `estimate_psf_sigma_edges` returns a `PSFEstimate` dataclass rather than a `dict`. In `pnp_sr.py`, we call `estimate_psf_sigma_edges(init_hr, ratio=r)` directly, reading `psf_est.sigma_lr`, `psf_est.sigma_rgb`, `psf_est.n_edges`, and `psf_est.used_fallback`. Because `init_hr` (`demosaic_drizzle` + 1-sweep IBP) carries an intrinsic deposition splat blur of $\sigma_{\text{splat}} \approx 0.22$ LR px, we calibrate each channel's optical blur by quadrature subtraction:

$$\sigma_{\text{lens}, c} = \text{clip}\!\left( \sqrt{\max\!\left(\hat{\sigma}_{\text{edge}, c}^2 - \sigma_{\text{splat}}^2, \, \sigma_{\min}^2\right)}, \; \sigma_{\min}, \; \sigma_{\max} \right)$$

with green ($\times 2$ spatial sampling density on the Bayer grid) anchoring the global estimate $\sigma_{\text{lens}}$ and red/blue bounded within physical chromatic dispersion ($\pm 0.08$ LR px).

---

## 2. Plug-and-Play (PnP-HQS / RED) Splitting Architecture

To overcome the five failure modes of 35-step Barzilai-Borwein MAP gradient descent (PSF mismatch, high-frequency solver stagnation on wide PSFs, period-2 Nyquist dot texture, flat-area noise floor, and deconvolutional colour drift), `pnp_super_resolution` decouples the inverse problem via Half-Quadratic Splitting (HQS) / Regularization by Denoising (RED):

$$\min_{x, z} \sum_{k=1}^{N} w_k \, \rho_{\delta}\!\left(A_k x - y_k\right) + \frac{\rho}{2} \|x - z\|_2^2 + \lambda \, \Phi_{\text{PnP}}(z)$$

### 2.1 Stage 1 — Forward-Model Data Reconstruction
- **Anti-Aliased Sensor-Aperture Inversion & HQS Conjugate Gradients:**
  - For blind multi-frame fusion on calibrated camera bursts, the data step solves the exact sensor-aperture Bayer forward problem ($\sigma_{\text{sensor}} = 0.289 r$ HR px, Huber threshold $\delta = 2\sigma_n$, $40$ BB/HQS steps with luminance BTV weight $0.08$ and chroma weight $4.5$), which is well-conditioned right up to the HR Nyquist frequency and avoids the low-frequency stagnation of wide-PSF first-order gradient descent.
  - When an explicit blur kernel $\sigma_{\text{lens}} > 0$ is specified for direct linear deconvolution, `_solve_hqs_cg_cpu` solves the Iteratively Reweighted Least Squares (IRLS) normal equations at continuation stage $t$:
    $$\left( \sum_{k=1}^{N} w_k A_k^T M_k^{(t)} A_k + \rho_t I \right) \Delta x = \sum_{k=1}^{N} w_k A_k^T M_k^{(t)} (y_k - A_k x^{(t)}) + \rho_t \left(z^{(t)} - u^{(t)} - x^{(t)}\right)$$
    via **Conjugate Gradients (CG)** with Huber IRLS weights $M_k^{(t)} = \min\!\left(1, \frac{\delta}{|A_k x^{(t)} - y_k|}\right)$ and continuation schedule $\rho_t = \rho_0 \cdot 1.35^t$, $\sigma_d^{(t)} = 1.35 \sigma_n \cdot 0.55^{t/(T-1)}$.

### 2.2 Stage 2 — Directional Period-2 Nyquist Notch Filter (`suppress_period2_notch`)
At $2\times$ upscaling, sub-pixel Bayer phase differences leave residual energy at the HR Nyquist frequencies $(\omega_x, \omega_y) \in \{(\pi, \pi), (\pi, 0), (0, \pi)\}$, which manifests visually as a regular period-2 dot/grid pattern on horizontal louvers (`HDR+ 0919`) and roof tiles (`own9`). Using the 1D high-pass $h_{\text{HP}} = \frac{1}{4}[-1, 2, -1]$ and low-pass $h_{\text{LP}} = \frac{1}{4}[1, 2, 1]$:
1. The pure 2D diagonal checkerboard mode $Y_{\pi,\pi} = (h_{\text{HP}} h_{\text{HP}}^T) \ast Y$ is subtracted completely: $Y_1 = Y - Y_{\pi,\pi}$.
2. The axis-Nyquist modes $Y_{\pi,0} = (h_{\text{LP}} h_{\text{HP}}^T) \ast Y_1$ and $Y_{0,\pi} = (h_{\text{HP}} h_{\text{LP}}^T) \ast Y_1$ are gated by the orthogonal Sobel gradient magnitudes $(|G_x|, |G_y|)$ so period-2 dots *along* horizontal louvers ($|G_y| \gg |G_x|$) or vertical struts ($|G_x| \gg |G_y|$) are damped without dulling the edge jump across the feature:
   $$Y_2 = Y_1 - 0.75 \left( \frac{|G_y| + 2\sigma_n}{|G_x| + |G_y| + 2\sigma_n} Y_{\pi,0} + \frac{|G_x| + 2\sigma_n}{|G_x| + |G_y| + 2\sigma_n} Y_{0,\pi} \right)$$
3. Isotropic 4-neighbor Laplacian oscillations below $2.8\sigma_n$ are soft-thresholded.

### 2.3 Stage 3 — Classical Non-Local Collaborative Prior (`denoise_patch_nlm_luma` / `bm3d` + `denoise_cross_channel_chroma`)
In luminance-chrominance coordinates $Y = 0.299 R + 0.587 G + 0.114 B$, $C_r = R - G$, $C_b = B - G$:
- **Permissive Default (`denoiser="nlm"`):** Evaluates shift-integral **Non-Local Patch Means** (Buades, Coll & Morel 2005) over 20 multi-phase spatial shifts $\Delta \in \mathcal{S}$ (radii 1, 2, 3) with $3 \times 3$ box-filtered patch distances and noise-variance floor subtraction on GPU (CuPy) or CPU:
  $$Y_{\text{den}}(p) = \frac{Y_2(p) + \sum_{(\Delta, \gamma_\Delta) \in \mathcal{S}} w_\Delta(p) \, Y_2(p + \Delta)}{1 + \sum_{(\Delta, \gamma_\Delta) \in \mathcal{S}} w_\Delta(p)}, \qquad w_\Delta(p) = \gamma_\Delta \exp\!\left( -\frac{\max\!\left(\text{Box}_{3\times 3}\!\left((Y_2 - S_\Delta Y_2)^2\right)(p) - 1.2\sigma_d^2, \, 0\right)}{1.6\sigma_d^2} \right)$$
- **Reference Upper Bound (`denoiser="bm3d"`):** Calls collaborative 3D transform-domain empirical Wiener filtering (Dabov et al. 2007, `bm3d.bm3d`, non-commercial reference) at $\sigma_d = 1.55\sigma_n$.
- **Edge-Coherent Flat-Region Noise Floor Lock:** Smoothly blends $Y_{\text{den}}$ with the un-amplified reference luma in zero-gradient flat zones ($|\nabla Y| < 2.2\sigma_n$) while retaining sharp edge transitions where $|\nabla Y| \ge 4.5\sigma_n$, guaranteeing flat-area Immerkaer $\sigma \le 1.25\times$ `old_default + enhance`.
- **Cross-Channel Guided Chroma Filtering (Heide et al. 2013):** Filters $C_r, C_b$ non-locally using exponential weights driven jointly by $\Delta Y_{\text{clean}}^2 + 0.25(\Delta C_r^2 + \Delta C_b^2)$ so chroma edges align strictly with luminance edges.

### 2.4 Stage 4 — PSF-Adaptive Multiscale Starlet Restoration, Envelope Clamp & Colour Lock
1. **PSF-Scaled A-Trous Starlet Detail Boost:** Decomposes $Y_{\text{clean}}$ into $J=3$ undecimated $B_3$-spline wavelet detail planes $\{w_j\}_{j=0}^{2}$ (`starlet_decompose`). Using the estimated optical lens blur factor $\beta_{\text{PSF}} = \text{clip}\!\left(\frac{\sigma_{\text{lens}} - 0.18}{0.35}, 0.55, 1.65\right)$, mid-band detail layers ($j=1, 2$, corresponding to $2$ and $4$ LR pixel periods) are boosted with $2.3\sigma$ MAD noise gating and scene-adaptive flat/highlight protection (`compute_scene_adaptive_mask`), restoring optical MTF contrast without amplifying $j=0$ Nyquist noise.
2. **Local Morphological Support Deringing:** Clamps $Y_{\text{out}}$ on step edges to the $5 \times 5$ local min/max envelope $[Y_{\min} - \tau R_{\text{loc}}, Y_{\max} + \tau R_{\text{loc}}]$ ($\tau = 0.012$), preventing edge ringing or overshoot growth.
3. **Low-Frequency Colour Fidelity Lock (`anchor_low_freq_colour`):** Locks coarse low-frequency chrominance ($\sigma_{\text{LP}} = 16$ HR px) and global interior RGB channel means to `init_hr`, guaranteeing $< 0.7\%$ colour drift across all real bursts.

---

## 3. Quantitative Evaluation (Protocols A–D)

*(Full numerical tables and visual panel analysis are updated upon completion of the evaluation runs in `scripts/eval_wp14.py`.)*

## Integration note

The job stopped before its final report. Protocol A (synthetic, mean scene PSNR) shows PnP at the level of the legacy MAP (25.41 / 24.34 / 22.95 dB at lens sigma 0.35 / 0.60 / 0.90) and below WP13 (25.58 / 25.43 / 24.93 dB). The `nlm` and `bm3d` denoiser variants gave identical numbers, which suggests the prior step had no effect in this evaluation; this was not investigated further. The module stays opt-in and is not wired into `SuperResolution`.

## Panels

- [`own9_2x.png`](wp14/own9_2x.png)
- [`hdrplus_0919_2x.png`](wp14/hdrplus_0919_2x.png)
- [`hdrplus_0382_2x.png`](wp14/hdrplus_0382_2x.png)
- `samsung14_2x.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only)

Image licenses: `hdrplus_*` panels derive from the HDR+ Burst Photography Dataset (Hasinoff et al., 2016), CC-BY-SA; `samsung14_*` panels are not included in this repository (source burst: Bhat et al. 2021, deep-rep, CC BY-NC-SA 4.0, non-commercial); both shared under the same license. `own9_*` panels are the author's own burst.
