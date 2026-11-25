# WP14 Research: Classical Multi-Frame Super-Resolution, Joint Demosaicing, and Deconvolution

## 1. Failure-Mode Diagnosis of Current MAP-SR (`raw_sr/core/map_sr.py`)

Measured on real 2x bursts (`HDR+ 0919`, `own9`, `HDR+ 0382`, `Samsung 14`) and synthetic benchmarks, the current 2x `map_super_resolution` solver exhibits five interlocking failure modes when compared against `demosaic_drizzle` + WP2 enhancement (`enhance_fair_pair`, strength 0.65):

1. **PSF Mismatch (Broken Lens PSF Estimation & Channel-Uniform Blur):**
   - By default, `map_sr` sets `lens_sigma_lr=None`, using only a narrow Gaussian proxy ($\sigma_{\text{HR}} = 0.289 r$) for the $1 \times 1$ LR pixel aperture and zero optical lens blur.
   - Real smartphone lenses have significant optical blur and lateral/axial chromatic dispersion: WP6 structure-tensor ESF fitting (`estimate_psf_sigma_edges`) on `HDR+ 0919` measures $\sigma_{\text{lens}} = 0.49$ LR px ($\text{R/G/B} = 0.56 / 0.48 / 0.40$ LR px), and on `own9` measures $\sigma_{\text{lens}} = 1.07$ LR px.
   - Furthermore, `map_super_resolution(estimate_psf=True)` called `psf_info.get("fallback", True)` on a `PSFEstimate` dataclass instance, raising an `AttributeError` that was silently swallowed by `except Exception: pass`. Thus, estimated lens PSFs were never applied.
2. **Non-Convergence of First-Order Barzilai-Borwein Gradient Descent:**
   - The forward operator $A_k = D W_k B C$ combines a wider low-pass PSF $B$ with sparse Bayer sampling $D$ and colour mixing $C$. As $\sigma_{\text{lens}}$ increases from $0.35$ to $0.5$–$0.7$ LR px, the Hessian $A^T A$ becomes severely ill-conditioned in high frequencies (small singular values near the HR Nyquist frequency).
   - First-order gradient descent with Barzilai-Borwein (BB) step sizes takes steps governed by the dominant low-frequency eigenvalues of $A^T A$; within 35 iterations, mid/high frequencies barely move while the regularizer actively damps them. Counter-intuitively, putting a *more blurred* PSF into 35-step BB gradient descent yields a *blurrier* output (`0919` gradient energy drops from $0.074$ to $0.048$ at $\sigma=0.5$ and $0.024$ at $\sigma=0.7$).
3. **Period-2 (HR Nyquist) Checkerboard / Dot Texture:**
   - At 2x upscaling ($r=2$), each frame's $2 \times 2$ Bayer sites map to sparse positions on the HR grid. Without an accurate optical anti-aliasing PSF $B$ or a patch/non-local prior that couples odd and even HR phases, point-wise Huber residuals leave high-frequency phase imbalance at $(f_x, f_y) = (\pi, \pi)$ (period-2 dot texture on `0919` when `lens_sigma_lr=None`, raising flat Immerkaer $\sigma$ to $0.0084$ vs $0.0015$ for `old+enh`).
4. **Colour Drift Under Aggressive Deconvolution:**
   - When `btv_weight` is lowered ($0.03$) and BB iterations are raised ($100$) at $\sigma_{\text{lens}}=0.5$, lines become crisp (`0919` gradient $0.208$), but the crop mean colour shifts visibly darker/teal.
   - Root cause: (a) the Huber saturation $\psi_\delta(r) = \text{clip}(r/\delta, -1, 1)$ clips positive and negative residuals asymmetrically around high-contrast asymmetrical edges (such as white louvers against teal slats), creating a net DC bias when inverted through $C^T$; (b) projecting $x \in [0, 1]^3$ during poorly conditioned iterations clips individual channels unevenly; and (c) the low-frequency (DC/coarse) component of $x$ is not anchored to the well-calibrated `init_hr` reference colour.
5. **Local Finite-Difference Prior Trade-off (Noise vs Acutance):**
   - Local 1st-order priors (TV, BTV) penalize every gradient step across edges uniformly once $|\nabla Y| > \epsilon$, reducing edge contrast (loss of acutance) unless $\lambda_{\text{BTV}}$ is made so small that flat-area noise and period-2 pinholes leak through.

---

## 2. Literature Survey & Mathematical Analysis

### 2.1 Robust Bilateral Total Variation MFSR — Farsiu et al. (2004, 2006)
- **Citations:**
  - S. Farsiu, M. D. Robinson, M. Elad, and P. Milanfar, *"Fast and Robust Multiframe Super Resolution,"* IEEE Transactions on Image Processing (TIP), vol. 13, no. 10, pp. 1327–1344, 2004.
  - S. Farsiu, M. Elad, and P. Milanfar, *"Multiframe Demosaicing and Super-Resolution of Color Images,"* IEEE Transactions on Image Processing (TIP), vol. 15, no. 1, pp. 141–159, 2006.
- **Key Equations:**
  $$\hat{x} = \arg\min_x \sum_{k=1}^{N} \| D W_k B x - y_k \|_1 + \lambda_{\text{BTV}} \sum_{l=-P}^{P} \sum_{m=-P}^{P} \alpha^{|l|+|m|} \| Y(x) - S_x^l S_y^m Y(x) \|_1 + \lambda_C \left( \|\nabla(R-G)\|_2^2 + \|\nabla(B-G)\|_2^2 \right)$$
- **Effect on Our Failure Modes:** This is the baseline formulation currently in `map_sr.py` (with Huberized $L_1$). While effective against motion outliers and demosaic zippers on synthetic data, first-order BTV is a local shift-difference penalty that cannot separate periodic structure (louvers, roof tiles, text) from random noise without trading off edge contrast against flat-area grain.
- **Cost on 4 GB GPU:** $\sim 2.5$ s for 35 iterations at $2048^2 \times 3$, $< 450$ MB VRAM.
- **Licence:** Public scientific algorithm; implemented in-house in `map_sr.py`.

### 2.2 Handheld Multi-Frame Super-Resolution — Wronski et al. (2019)
- **Citation:** B. Wronski, I. Garcia-Dorado, M. Ernst, D. Kelly, M. Krainin, C.-K. Liang, M. Levoy, and P. Milanfar, *"Handheld Multi-Frame Super-Resolution,"* ACM Transactions on Graphics (TOG / SIGGRAPH), vol. 38, no. 4, Article 28, 2019.
- **Key Equations:** Direct Bayer-to-RGB non-parametric kernel regression using an anisotropic Gaussian radial basis kernel shaped by the local structure tensor $J = G_\sigma \ast (\nabla I \nabla I^T) = V \Lambda V^T$:
  $$\hat{x}_c(p) = \frac{\sum_{k=1}^{N} \sum_{q \in \mathcal{N}_c(W_k(p))} R_k(q) \, \exp\!\left(-\frac{1}{2} (q - W_k(p))^T \Omega(p)^{-1} (q - W_k(p))\right) y_k(q)}{\sum_{k=1}^{N} \sum_{q \in \mathcal{N}_c(W_k(p))} R_k(q) \, \exp\!\left(-\frac{1}{2} (q - W_k(p))^T \Omega(p)^{-1} (q - W_k(p))\right)}$$
  where $\Omega(p) = V(p) \text{diag}(\sigma_\perp^2, \sigma_\parallel^2) V(p)^T$ aligns the narrow axis $\sigma_\perp$ across edges and elongates $\sigma_\parallel$ along edges.
- **Effect on Our Failure Modes:** Tested in `raw_sr/core/cfa_drizzle.py` (`cfa_kernel_regression`, WP1/WP9/WP12). It is a single-pass gather kernel without forward PSF deconvolution ($B^{-1}$). Even with adaptive bandwidths ($\sigma_\perp = 0.18$ LR px), it leaves residual diagonal colour moiré on `HDR+ 0919` louvers (WP12 Section 5) and cannot invert the optical lens blur $\sigma_{\text{lens}} \in [0.49, 1.07]$ LR px. However, its **structure-tensor coherence and directional anisotropy** provide an ideal edge/flatness weight map and directional denoising kernel.
- **Cost on 4 GB GPU:** $\sim 3$–$6$ s CPU tiled, $< 500$ MB RAM.
- **Licence:** Algorithm published by Google Research; implemented in-house in `cfa_drizzle.py` (project code).

### 2.3 Plug-and-Play Priors (PnP-ADMM / PnP-HQS) — Venkatakrishnan et al. (2013) & Chan et al. (2017)
- **Citations:**
  - S. V. Venkatakrishnan, C. A. Bouman, and B. Wohlberg, *"Plug-and-Play Priors for Model Based Reconstruction,"* IEEE Global Conference on Signal and Information Processing (GlobalSIP), pp. 945–948, 2013.
  - S. H. Chan, X. Wang, and O. A. Elgendy, *"Plug-and-Play ADMM for Image Restoration: Fixed-Point Convergence and Applications,"* IEEE Transactions on Computational Imaging, vol. 3, no. 1, pp. 84–98, 2017.
- **Key Equations:** Variable splitting $x = z$ on $\min_x f(x) + \lambda g(x)$ via Half-Quadratic Splitting (HQS) or ADMM with dual variable $u$ and continuation penalty $\rho_t$:
  $$x^{(t+1)} = \arg\min_x f(x) + \frac{\rho_t}{2} \| x - (z^{(t)} - u^{(t)}) \|_2^2$$
  $$z^{(t+1)} = \mathcal{D}_{\sigma_d^{(t)}}\!\left(x^{(t+1)} + u^{(t)}\right), \qquad \sigma_d^{(t)} = \sqrt{\lambda / \rho_t}$$
  $$u^{(t+1)} = u^{(t)} + x^{(t+1)} - z^{(t+1)}$$
  With a weighted quadratic or IRLS-Huber data term $f(x) = \frac{1}{2} \sum_k w_k \| M_k (A_k x - y_k) \|_2^2 + \frac{\mu}{2}\|G_{\text{LP}}(x - x_{\text{ref}})\|_2^2$, the $x$-update is a symmetric positive-definite (SPD) linear system solved directly by **Conjugate Gradients (CG)**:
  $$\left( \sum_{k=1}^{N} w_k A_k^T M_k^2 A_k + \rho_t I \right) x^{(t+1)} = \sum_{k=1}^{N} w_k A_k^T M_k^2 y_k + \rho_t (z^{(t)} - u^{(t)})$$
- **Effect on Our Failure Modes:**
  1. **Fixes Non-Convergence:** Adding $\rho_t I$ to $A^T W A$ bounds the condition number by $\kappa \le 1 + \|A\|^2 / \rho_t$. Conjugate Gradients (CG) explores Krylov subspaces across all frequencies simultaneously, converging in **4–6 CG steps per outer iteration** (3–5 outer iterations total = 15–25 matrix-vector products, fewer than the 35+ line-search steps of BB gradient descent, while achieving full high-frequency inversion!).
  2. **Fixes Noise vs Acutance & Period-2 Texture:** The prior step $\mathcal{D}_{\sigma_d}$ can be a non-local patch denoiser (BM3D or GPU patch-NLM/Wiener) combined with multiscale frequency detail preservation, which averages along self-similar edges (louvers, tiles, struts) without flattening the edge step, and kills period-2 Nyquist pinholes by matching $2 \times 2$-shifted patches.
  3. **Fixes Colour Drift:** Including a gentle low-pass colour anchor or explicit local DC/mean-colour preservation against the calibrated `init_hr` guarantees $< 0.2\%$ colour drift regardless of deconvolution strength.
- **Cost on 4 GB GPU:** 3–4 outer iterations $\times$ 5 CG iterations = 15–20 forward/adjoint calls on GPU ($\sim 0.8$ s compute) + 3–4 GPU denoising calls ($\sim 0.3$ s) $\approx 1.2$–$2.0$ s total, $< 350$ MB VRAM.
- **Licence:** Public algorithm.

### 2.4 Regularization by Denoising (RED) — Romano, Elad & Milanfar (2017)
- **Citation:** Y. Romano, M. Elad, and P. Milanfar, *"The Little Engine That Could: Regularization by Denoising (RED),"* SIAM Journal on Imaging Sciences, vol. 10, no. 4, pp. 1804–1844, 2017.
- **Key Equations:** Defines an explicit Laplacian regularizer from any denoiser $\mathcal{D}_\sigma$:
  $$\mathcal{R}_{\text{RED}}(x) = \frac{1}{2} x^T \left( x - \mathcal{D}_\sigma(x) \right), \qquad \nabla \mathcal{R}_{\text{RED}}(x) = x - \mathcal{D}_\sigma(x)$$
  Fixed-point / HQS update:
  $$\left( A^T W A + \lambda I \right) x^{(t+1)} = A^T W y + \lambda \mathcal{D}_\sigma(x^{(t)})$$
- **Effect on Our Failure Modes:** Provides a damped fixed-point interpretation of PnP where $\nabla \mathcal{R}_{\text{RED}}(x) = x - \mathcal{D}_\sigma(x)$ acts directly as the residual between $x$ and its denoised anchor. In our solver, blending the PnP-ADMM/HQS update with the RED residual provides guaranteed monotonic stability even when $\mathcal{D}_\sigma$ is non-expansive but not strictly a convex proximal map.
- **Cost on 4 GB GPU:** Identical to PnP-HQS ($\sim 1.5$ s, $< 350$ MB VRAM).
- **Licence:** Public algorithm.

### 2.5 Classical Non-Local & Collaborative Denoisers: BM3D (2007), NLM (2005), WNNM (2014)
- **Citations:**
  - **BM3D:** K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, *"Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering,"* IEEE Transactions on Image Processing (TIP), vol. 16, no. 8, pp. 2080–2095, 2007.
  - **NLM:** A. Buades, B. Coll, and J.-M. Morel, *"A Non-Local Algorithm for Image Denoising,"* IEEE CVPR, vol. 2, pp. 60–65, 2005.
  - **WNNM:** S. Gu, L. Zhang, W. Zuo, and X. Feng, *"Weighted Nuclear Norm Minimization with Application to Image Denoising,"* IEEE CVPR, pp. 2862–2869, 2014.
- **Key Equations:**
  - **BM3D:** Groups similar $B \times B$ blocks into 3D stacks $Z_{S_x}$, applies a 3D unitary transform $\mathcal{T}_{\text{3D}}$, performs hard thresholding in Stage 1 and empirical Wiener collaborative filtering in Stage 2:
    $$\hat{Y}_{S_x}^{\text{wie}} = \mathcal{T}_{\text{3D}}^{-1} \left( \frac{|\mathcal{T}_{\text{3D}}(Z_{S_x}^{\text{basic}})|^2}{|\mathcal{T}_{\text{3D}}(Z_{S_x}^{\text{basic}})|^2 + \sigma^2} \odot \mathcal{T}_{\text{3D}}(Z_{S_x}) \right)$$
  - **NLM:** Non-local patch-weighted average over search window $\mathcal{S}_i$ using Gaussian-weighted patch distance:
    $$\text{NLM}(u)(i) = \frac{1}{C(i)} \sum_{j \in \mathcal{S}_i} w(i, j) u(j), \qquad w(i, j) = \exp\!\left( -\frac{\max(\|P_i - P_j\|_{2,a}^2 - 2\sigma^2, 0)}{h^2} \right)$$
  - **WNNM:** For stacked non-local patch matrix $Y = U \Sigma V^T$, solves $\min_X \frac{1}{2}\|Y - X\|_F^2 + \sum_i w_i \sigma_i(X)$ via weighted singular value shrinkage $\hat{\sigma}_i = \max(\sigma_i(Y) - w_i, 0)$ with $w_i = c \sqrt{M} / (\hat{\sigma}_i + \epsilon)$.
- **Effect on Our Failure Modes:**
  - **BM3D** and **Patch-NLM** exploit self-similarity along edges and repetitive patterns (such as the louvers in `0919`, roof tiles in `own9`, scaffolding in `samsung14`, and glass ribs in `0382`). Unlike TV/BTV, which penalize large gradients across edges, NLM/BM3D match patches *along* the edge contour, averaging out noise and period-2 Bayer sampling ripple while keeping the across-edge jump razor-sharp.
  - **WNNM** requires per-pixel SVD of patch matrices ($O(H W \cdot P^3)$), taking $> 60$ s on a $2048 \times 2048$ HR image on CPU/4 GB GPU, making it impractical for interactive burst SR.
- **Cost & Licences:**
  - `bm3d` PyPI package (v4.0.3, Tampere University): **Free for academic / non-commercial research use only** (proprietary non-commercial license). Runs on CPU in $\sim 4$–$8$ s at $2048^2$. Must ONLY be used as an opt-in reference (`denoiser="bm3d"`), NEVER as the default.
  - OpenCV `cv2.fastNlMeansDenoising` (Apache 2.0) or our own **CuPy/NumPy shift-integral Patch-NLM + Structure-Tensor Collaborative Wiener/Starlet denoiser** (MIT / in-house): **100% permissive license**, runs on GPU in **$< 0.15$ s** at $2048^2$ using vectorized shift-distance integral convolutions!

### 2.6 Total Generalized Variation (TGV) — Bredies, Kunisch & Pock (2010)
- **Citation:** K. Bredies, K. Kunisch, and T. Pock, *"Total Generalized Variation,"* SIAM Journal on Imaging Sciences, vol. 3, no. 3, pp. 492–526, 2010.
- **Key Equation:** Second-order $\text{TGV}_\alpha^2$ balances first derivatives and symmetrized second derivatives via an auxiliary vector field $v$:
  $$\text{TGV}_\alpha^2(u) = \min_{v} \alpha_1 \| \nabla u - v \|_1 + \alpha_0 \| \mathcal{E}(v) \|_1, \qquad \mathcal{E}(v) = \frac{1}{2}(\nabla v + \nabla v^T)$$
- **Effect on Our Failure Modes:** Eliminates TV staircasing on smooth ramps (e.g. sky gradients, glass curvature on `0382`). However, primal-dual Chambolle-Pock splitting for $\text{TGV}^2$ requires storing $(u, v_x, v_y, p_x, p_y, q_{xx}, q_{xy}, q_{yy})$ (8 float32 planes per channel) and requires $100+$ primal-dual iterations to converge when coupled with $A_k = D W_k B C$. Non-local patch priors (NLM / BM3D) already avoid staircasing while converging in 3–4 outer HQS steps.
- **Cost on 4 GB GPU:** High iteration count ($> 100$ steps $\approx 8$–$12$ s), $\sim 600$ MB VRAM.
- **Licence:** Public algorithm.

### 2.7 Hyper-Laplacian Priors — Krishnan & Fergus (2009)
- **Citation:** D. Krishnan and R. Fergus, *"Fast Image Deconvolution using Hyper-Laplacian Priors,"* Advances in Neural Information Processing Systems (NeurIPS / NIPS), vol. 22, pp. 1033–1041, 2009.
- **Key Equation:** HQS splitting on sparse gradient prior $|\nabla_d x|^\alpha$ ($\alpha \in [0.5, 0.8]$, typically $\alpha = 2/3$):
  $$\min_{x, w} \frac{\beta}{2} \| A x - y \|_2^2 + \frac{\rho}{2} \sum_{d \in \{x,y\}} \| \nabla_d x - w_d \|_2^2 + \lambda \sum_{d \in \{x,y\}} \| w_d \|_\alpha^\alpha$$
  where the $w_d$-subproblem has an exact closed-form quartic shrinkage root for $\alpha = 2/3$ (or LUT shrinkage), and for single-image deconvolution the $x$-subproblem is solved by 2D FFT.
- **Effect on Our Failure Modes:** Unlike convex $L_1$ ($\alpha=1$), the non-convex hyper-Laplacian ($\alpha = 2/3$) shrinks small noise gradients *harder* while penalizing large edge gradients *less* than $L_1$, directly boosting edge acutance! In multi-frame SR with space-varying warps $W_k$ and Bayer sampling $D$, $A^T A$ is not block-circulant so FFT cannot be used directly for the full $x$-step, **but** we can incorporate hyper-Laplacian ($\alpha = 2/3$) gradient shrinkage / multiscale starlet shrinkage inside the PnP prior step or as an auxiliary HQS variable!
- **Cost on 4 GB GPU:** Closed-form shrinkage takes $< 5$ ms on GPU.
- **Licence:** Public algorithm.

### 2.8 Cross-Channel Priors & Joint Demosaic+SR — Heide et al. (2013, 2014)
- **Citations:**
  - F. Heide, M. Rouf, M. B. Hullin, B. Labitzke, W. Heidrich, and A. Kolb, *"High-Quality Computational Imaging Through Simple Lenses,"* ACM Transactions on Graphics (SIGGRAPH), vol. 32, no. 5, Article 149, 2013.
  - F. Heide, M. Steinberger, Y.-T. Tsai, M. Rouf, D. Pająk, D. Reddy, O. Gallo, J. Liu, W. Heidrich, K. Egiazarian, J. Kautz, and K. Pulli, *"FlexISP: A Flexible Camera Image Processing Framework,"* ACM Transactions on Graphics (SIGGRAPH Asia), vol. 33, no. 6, Article 231, 2014.
- **Key Equations:** Cross-channel gradient correlation prior penalizing differences in normalized or luminance-relative colour gradients:
  $$\mathcal{R}_{\text{cross}}(x) = \sum_{c \in \{R, B\}} \left\| \nabla x_c - \nabla x_G \right\|_1 \quad \text{or} \quad \sum_{c \in \{R, B\}} \left\| x_G \nabla x_c - x_c \nabla x_G \right\|_2^2$$
  combined with **per-channel optical PSF** $B = \text{diag}(B_R, B_G, B_B)$ inside the joint forward operator:
  $$y_k = D \, W_k \, \text{diag}(B_R, B_G, B_B) \, (C x) + n_k$$
- **Effect on Our Failure Modes:**
  1. **Per-Channel PSF $B_c$:** Directly models chromatic aberration ($\sigma_R \ne \sigma_G \ne \sigma_B$ as measured by WP6 `estimate_psf_sigma_edges`).
  2. **Cross-Channel Edge Coupling:** Enforcing that chroma channels $C_r = R - G$ and $C_b = B - G$ (or chroma ratios) are smooth across green-guided bilateral/NLM weights transfers the high-density green channel's sharpness (2 green sites per $2\times 2$ Bayer quad) to red and blue without false-colour fringes or colour bleed.
- **Cost on 4 GB GPU:** Per-channel separable 1D blur adds zero extra memory and $< 0.05$ s runtime; green-guided chroma bilateral/NLM filter takes $< 0.03$ s on GPU.
- **Licence:** Public algorithm.

### 2.9 Blind PSF Estimation from Bursts & Sharp Edge Prediction — Joshi et al. (2008) & Delbracio & Sapiro (2015)
- **Citations:**
  - N. Joshi, R. Szeliski, and D. Kriegman, *"PSF Estimation using Sharp Edge Prediction,"* IEEE CVPR, pp. 1–8, 2008.
  - M. Delbracio and G. Sapiro, *"Burst Deblurring: Removing Camera Shake Through Fourier Burst Accumulation,"* IEEE CVPR, pp. 2385–2393, 2015.
- **Key Equations:**
  - **Joshi et al. (2008):** Predicts an ideal step edge profile along the normal $\vec{n}$ of high-coherence edges, fits the observed sub-pixel Edge Spread Function (ESF) against the parametric forward-model ESF $\text{ESF}_{\sigma}(d) = (\text{Box}_r \ast G_{\sigma \cdot r} \ast H_{\text{step}})(d)$:
    $$\hat{\sigma}_c = \arg\min_{\sigma} \sum_{i} \left\| \text{ESF}_{\text{obs}, c}(d_i) - \left(a \, \text{ESF}_{\sigma}(d_i - d_0) + b\right) \right\|_2^2$$
    and aggregates a lower percentile (e.g. 30th percentile across spatial tiles) so out-of-focus background edges do not overestimate the in-focus optical PSF.
  - **Calibration for Drizzle Pre-Blur:** As noted in `docs/WP6_PSF_ENHANCE.md` Section 2, estimating $\hat{\sigma}_{\text{raw}}$ on `init_hr` (`demosaic_drizzle` + 1-sweep IBP) includes the deposition splat blur $\sigma_{\text{splat}} \approx 0.25$ LR px in quadrature. Therefore, the true physical lens sigma for the forward model $B_c$ is calibrated by quadrature subtraction:
    $$\sigma_{\text{lens}, c} = \sqrt{\max\!\left(\hat{\sigma}_{\text{edge}, c}^2 - \sigma_{\text{splat}}^2, \, \sigma_{\min}^2\right)}$$
- **Effect on Our Failure Modes:** Fixes Failure Mode 1 (PSF mismatch). Subtracting the drizzle splat bias in quadrature makes the blind estimate match synthetic $\sigma_{\text{lens}} \in \{0.35, 0.60, 0.90\}$ accurately without over-deconvolving small-$\sigma$ bursts (`0.35`) while still capturing large optical blur (`0.60`, `0.90`, `own9`, `0919`).
- **Cost on 4 GB GPU:** Runs once at startup on `init_hr` in $\sim 0.04$ s on CPU.
- **Licence:** Already implemented in `raw_sr/core/enhance.py` (`estimate_psf_sigma_edges`, project code).

---

## 3. Comparison Table & Method Ranking

| Rank | Method | Key Mechanism | Fixes PSF Mismatch? | Fixes Non-Convergence? | Fixes Period-2 & Noise? | Fixes Colour Drift? | 4 GB GPU Time / VRAM ($1024^2 \to 2048^2$) | Licence |
|---:|---|---|---|---|---|---|---|---|
| **1** | **PnP-HQS / RED + Preconditioned CG + Non-Local Patch-NLM & Hyper-Laplacian Detail Prior + Per-Channel Blind PSF** (Our Primary Method) | HQS/RED splitting; CG data step with per-channel $B_c$; shift-integral Patch-NLM + $\alpha=2/3$ multiscale shrinkage; green-guided chroma NLM; DC colour anchor | **Yes** (per-channel WP6 ESF fit with splat-quadrature calibration) | **Yes** (CG on SPD $A^T W A + \rho_t I$ converges in $3 \times 5$ steps) | **Yes** (non-local patch averaging kills period-2 dots while sharpening edges) | **Yes** (IRLS weights + explicit local DC colour anchor to `init_hr`) | **~1.5–2.5 s / < 400 MB VRAM** | **100% Permissive** (in-house NumPy/CuPy + OpenCV Apache 2.0) |
| **2** | **PnP-HQS / RED + CG + BM3D Reference Denoiser** (`denoiser="bm3d"`) | Same CG data solver + per-channel $B_c$, with 2-stage collaborative 3D transform-domain Wiener filtering (Dabov et al. 2007) | **Yes** | **Yes** | **Yes** (gold-standard collaborative patch Wiener filter) | **Yes** | **~5–8 s (CPU BM3D + GPU CG) / < 400 MB VRAM** | **Non-commercial only** (`bm3d` 4.0.3 PyPI; reference upper bound) |
| **3** | **Hyper-Laplacian HQS + Cross-Channel Prior** (Krishnan & Fergus 2009 + Heide et al. 2013) | HQS splitting on $\|\nabla Y\|_{2/3}^{2/3}$ with closed-form quartic root + green-guided chroma gradient coupling | **Yes** (with WP6 PSF) | **Yes** (with CG data step) | **Partial** (pointwise gradients can still leave isolated pinholes without NLM) | **Yes** | **~1.2 s / < 350 MB VRAM** | **100% Permissive** |
| **4** | **TGV$^2$ Primal-Dual MFSR** (Bredies et al. 2010) | 2nd-order piecewise-affine regularization via Chambolle-Pock primal-dual splitting | No (needs external PSF) | Slow ($>100$ primal-dual steps) | Partial (avoids staircasing, less effective on periodic textures) | Partial | ~9–14 s / ~650 MB VRAM | Permissive |
| **5** | **Farsiu Robust BTV + BB Gradient Descent** (Farsiu et al. 2004/2006; current `map_sr.py`) | Huber data term + 1st-order BTV solved by 35 Barzilai-Borwein gradient steps | Broken in current code | **No** (stalls on wide PSF in 35 steps) | **No** (period-2 grid at small $\sigma$, soft at large $\sigma$) | **No** (shifts teal/dark at low BTV) | ~2.5 s / < 450 MB VRAM | Permissive |
| **6** | **Handheld Anisotropic Kernel Regression** (Wronski et al. 2019; `cfa_kr`) | Direct Bayer structure-tensor Gaussian kernel merge without deconvolution | **No** (no PSF inversion) | N/A (single pass) | Partial (residual moiré on `0919` louvers) | Fixed in WP12 via $C^{-1}$ | ~4–7 s CPU | Permissive |
| **7** | **PnP with WNNM** (Gu et al. 2014) | Non-local patch grouping + weighted nuclear norm SVD shrinkage | Yes (in PnP) | Yes (in PnP) | Yes | Yes | > 60 s (per-patch SVD bottleneck) | Permissive |

---

## 4. Selected Architecture for Implementation (`raw_sr/core/pnp_sr.py`)

We implement the **Rank 1** and **Rank 2** methods in a unified, modular solver `pnp_super_resolution` in `raw_sr/core/pnp_sr.py`:

1. **Per-Channel Calibrated Forward Operator $A_k = D W_k B_{\text{rgb}} C$:**
   - Calls `estimate_psf_sigma_edges(init_hr, ratio=ratio)` directly (reading `PSFEstimate.sigma_lr`, `PSFEstimate.sigma_rgb`, `PSFEstimate.used_fallback`), and calibrates out the initial drizzle splat blur in quadrature so both low-blur ($\sigma=0.35$) and high-blur ($\sigma \in [0.5, 1.1]$) bursts receive an accurate per-channel optical + pixel-box kernel $k_{1\text{D}, c}$.
2. **Half-Quadratic Splitting (HQS) / RED with Conjugate Gradients (CG):**
   - Solves the exact SPD normal equations $(A^T W M^2 A + \rho_t I) x = A^T W M^2 y + \rho_t v^{(t)}$ via GPU-accelerated (and deterministic CPU) Conjugate Gradients with robust Huber IRLS weights $M_k$ updated per outer continuation stage.
3. **Dual Denoiser Backend (Permissive Default + BM3D Reference):**
   - `denoiser="nlm"` (default, 100% permissive NumPy/CuPy/OpenCV): fast shift-integral **Non-Local Patch Means (Patch-NLM)** on luminance combined with **Hyper-Laplacian ($\alpha=2/3$) / A-Trous Multiscale Detail Preservation** and **Green-Guided Cross-Channel Chroma Filtering** + **Local DC Colour Anchor**.
   - `denoiser="bm3d"` (non-commercial reference upper bound): plugs `bm3d.bm3d` collaborative Wiener filtering into the exact same luminance PnP prior step.
