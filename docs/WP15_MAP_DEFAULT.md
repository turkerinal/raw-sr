# WP15: Robust PSF, Residual-Adaptive Deringing, and 2× Default MAP

WP15 resolves the remaining failure modes of `map_mode="wp13"` (parallax-inflated PSF estimation, ringing specks on thin struts, specular overshoot, and box-only CFA held-out bias) and promotes `map_mode="wp13"` to the default MAP mode used by `fusion="auto"` at 2× on CUDA GPUs. `map_mode="legacy"` remains selectable, and CPU runs stay on `demosaic_drizzle` under `fusion="auto"`.

---

## 1. Changes

1. **Single-Frame + Drizzle Min PSF (`estimate_burst_psf_rgb` in `raw_sr/core/map_sr.py`):**
   - `SuperResolution` passes the demosaiced reference LR frame (`ref_rgb`) alongside `init_hr`.
   - Slanted-edge PSF estimation runs on both `ref_rgb` (`ratio=1.0`, subtracting `0.40` LR px 1× demosaic quadrature bias + common R/B excess) and `init_hr` (`ratio`, subtracting `0.10` LR px drizzle bias + common R/B excess).
   - Requires at least `min_edges=15` clean edges for reliability; edge-starved/parallax bursts such as `samsung14` (`5` ref edges, `10` HR edges) fall back cleanly to `0.35` LR px instead of inflating to `0.90`.
   - When both estimates are reliable, takes the per-channel minimum `min(ref, hr)` so neither parallax/drizzle smear nor 1× demosaic blur can inflate the kernel.

2. **Residual-Adaptive Local Envelope (`_compute_local_envelope` in `raw_sr/core/map_sr.py`):**
   - Computes the per-HR-pixel inter-frame standardized residual std `mismatch_hr = sqrt(max(E[u²] - E[u]², 0))` in one adjoint pass during solver setup (static sharp edges have consistent sign across frames so `mismatch_hr ≈ 1`; parallax/misregistration produces opposing signs and large `mismatch_hr`).
   - Combines `mis_risk = clip((smooth(mismatch_hr) - 2.0) / 2.0, 0, 1)` with near-clipped highlight risk `hi_risk = clip((max_c(local_max) - 0.72) / 0.20, 0, 1)`.
   - Blends the Projected L-BFGS feasible box between `[0, 1]` (clean static pixels) and `[local_min - m, local_max + m]` (`env_tol=0.08`, `env_radius=2`, `m = env_tol * range + 2*sigma_med`) on mismatched/specular pixels, keeping the problem strictly convex.

3. **Optional Per-Burst Lens PSF in `evaluate_cfa_heldout` (`raw_sr/core/cfa_heldout.py`):**
   - Added optional `lens_sigma_lr` (scalar or 3-tuple of LR px, default `None` unchanged) to `predict_cfa_from_hr` and `evaluate_cfa_heldout`, combining sensor box PSF and optical lens PSF in quadrature per channel with full `2 * ceil(3 * sigma) + 1` kernel support so every method is scored under the same burst PSF.

---

## 2. Synthetic GT (Protocol A, Mean over `n ∈ {9, 20}`, Seeds `{7, 23}`)

| `sigma_lens` | Single | Old + Enh | WP13 MAP | **WP15 MAP** | WP15 vs Old+Enh | MTF50 (Old+Enh -> WP15) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 22.97 | 24.57 | 25.59 | **25.58 dB** | **+1.01 dB** | 0.598 -> **0.833** |
| 0.60 | 22.58 | 24.04 | 25.42 | **25.45 dB** | **+1.41 dB** | 0.436 -> **0.753** |
| 0.90 | 21.81 | 23.05 | 24.93 | **24.93 dB** | **+1.88 dB** | 0.308 -> **0.583** |

---

## 3. Real Bursts (2×, 1024×1024 ROI, 3 Held-Out Frames)

### Per-Burst PSF Estimates (LR px: `ref` / `hr` -> `WP15`)

| Burst | Single-Frame `ref` | Drizzle `hr` | WP13 | **WP15** (`source`) |
|---|---|---|---|---|
| `own9` | (.90, .90, .90) | (.90, .90, .90) | (.90, .90, .90) | **(.90, .90, .90)** (`wp15_ref_hr_min`) |
| `hdrplus_0919` | (.59, .50, .47) | (.52, .49, .41) | (.52, .49, .41) | **(.52, .49, .41)** (`wp15_ref_hr_min`) |
| `hdrplus_0382` | (.87, .90, .43) | (.55, .25, .23) | (.40, .25, .23) | **(.40, .25, .23)** (`wp15_ref_hr_min`) |
| `samsung14` | (1.27, 1.19, 1.35) [5 edges] | (.45, .90, .90) [10 edges] | (.75, .90, .90) | **(.35, .35, .35)** (`wp15_low_edge_fallback`) |
| `pixel4a20` (int.) | (.38, .28, .28) | (.37, .29, .29) | (.37, .29, .29) | **(.37, .28, .28)** (`wp15_ref_hr_min`) |

### Crop Metrics & Held-Out CFA Gain (`Old+Enh` / `WP13` -> **`WP15`**)

| Burst | Grad Energy | Flat `sigma` | Overshoot (vs Old+Enh) | Drift% | CFA-Box (dB) | CFA-PSF (dB) |
|---|---|---|---|---|---|---|
| `own9` | .0080/.0086 -> **.0086** | .000107 -> **.000010** (0.09×) | .424/.550 -> **.550** (1.30×) | **0.17%** | -0.31 / -0.97 -> -0.95 | +0.09 / +0.29 -> **+0.28** |
| `hdrplus_0919` | .0739/.1375 -> **.1373** | .000127 -> **.000052** (0.41×) | .106/.107 -> **.108** (1.02×) | **0.45%** | +2.36 / +0.76 -> +1.04 | +0.62 / +3.08 -> **+3.00** |
| `hdrplus_0382` | .0226/.0280 -> **.0262** | .000185 -> **.000107** (0.58×) | 2.02/3.37 -> **2.03** (1.01×) | **0.44%** | +0.50 / +0.50 -> **+1.33** | -0.17 / +1.47 -> **+0.90** |
| `samsung14` | .0215/.0318 -> **.0214** | .000159 -> **.000060** (0.38×) | .840/1.31 -> **.730** (0.87×) | **0.88%** | -11.19 / -10.77 -> **-10.69** | -11.18 / -10.33 -> **-10.72** |
| `pixel4a20` (int.) | .0323/.0414 -> **.0409** | .000175 -> **.000102** (0.58×) | .909/.830 -> **.866** (0.95×) | **0.63%** | +1.20 / +2.93 -> **+2.86** | +1.15 / +2.55 -> **+2.43** |

---

## 4. Visual Inspection (`docs/wp15/*.png`)

- [`own9_2x.png`](wp15/own9_2x.png): Visibly sharper diagonal shadow and cleaner horizontal siding than `old+enh`, zero grain, exact colour (0.17% drift).
- [`hdrplus_0919_2x.png`](wp15/hdrplus_0919_2x.png): Dramatically crisper turquoise louvers than `old+enh`, period-2 grid gone, matches WP13 sharpness with no halos (0.45% drift).
- [`hdrplus_0382_2x.png`](wp15/hdrplus_0382_2x.png): Sharper amber glass stem and cleaner dark background than `old+enh`, with specular overshoot tamed to 1.01× `old+enh` (down from 1.67× in WP13).
- `samsung14_2x.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only): All WP13 black/white ringing specks on the thin vertical scaffold struts are completely eliminated; cleaner sky; sharpness of the lattice bars is about equal to `old+enh`, not clearly better (0.88% drift, 0.87× overshoot). Too few clean edges were found on this burst, so the PSF falls back to 0.35 LR px.

Image licenses: `hdrplus_*` panels derive from the HDR+ Burst Photography Dataset (Hasinoff et al., 2016), CC-BY-SA; `samsung14_*` panels are not included in this repository (source burst: Bhat et al. 2021, deep-rep, CC BY-NC-SA 4.0, non-commercial); both shared under the same license. `own9_*` panels are the author's own burst.
