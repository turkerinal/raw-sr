# WP11: MAP Super-Resolution Colour Model Fix (`raw_sr/core/map_sr.py`)

Classical (non-neural) fix for the camera-to-sRGB colour-space mismatch and Bayer green-imbalance grid artefact in `SuperResolution(fusion="map")`.

---

## 1. Root Cause & Fix

In WP8, `x` was initialised from LibRaw's linear sRGB output (`demosaic_drizzle`), while the MAP data term compared `x` directly against WB-scaled camera-native Bayer mosaics. That pulled `x` toward camera-native RGB, causing desaturation on real bursts and a -0.6 to -6.0 dB DCB held-out penalty.

WP11 puts the exact 3x3 colour transform `C` into the forward model:

$$y_k = D \cdot W_k \cdot B \cdot (C x) + n_k$$

- **Why `C` inside the forward model (vs solving in camera space + `C^-1` at the end):** keeps `x` in output linear sRGB `[0, 1]` so projected gradient clipping does not clip out-of-gamut camera values, and luminance/chroma priors (`Y`, `R-G`, `B-G`) act directly in sRGB.
- **Exact `C` estimation (`estimate_colour_matrix`):** fits `ref_mosaic[site_c] ~ ref_rgb[site_c] @ C[c]` on interior unclipped pixels (`0.005 < rgb, mosaic < 0.90`). Across all 5 real bursts, LR fit RMS drops from `0.009-0.039` to **`0.00001`** (16-bit quantization floor), with well-conditioned `cond(C) = 1.00-2.87`.
- **Exact adjoint:** `A^T(r) = B^T(C^T * sum_k W_k^T D^T r_k)`, verified to `< 1e-12` (float64) and `< 1e-6` (float32).
- **Samsung sky grid fix (`equalize_green_imbalance`):** Samsung 14 had a 1.05% Gr/Gb green-channel gain imbalance (`Gr/Gb = 1.0105`), which created a period-2 checkerboard in flat bright sky. Equalizing the two green phases on copies of the input mosaics eliminates the grid artefact completely.
- **Re-tuned & frozen priors** (tuned on synthetic 2x `n=9` + `own9` only, then frozen for all other bursts): `btv_weight=0.10`, `chroma_weight=4.0`, `huber_k=2.0`, `max_iter=35`. Stronger chroma regularization suppresses colour-noise amplification through `C`.

---

## 2. Synthetic GT Evaluation (Before WP8 -> After WP11)

| Config | `default` | `default+wp2` | `map` (WP8) | **`map` (WP11)** | Est MTF50 | Exact `map` (WP11) |
|---|---:|---:|---:|---:|---:|---:|
| 2x, n=9 | 30.22 | 33.86 | 33.50 | **37.20 dB** (+6.98) | 0.4933 | 37.25 dB |
| 2x, n=20 | 30.33 | 34.09 | 35.67 | **38.75 dB** (+8.42) | 0.4878 | 38.89 dB |
| 3x, n=9 | 30.45 | 33.02 | 32.01 | **35.67 dB** (+5.22) | 0.4582 | 35.72 dB |
| 3x, n=20 | 30.74 | 33.11 | 34.36 | **37.76 dB** (+7.02) | 0.4945 | 37.89 dB |

### Simulated Test Chart (`scripts/eval_wp11.py chart`, 2x, 16 frames, analytical f50=0.3484 / f10=0.6234)

| Method | Star f50 (cyc/LR px) | Star f10 (cyc/LR px) | Edge MTF50 | Zoom GT PSNR | Zoom GT SSIM |
|---|---:|---:|---:|---:|---:|
| Single (bicubic) | — | — | — | 17.16 dB | — |
| `default` (`demosaic_drizzle`) | 0.3500 | 0.4556 | 0.3354 | 17.89 dB | 0.6764 |
| **`map` (WP11)** | **0.4179** | **0.6028** | **0.4242** | **21.27 dB** (+3.39) | **0.7993** |

---

## 3. Real Bursts, 2x, 1024x1024 ROI (5 Held-Out Frames)

Both DCB-space and CFA-space held-out gain (dB, mean over 5 frames). Note that `cfa_heldout.py` compares camera-space mosaics directly against sRGB HR without a 3x3 colour fit (`CFA raw`); mapping HR through `C` (`CFA cam`) puts both in the same colour space.

| Burst | DCB `def` | DCB `def+wp2` | DCB `map` old | **DCB `map` new** | CFA(cam) `def` | **CFA(cam) `map` new** | `map+wp6` DCB | `map+rr1` DCB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `own9` | +0.057 | -0.397 | -2.969 | **-0.748** | +0.358 | +0.079 | -0.907 | -0.742 |
| `hdrplus_0919` | +1.386 | +1.253 | -2.359 | **+1.533** | +1.845 | **+4.001** | +1.334 | +1.538 |
| `hdrplus_0382` | +0.748 | +0.619 | -5.313 | **+0.149** | +0.002 | **+1.751** | -0.368 | +0.142 |
| `samsung14` | +1.027 | +0.028 | -3.358 | **+0.313** | -9.159 | **-8.926** | -0.149 | +0.167 |
| `pixel4a20` | +1.581 | +1.774 | +0.976 | **+1.382** | +0.759 | **+2.342** | +0.909 | +1.380 |

---

## 4. Honest Visual Inspection (`docs/wp11/*.png`, all < 1.2 MB)

- [`own9_2x.png`](wp11/own9_2x.png): exact colour match to `default`/`default+wp2`; roof tiles and facade edges equal-or-crisper than `default`, zero halos or cast.
- [`hdrplus_0919_2x.png`](wp11/hdrplus_0919_2x.png): turquoise blind colour fully restored; horizontal louvers cleaner and sharper than `default` with zero moire.
- [`hdrplus_0382_2x.png`](wp11/hdrplus_0382_2x.png): full amber/green glass saturation restored; clean speculars and dark background, equal to `default` (slightly softer micro-contrast than `default+wp2`).
- `samsung14_2x.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only): sky period-2 grid artefact completely gone; smooth sky and crisp scaffolding struts with no ghosting.
- `pixel4a20` (internal only, not committed): "HOTEL" lettering and window mullions visibly crisper than `default` with matching blue saturation.

---

## 5. Default Decision

**Keep `fusion="map"` opt-in (default remains `demosaic_drizzle`).**

WP11 MAP passes the synthetic bar (+5.22 to +8.42 dB; +3.39 dB on simulated test chart) and the visual bar (4/4 publishable bursts equal-or-sharper with exact colour and no new artefacts), and beats `default` on `hdrplus_0919` in both metrics (+1.53 dB DCB, +4.00 dB CFA-cam). However, it fails the strict held-out criterion ("no loss > 0.1 dB in either held-out metric on any real burst"): DCB held-out trails `default` by 0.20-0.81 dB on `own9`, `0382`, `samsung14`, and `pixel4a20`, and CFA(cam) trails `default` by 0.28 dB on `own9`.

Image licenses: `hdrplus_*` panels derive from the HDR+ Burst Photography Dataset (Hasinoff et al., 2016), CC-BY-SA; `samsung14_*` panels are not included in this repository (source burst: Bhat et al. 2021, deep-rep, CC BY-NC-SA 4.0, non-commercial); both shared under the same license. `own9_*` panels are the author's own burst.
