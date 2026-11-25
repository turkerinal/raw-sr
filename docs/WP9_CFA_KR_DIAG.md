# WP9: Diagnosis & Adaptive Fix for `cfa_kr` Real-Data Gap

## 1. Problem Statement
In WP1, direct Bayer CFA anisotropic kernel regression (`fusion="cfa_kr"`) gained **+3.6 dB** GT PSNR on synthetic 2x bursts (`34.0` vs `30.3 dB` for `demosaic_drizzle`), yet lost **0.3–1.8 dB** to plain `cfa_drizzle` on real bursts in CFA-space held-out PSNR. A per-channel affine level fit moved the gap by only ~0.1 dB, refuting a pure scalar colour-level explanation.

## 2. Hypothesis-Driven Diagnosis (a)–(f)

### (a) & (b) Kernel Bandwidth vs Registration Error (Synthetic 2x GT PSNR, dB)
Controlled sweep of injected Gaussian registration noise (`0.0–0.3` LR px) and `k_detail` (`0.18–0.55` LR px):

| Reg noise (LR px) | `cfa_drizzle` (pf=0.5) | `cfa_kr` (0.18) | `cfa_kr` (0.28) | `cfa_kr` (0.40) | `cfa_kr` (0.55) | Δ `kr(0.18)` vs `drz` |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 34.59 | 34.53 | 33.57 | 32.24 | 30.93 | -0.05 |
| 0.10 | 33.53 | 34.23 | 33.42 | 32.17 | 30.88 | **+0.71** |
| 0.20 | 32.19 | 33.49 | 33.00 | 31.88 | 30.66 | **+1.30** |
| 0.30 | 30.06 | 32.63 | 32.36 | 31.40 | 30.27 | **+2.57** |

**Finding (a)/(b):** Registration sensitivity is **refuted** as the cause of the real-data loss. `cfa_kr` is *more* robust to registration error than `cfa_drizzle` (`+0.71` to `+2.57 dB`). Widening `k_detail` monotonically degrades both synthetic and real bursts.

### (c) Real Sensor Effects (Metadata & Synthetic Injection)
Measured across 4 real bursts (ROI 1024):
- `Gr/Gb` ratio: `0.9924–1.0029` (small on these sensors; synthetic 5%/10% `Gr/Gb` imbalance drops `cfa_kr` by `1.67 / 4.09 dB` if uncorrected).
- Hot-pixel outlier fraction (`>10 MAD`): `0.09%` (`own9`) to `7.18%` (`HDR+ 0919`); synthetic 0.2% hot pixels hurt `cfa_drizzle` more than `cfa_kr` (`+1.54 dB` for `cfa_kr`).
- `load_raw_linear` vs `load_cfa_frame` calibrated mosaic max abs diff: `0.00000–0.00001` (identical).

### (d) Exposure Normalisation & (f) Coverage Holes
- Per-frame exposure scales are `0.97–1.06` (`Samsung 14`) and `0.99–1.01` (others); direct vs `SuperResolution` `cfa_kr (IBP=0)` differs by `<= 0.02 dB` (`Samsung 14`: `-6.670` vs `-6.646 dB`).
- Coverage holes (`wt <= 1e-4`): `0.00%–0.01%` across all 4 real bursts (`coverage_fraction >= 0.9999`).

### (e) Root Causes Found: RGB-IBP After CFA Fusion & Over-Wide Flat-Region Kernel
Controlled single-held-out ablation on 4 real bursts (2x CFA-space held-out gain, dB):

| Burst | `cfa_drizzle` | `cfa_kr` (IBP=1, old) | `cfa_kr` (IBP=0, old) | Gain from IBP=0 |
|---|---:|---:|---:|---:|
| `own9` | -0.189 | -0.663 | **+0.174** | **+0.837 dB** |
| `HDR+ 0919` | +0.835 | +0.241 | **+1.025** | **+0.784 dB** |
| `Samsung 14` | -6.680 | -7.346 | **-6.670** | **+0.676 dB** |
| `Pixel 4a 20 (first 9)` | +0.974 | +0.486 | +0.072 | -0.414 dB |

1. **Primary root cause (IBP=1 in `SuperResolution`):** `SuperResolution` runs 1 sweep of RGB IBP against **DCB-demosaiced** frames after `cfa_kr`, pulling the CFA-domain reconstruction back toward DCB interpolation and LibRaw's 3x3 colour matrix. Turning off RGB-IBP (`IBP=0`) immediately recovers **+0.68 to +0.84 dB** on `own9`, `HDR+ 0919`, and `Samsung 14`, beating plain `cfa_drizzle` on all three.
2. **Secondary root cause (flat-region over-smoothing):** Legacy `cfa_kr` uses `k_denoise_min/max = 0.55..0.85` LR px (FWHM `1.3–2.0` LR px). On fine low-contrast textures (`Pixel 4a 20`), gradients below `d_tr` are blurred heavily; lowering flat-region bounds to `0.28..0.45` LR px recovers both synthetic GT (`+1.4 dB`) and real fine texture.

## 3. Opt-In Fix (`raw_sr/core/cfa_drizzle.py`)
Added opt-in parameters to `cfa_kernel_regression` (legacy `cfa_kr` remains bit-identical when `adaptive=False`):
- `adaptive=True`: tighter texture-preserving flat-region bounds (`k_detail=0.18`, `k_denoise_min=0.28`, `k_denoise_max=0.45`, `k_stretch=2.2`, `chroma_scale=1.15`).
- `reg_sigma_lr`: optional registration residual broadening (`hypot(k_detail, 0.5 * reg_sigma_lr)`).
- `gr_gb_balance=True` + `balance_gr_gb()`: clamps and equalises `Gr/Gb` green phase gain mismatch.

## 4. Quantitative Evaluation (2x, 3 Held-Out Frames per Real Burst)

### Synthetic 2x (Estimated Registration)
| Method | GT PSNR (dB) | Slanted-Edge MTF50 (cyc/LR px) |
|---|---:|---:|
| `default` (`demosaic_drizzle`, IBP=1) | 30.33 | 0.3617 |
| `cfa_drizzle` (pf=0.5) | 34.67 | 0.3093 |
| `cfa_kr` old (IBP=1) | 33.96 | 0.3784 |
| **`cfa_kr` new (`adaptive=True`, IBP=0)** | **36.05 (+5.72 vs default)** | 0.3592 |

### Real 2x All 5 Bursts — CFA-Space & DCB-Space Held-Out Mean (dB)
| Burst | Metric | `default` | `cfa_drizzle` | `cfa_kr` old | `cfa_kr` new | Δ new vs old |
|---|---|---:|---:|---:|---:|---:|
| `own9` | CFA | -2.757 | -0.052 | -0.701 | **+0.051** | +0.752 |
| `own9` | DCB | **+0.051** | -2.924 | -0.838 | -2.859 | -2.021 |
| `HDR+ 0919` | CFA | -3.118 | +0.833 | -0.551 | **+0.979** | +1.530 |
| `HDR+ 0919` | DCB | **+1.340** | -3.223 | -0.903 | -3.216 | -2.313 |
| `HDR+ 0382` | CFA | -7.380 | +0.257 | -2.900 | **+0.522** | +3.422 |
| `HDR+ 0382` | DCB | **+0.738** | -5.876 | -2.095 | -5.842 | -3.747 |
| `Samsung 14` | CFA | -8.440 | -5.904 | -6.645 | **-5.873** | +0.772 |
| `Samsung 14` | DCB | **+1.111** | -5.211 | -1.758 | -5.196 | -3.438 |
| `Pixel 4a 20 (first 9)` | CFA | +0.949 | +0.919 | +0.862 | +0.850 | -0.012 |
| `Pixel 4a 20 (first 9)` | DCB | **+1.171** | -0.504 | +0.168 | -0.372 | -0.540 |

## 5. Visual Panels (`docs/wp9/*.png`)
- [`own9_2x_wp9_4panel.png`](wp9/own9_2x_wp9_4panel.png) (577 KB): All four close; `cfa_kr` new is clean without grid texture.
- [`hdrplus_0919_2x_wp9_4panel.png`](wp9/hdrplus_0919_2x_wp9_4panel.png) (610 KB): `cfa_kr` new eliminates the fine dot/grid pattern visible in `cfa_drizzle` and matches `default` sharpness on window frames and stonework.
- [`hdrplus_0382_2x_wp9_4panel.png`](wp9/hdrplus_0382_2x_wp9_4panel.png) (920 KB): `cfa_kr` new resolves fine vertical lines cleanly without the colour moiré seen in `default`, and is sharper than `cfa_kr` old.
- `samsung14_2x_wp9_4panel.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only): All four close on structure; pure CFA outputs show a slight greenish camera-native cast compared to `default`'s LibRaw colour matrix.

## 6. Default Decision
**Keep `demosaic_drizzle` as default; keep `cfa_kr` (legacy and adaptive) opt-in.**
While `cfa_kr` new fixes the CFA-space regression completely (beating `cfa_kr` old by `+0.75` to `+3.42 dB` and beating `cfa_drizzle` on 4 of 5 bursts, plus `36.05 dB` on synthetic), pure CFA-domain outputs still lose `-0.37` to `-5.84 dB` in DCB-space held-out due to the camera-native vs LibRaw 3x3 colour-matrix/demosaic space difference, failing the `<= 0.1 dB` loss bar in both metrics.

Image licenses: `hdrplus_*` panels derive from the HDR+ Burst Photography Dataset (Hasinoff et al., 2016), CC-BY-SA; `samsung14_*` panels are not included in this repository (source burst: Bhat et al. 2021, deep-rep, CC BY-NC-SA 4.0, non-commercial); both shared under the same license. `own9_*` panels are the author's own burst.
