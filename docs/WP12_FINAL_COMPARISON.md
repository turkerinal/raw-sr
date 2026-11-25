# WP12: Final Fair Comparison & Default Fusion Decision

All CFA-domain fusions (`cfa_drizzle`, `cfa_kr`, `map`) now operate in the same output linear sRGB colour space as `default` (`demosaic_drizzle`) with Gr/Gb green-channel equalisation, and `evaluate_cfa_heldout` supports a per-burst 3x3 colour matrix `C` (`colour_matrix=C`) so every method is scored fairly in camera measurement space.

---

## 1. Changes Implemented

- **`raw_sr/core/superres.py`**:
  - Added `fusion="cfa_drizzle"` and opt-in `cfa_adaptive: bool = False` for `fusion="cfa_kr"`.
  - Applied `equalize_green_imbalance` to input mosaics and `estimate_colour_matrix` -> `C^-1` to `cfa_drizzle` and `cfa_kr` outputs (and `C` to `cfa_kr`'s `fallback_hr` before hole blending), matching `map` and `default` colour space.
  - Skipped DCB-space RGB IBP after `cfa_drizzle` and `cfa_kr` (only `demosaic_drizzle` runs RGB IBP).
- **`raw_sr/core/cfa_heldout.py`**:
  - Added optional `colour_matrix: Optional[np.ndarray] = None` to `predict_cfa_from_hr` and `evaluate_cfa_heldout` to map sRGB HR outputs into camera space before forward PSF blur and Bayer sampling. Default `None` preserves the legacy raw metric for reproducibility.
- **`tests/test_wp12_colour_fair.py`**: Constant-colour round-trip tests across all 4 fusions (`demosaic_drizzle`, `cfa_drizzle`, `cfa_kr` legacy/adaptive, `map`), Gr/Gb equalisation tests, and colour-fair vs raw CFA held-out tests.

---

## 2. Synthetic GT & Metric-Bias Check (2x & 3x)

### Synthetic GT PSNR (dB, estimated homography registration)

| Config | `default` | `default+wp2` | `default+wp6` | `cfa_drizzle` | `cfa_kr` adapt | **`map`** | `map+rr1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2x, n=9 | 30.22 | 33.86 | 30.22 | 34.64 | 35.85 | **37.20** (+6.98) | 37.20 |
| 2x, n=20 | 30.33 | 34.09 | 30.33 | 35.20 | 36.25 | **38.75** (+8.42) | 38.67 |
| 3x, n=9 | 30.45 | 33.02 | 30.45 | 33.59 | **36.63** | 35.67 (+5.22) | 35.58 |
| 3x, n=20 | 30.74 | 33.11 | 33.87 | 35.12 | 37.39 | **37.76** (+7.02) | 37.67 |

### Synthetic Metric-Bias Check (2x, n=9 fused, frame 10 held out)

Comparing both held-out metrics against known HR ground truth proves that **DCB held-out is inversely correlated with true GT accuracy** (favouring demosaic-first `default` as Rank 1 while ranking `map` last despite +7.12 dB GT gain), whereas **colour-fair CFA held-out ranks methods in the same direction as GT**:

| Method | GT PSNR (dB) | GT Rank | DCB Held-Out (dB) | DCB Rank | CFA-Fair Held-Out (dB) | CFA-Fair Rank |
|---|---:|---:|---:|---:|---:|---:|
| `map` | **37.04** | 1 | -0.996 | 6 | **+4.098** | 1 |
| `map+rr1` | 36.87 | 2 | -1.004 | 7 | +4.094 | 2 |
| `cfa_kr` adapt | 34.31 | 3 | +0.134 | 4 | +1.026 | 5 |
| `default+wp2` | 33.31 | 4 | +0.317 | 3 | +1.390 | 3 |
| `cfa_drizzle` | 32.31 | 5 | +0.079 | 5 | +1.231 | 4 |
| `default` | 29.92 | 6 | **+0.886** | 1 | -1.494 | 6 |
| `default+wp6` | 29.92 | 6 | **+0.886** | 1 | -1.494 | 6 |

---

## 3. Simulated Test Chart (WP10 Kit, 16 Frames, Zoom GT PSNR dB)

| Ratio | `default` | `default+wp2` | `default+wp6` | `cfa_drizzle` | `cfa_kr` adapt | **`map`** | `map+rr1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2x | 17.86 | 18.95 | 18.68 | 19.88 | 19.19 | **21.31** (+3.45) | 21.30 |
| 3x | 15.46 | 16.45 | 16.19 | 16.52 | 15.82 | **18.42** (+2.97) | **18.45** |

---

## 4. Real Bursts, 2x, 1024x1024 ROI (All 5 Bursts, 3 Held-Out Frames)

Siril Bayer drizzle was skipped in this run (existing `docs/SIRIL_COMPARISON.md` already established that Siril cannot register daytime bursts natively and matches `cfa_drizzle` with external alignment).

### Colour-Fair CFA Held-Out Mean (dB) [Old Raw CFA in brackets for `default` / `map`]

| Burst | `default` | `default+wp2` | `default+wp6` | `cfa_drizzle` | `cfa_kr` adapt | **`map`** | `map+rr1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `own9` | **+0.209** [-2.82] | -0.355 | -0.414 | -0.156 | -0.016 | -0.060 [-3.23] | +0.020 |
| `hdrplus_0919` | +2.135 [-3.08] | +2.355 | +2.458 | +0.570 | +0.927 | **+4.318** [-2.51] | **+4.330** |
| `hdrplus_0382` | +0.190 [-7.28] | +0.525 | +0.299 | +0.179 | +0.397 | **+1.902** [-7.25] | +1.865 |
| `samsung14` | -11.265 [-8.76] | -11.193 | -11.265 | -10.767 | -10.758 | -10.066 [-7.95] | **-8.698** |
| `pixel4a20` | +0.903 [+0.90] | +1.214 | +1.072 | +1.054 | +0.954 | **+2.610** [+2.61] | **+2.611** |

### DCB Held-Out Mean (dB)

| Burst | `default` | `default+wp2` | `default+wp6` | `cfa_drizzle` | `cfa_kr` adapt | `map` | `map+rr1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `own9` | **-0.015** | -0.422 | -0.494 | -1.024 | -0.733 | -0.797 | -0.720 |
| `hdrplus_0919` | +1.681 | +1.649 | +1.760 | -0.867 | -0.572 | **+1.795** | **+1.797** |
| `hdrplus_0382` | **+0.792** | +0.575 | +0.731 | -1.006 | -0.793 | +0.183 | +0.167 |
| `samsung14` | -3.535 | -3.559 | -3.720 | -3.587 | -3.427 | -2.012 | **+0.386** |
| `pixel4a20` | +1.528 | **+1.714** | +1.657 | +0.702 | +0.677 | +1.511 | +1.498 |

### Full-Burst Runtime (s, 2x, 1024 ROI, GPU Auto `cupy` for `map`) & Peak RSS

| Burst (N) | `default` | `default+wp2` | `cfa_drizzle` | `cfa_kr` adapt | `map` (ratio) | `map+rr1` | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `own9` (9) | 8.67 s | 9.64 s | 16.87 s | 13.44 s | 11.91 s (1.37x) | 40.14 s | 2760 MB |
| `hdrplus_0919` (8) | 6.66 s | 7.88 s | 51.80 s | 14.67 s | 12.28 s (1.84x) | 32.68 s | 3490 MB |
| `hdrplus_0382` (8) | 5.52 s | 6.55 s | 27.54 s | 13.49 s | 11.73 s (2.12x) | 38.41 s | 3490 MB |
| `samsung14` (14) | 14.50 s | 15.41 s | 105.45 s | 24.78 s | 18.15 s (1.25x) | 89.83 s | 4097 MB |
| `pixel4a20` (20) | 15.85 s | 16.84 s | 29.51 s | 29.11 s | 19.19 s (1.21x) | 104.08 s | 3350 MB |

---

## 5. Honest Visual Inspection (`docs/wp12/*.png`, all < 1.2 MB)

Panels compare `Single (bicubic) | Default (demosaic_drizzle) | CFA-KR (adaptive, colour-fixed) | MAP (WP11, colour-fixed)` at 2x nearest zoom with identical display transform:

- [`own9_2x.png`](wp12/own9_2x.png) (869 KB): Exact colour match across all three SR methods. `map` is cleanest and crisper on roof tiles and facade edges; `cfa_kr` is slightly softer.
- [`hdrplus_0919_2x.png`](wp12/hdrplus_0919_2x.png) (950 KB): `map` completely removes diagonal moiré and resolves the sharpest horizontal louvers with exact turquoise colour; `cfa_kr` still shows diagonal colour moiré bands across the louvers.
- [`hdrplus_0382_2x.png`](wp12/hdrplus_0382_2x.png) (1055 KB): Exact amber/green glass saturation across all three; `map` is crisp and clean on speculars, while `cfa_kr` is slightly softer.
- `samsung14_2x.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only): Exact blue sky colour; `map` has smooth sky (no Gr/Gb grid) and crisp scaffolding struts; `cfa_kr` shows faint residual grid texture in bright sky.
- `pixel4a20` (internal only, not committed): `map` has visibly the sharpest "HOTEL" lettering and window mullions with exact blue.

---

## 6. Mechanical Default Decision

| Candidate | (a) Synth & Chart >= +0.5 dB | (b) CFA-Fair loss <= 0.1 dB | (c) DCB loss <= 0.3 dB | (d) Panels >= 3/4 | (e) Runtime <= 3x CPU / 1.5x GPU | Decision |
|---|---|---|---|---|---|---|
| `default+wp2` | PASS (+3.64 / +1.09) | **FAIL** (`own9` -0.56) | **FAIL** (`own9` -0.41) | — | PASS (1.1x) | Opt-in |
| `default+wp6` | **FAIL** (+0.00 / +0.82) | **FAIL** (`own9` -0.62) | **FAIL** (`own9` -0.48) | — | PASS (1.1x) | Opt-in |
| `cfa_drizzle` | PASS (+4.42 / +2.02) | **FAIL** (`0919` -1.57) | **FAIL** (`0919` -2.55) | — | **FAIL** (1.9–7.8x) | Opt-in |
| `cfa_kr` adapt | PASS (+5.63 / +1.33) | **FAIL** (`0919` -1.21) | **FAIL** (`0919` -2.25) | **FAIL** (`0919` moiré) | PASS (1.6–2.4x) | Opt-in |
| **`map` (WP11)** | **PASS (+6.98 / +3.45)** | **FAIL** (`own9` -0.27; +1.2..+2.2 on 4/5) | **FAIL** (`own9` -0.78, `0382` -0.61) | **PASS (4/4)** | **FAIL** (1.84x/2.12x on 2/5) | **Opt-in** |
| `map+rr1` | PASS (+6.98 / +3.44) | **FAIL** (`own9` -0.19) | **FAIL** (`own9` -0.71, `0382` -0.63) | — | **FAIL** (4.6–7.0x) | Opt-in |

**Final Decision:** **Keep `demosaic_drizzle` as the default** for 2x, 3x, and 4x. No candidate passes all five mechanical criteria simultaneously. `fusion="map"` (WP11) is the clear best opt-in classical fusion (winning synthetic GT by +5.2..+8.4 dB, test chart by +3.0..+3.5 dB, colour-fair CFA held-out on 4/5 real bursts by +1.2..+2.2 dB, and 4/4 visual panels), failing only on `own9` CFA-fair (-0.27 dB), DCB held-out on `own9`/`0382`, and GPU runtime ratio on 8-frame bursts (1.84x–2.12x).

Image licenses: `hdrplus_*` panels derive from the HDR+ Burst Photography Dataset (Hasinoff et al., 2016), CC-BY-SA; `samsung14_*` panels are not included in this repository (source burst: Bhat et al. 2021, deep-rep, CC BY-NC-SA 4.0, non-commercial); both shared under the same license. `own9_*` panels are the author's own burst.
