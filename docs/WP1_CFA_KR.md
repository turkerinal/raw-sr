# WP1: Direct Bayer CFA Anisotropic Kernel Regression (`cfa_kr`) & Exposure Gating

**Status:** Implemented on `feat/wp1` as opt-in `SuperResolution(..., fusion="cfa_kr")`. **`fusion="demosaic_drizzle"` remains the default** because Criteria 2, 3, 4 (partial), and 5 (time) did not pass. Exposure outlier gating (`max_exposure_ratio=1.5`) + per-frame exposure normalization is active by default and fixes the HDR+ mixed-exposure (`N009`) bug.

---

## 1. Method

1. **Calibrated Bayer CFA Plumbing (`raw_sr/core/superres.py`):**
   - `load_raw_linear()` extracts `raw.raw_image_visible` + `raw.raw_pattern` (phase-rolled for odd ROI origins), applies per-site black/white normalization, camera WB gains, and per-channel least-squares level calibration against LibRaw's linear RGB, attaching `mosaic` and `pattern` to `RawFrame`.
2. **Anisotropic Structure-Tensor Kernel Regression (`raw_sr/core/cfa_drizzle.py::cfa_kernel_regression`):**
   - Computes the local structure tensor on a half-resolution reference green plane (`compute_structure_tensor_precision`), producing symmetric positive-definite precision matrices `Omega^{-1}` in LR pixel^-2 coordinates: compact across edges (`k_detail=0.18` LR px), elongated along edges (`k_stretch=2.8`), and isotropic + noise-adaptive in flat regions (`k_denoise in [0.55, 0.85]` LR px).
   - Uses a `chroma_scale=1.25` wider kernel for R/B channels to suppress colour moiré on sparse Bayer lattices.
   - Tiled row-by-row (`tile_rows=256`) gather over a `3x3` LR Bayer neighborhood per frame keeps peak RSS within `+1.4%` of `demosaic_drizzle`.
   - Reuses the existing ghost gate (`_auto_robust_fusion`): when motion/parallax is detected, computes per-frame LR Tukey-biweight robustness maps; blends any low-coverage pixels smoothly with a bicubic reference fallback.
3. **Exposure Outlier Gating & Normalization (`raw_sr/core/superres.py`):**
   - Before reference selection, rejects frames whose mean green deviates `> 1.5x` from the burst median (`metrics["exposure_gate_dropped"]`), ensuring reference selection never picks an exposure outlier.
   - Normalizes kept frames whose mean green differs by `> 2%` from the reference frame.

---

## 2. Quantitative Results (Before -> After, All with Estimated Registration)

### Criterion 1 — Synthetic GT (`scripts/audit_c11_part2.py` harness, Siemens star + slanted edge) — **PASS**

| Scale | Method | MTF50 (cyc/LR px) | GT PSNR vs Sensor (dB) | GT PSNR vs Lens (dB) | Flat Noise σ | Star Chroma RMS | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 2x | `current_sr` (demosaic_drizzle, IBP=1) | 0.3617 | 30.22 | 27.06 | 0.00029 | 0.09097 | 2.24 |
| 2x | `cfa_kr` (IBP=0) | 0.3598 | **34.44 (+4.22)** | **28.58 (+1.52)** | 0.00005 | **0.04007** | 1.30 |
| 2x | `cfa_kr` (IBP=1) | **0.3784** | **33.86 (+3.64)** | **28.43 (+1.37)** | 0.00003 | 0.05308 | 1.37 |
| 4x | `current_sr` (demosaic_drizzle, IBP=1) | 0.3814 | 30.42 | 28.12 | 0.00025 | 0.10557 | 2.29 |
| 4x | `cfa_kr` (IBP=0) | 0.3721 | **35.34 (+4.92)** | **31.18 (+3.06)** | 0.00001 | **0.04353** | 3.68 |
| 4x | `cfa_kr` (IBP=1) | **0.3908** | **34.56 (+4.14)** | **30.71 (+2.59)** | 0.00000 | 0.05545 | 4.24 |

### Criterion 2 — Real-Edge MTF50 at 4x — **FAIL**

| Burst | Edge ROI (HR) | Single Bicubic | `current_sr` | `cfa_kr` (IBP=0) | `cfa_kr` (IBP=1) | Gain (IBP=1 vs current) |
|---|---|---:|---:|---:|---:|---:|
| `own9` | `[1728, 2240, 128, 128]` | 0.2204 | 0.2185 | 0.2156 | 0.1213 | **-44.5%** |
| `HDR+ 0919` | `[1984, 1856, 128, 128]` | 0.2408 | 0.2313 | 0.2304 | 0.2392 | **+3.4%** (target >= +15%) |

### Criterion 3 — Honest Held-Out Sweep (`scripts/sr_acceptance.py --heldout-sweep --fusion cfa_kr`) — **FAIL**

| Burst | Ratio | Current P1 min (dB) | `cfa_kr` P1 min (dB) | `cfa_kr` mean (dB) | Status |
|---|---|---:|---:|---:|---|
| `own9` | 2x | +0.027 | -1.009 | -0.903 | FAIL |
| `own9` | 3x | +0.017 | -1.010 | -0.901 | FAIL |
| `own9` | 4x | +0.005 | OOM/timeout | — | FAIL |
| `HDR+ 0919` | 2x | +0.932 | -1.515 | -0.774 | FAIL |
| `HDR+ 0919` | 3x | +1.526 | -1.287 | -0.601 | FAIL |
| `HDR+ 0919` | 4x | +1.534 | OOM/timeout | — | FAIL |
| `HDR+ 0382` | 2x | +0.616 | -2.256 | -2.086 | FAIL |
| `HDR+ 0382` | 3x | +0.595 | -2.217 | -2.038 | FAIL |
| `HDR+ 0382` | 4x | +0.572 | OOM/timeout | — | FAIL |
| `HDR+ 33TJ` (short) | 2x | -0.048 | -0.146 | -0.128 | FAIL |
| `HDR+ 33TJ` (short) | 3x/4x | -0.059 / -0.063 | timeout (3000s) | — | FAIL |
| `HDR+ 6G7M` (short) | 2x/3x/4x | +0.212 / +0.272 / +0.255 | timeout (3000s) | — | FAIL |

### Criterion 4 — Periodic 4x4 Grid Spectral Peak (`f = 1/4` cyc/HR px, flat patch) — **PARTIAL**

| Burst | `current_sr` | `cfa_kr` (IBP=0) | `cfa_kr` (IBP=1) |
|---|---:|---:|---:|
| `own9` | 25.27 | **4.02** | **0.81** (grid eliminated) |
| `HDR+ 0919` | 16.62 | 28.63 | 189.32 (FAIL) |

### Criterion 5 — Peak RSS & Runtime at 4x ROI 1024 (`own9`) — **RSS PASS / TIME FAIL**

| Mode | Peak RSS (MB) | Runtime (s) | vs Current |
|---|---:|---:|---|
| `demosaic_drizzle` | 2319.5 | 17.85 | baseline |
| `cfa_kr` | **2351.9** | 54.76 | **RSS +1.4% (PASS <= +30%)**, **Time 3.07x (FAIL <= 2x)** |

### Exposure Gating Test (Full 10-Frame `HDR+ 33TJ_20150810_140637_256`) — **PASS**
- `used_frames = 9 / 10`, `reference_index = 5` (`payload_N005.dng`), rejected `payload_N009.dng` (`exposure_ratio = 3.8028`).

### Criterion 6 — Full Root `pytest` — **PASS**
- `478 passed, 31 skipped, 0 failed` (including 5 new WP1 unit tests in `tests/test_cfa_kr.py`).

---

## 3. Root Causes of Failures

1. **DCB-space held-out drop (`-0.1` to `-2.2 dB`):** `load_raw_linear`'s Bayer mosaic applies per-channel WB + scalar level calibration, not LibRaw's 3x3 camera-to-sRGB colour matrix. Scoring a camera-native linear RGB reconstruction against a post-colour-matrix LibRaw DCB target penalizes the colour-matrix difference by ~1-2 dB.
2. **Real-edge MTF50 at 4x:** On real handheld bursts, lens + AA-filter PSF dominates MTF50; pure fusion without deconvolution (WP2) cannot invert optical blur, and 1-sweep IBP back-projecting against demosaiced RGB degraded `own9`'s edge MTF50.
3. **4x Runtime (`3.07x`):** Pure NumPy `3x3` gather over `9 frames x 16.7M` HR pixels takes ~40s; moving the inner tile loop to OpenCL or C/Cython is needed to beat `2x`.

---

## 4. Visual Comparisons (4-Panel 4x Crops)

- [`docs/wp1/own9_4x_wp1_4panel.png`](wp1/own9_4x_wp1_4panel.png) (743.1 KB)
- [`docs/wp1/hdrplus_0919_4x_wp1_4panel.png`](wp1/hdrplus_0919_4x_wp1_4panel.png) (691.2 KB)
