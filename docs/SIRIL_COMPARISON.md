# Comparison with Siril 1.4.5 (open-source astrophotography stacker)

Date: 2026-11-23. Scripts: `scripts/siril_compare/` (Siril via Flathub `org.siril.Siril`).

## Setup
- Siril's CLI registration is star-based only; on daytime bursts it failed (HDR+ 0919: 7 of 8 frames unregistered).
  Registration was therefore **injected** into Siril's `.seq` (H sign convention calibrated on a separate probe seed).
- Siril fusion: `seqapplyreg -drizzle -scale=2` (Bayer drizzle on undebayered CFA), `stack rej w 3 3 -nonorm -32b`.
- Every HR estimate gets the same per-channel affine level fit against the reference (kept) frame's mosaic.

## Synthetic GT (slanted edge + Siemens star, known shifts), PSNR vs sensor GT (dB)

| Method | 2x n=9 | 2x n=20 | 3x n=9 | 3x n=20 |
|---|---|---|---|---|
| single bicubic | 28.60 | 28.61 | 28.89 | 28.87 |
| raw_sr default (demosaic_drizzle) | 30.33 | 30.44 | 30.53 | 30.84 |
| raw_sr cfa_kr | 33.96 | 34.11 | 34.55 | 34.80 |
| Siril gaussian pf0.7, raw_sr registration | 31.83 | 34.03 | 33.00 | 34.39 |
| Siril gaussian pf0.7, exact registration | 34.67 | 35.15 | 35.12 | 36.18 |

Low pixfrac (0.5) in Siril leaves uncovered pixels with 9 frames (PSNR collapses to 14-20 dB).

## Real bursts, 2x, CFA-space held-out gain (dB, mean over held-out frames; min in brackets)

| Burst | default | cfa_kr | cfa_drizzle | Siril gaussian pf0.7 | Siril square pf1.0 |
|---|---|---|---|---|---|
| HDR+ 0919 (8) | -3.19 (-4.33) | -0.79 (-1.71) | **+0.99** (+0.74) | +0.39 (+0.19) | -0.02 (-3.15) |
| own 9 | -2.88 (-3.16) | -0.71 (-0.81) | -0.04 (-0.19) | **+0.17** (-0.02) | -0.60 (-2.10) |
| Samsung 14 (5 held-out) | -8.26 | -6.62 | **-5.27** | -5.35 | -5.36 |
| Pixel 4a 20 (5 held-out) | +0.79 | +0.72 | **+1.04** | +0.47 | +0.45 |

## Findings
1. With the same registration, raw_sr's own plain CFA drizzle (`raw_sr/core/cfa_drizzle.py`, pixfrac 0.5) is equal to or
   better than Siril's Bayer drizzle on 3 of 4 real bursts; Siril is ahead on own9 by 0.2 dB.
2. `cfa_kr` loses to plain `cfa_drizzle` on all real bursts (0.3-1.8 dB) while winning on synthetic data. The level fit
   changes cfa_kr by only ~0.1 dB, so the earlier "colour matrix mismatch" explanation is **refuted**; the kernel
   regression itself does not transfer to real data.
3. The default demosaic-first path is last in CFA space on 3 of 4 bursts (it is first in the DCB-space metric; the two
   metrics are biased in opposite directions, see docs/CFA_DRIZZLE.md).
4. Samsung 14: every method is far below the CFA-domain stack baseline; unexplained (frame brightness drifts +1.4 %
   over the burst; scene motion not yet checked).
5. Bug fixed: `sr_acceptance.evaluate_heldout_cfa` referenced an undefined `min_align_score` (every `--cfa-heldout`
   run raised NameError).

## Visual check (2x, `scripts/siril_compare/panels.py`, pixel-aligned < 0.05 HR px)
The CFA-space ranking does **not** match the eye. HDR+ 0919: default is cleanest; CFA/Siril drizzle leave a fine grid
texture (Siril also residual moire). Pixel 4a 20: CFA/Siril drizzle separate the perforations slightly better. own9: all
close. Samsung 14: default clearly sharpest, CFA/Siril drizzle show ghosting (likely local motion / registration).
Conclusion: keep demosaic_drizzle as default; no fusion variant gives a dramatic visible jump at 2x.

## Correction (WP7)
The synthetic "Siril with raw_sr registration" rows above were built from raw_sr's coarse phase-correlation
`shifts` (translation only), not from its final `transforms`. raw_sr's ECC homographies are already accurate to
0.022-0.029 LR px on this synthetic burst (docs/WP7_REG_REFINE.md), so the 1.2-2.9 dB "exact vs estimated
registration" gap is an artefact of this script, not a registration loss in raw_sr. The real-data comparison
(`real_cmp.py`, `panels.py`) used the full `transforms` and is unaffected.
