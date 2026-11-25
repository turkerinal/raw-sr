# WP5: Tile-Based Local Alignment and Per-Tile Frame Selection

## 1. Implementation (`raw_sr/core/local_align.py`)

Opt-in via `SuperResolution(local_align=True, local_tile_size=48, local_max_shift=2.0, local_select=None, local_robust=True)`:

1. **Coarse-to-fine tile residual alignment:** After global homography `W` (`ref -> mov`), each frame's green channel is warped into reference coordinates (`cv2.INTER_CUBIC`). Overlapping tiles (`48×48` LR px, 50% overlap, 2-level pyramid) estimate residual sub-pixel displacements using windowed phase correlation (`cv2.phaseCorrelate`) refined by 3-iteration Tikhonov-damped Lucas-Kanade Gauss-Newton. Tiles with low texture (below `2.5 * noise_sigma`), low overlap, or no residual improvement (`rms_after > 0.97 * rms_before`, or `< 0.04` LR px) fall back to global `(0, 0)`. Confidence-weighted spatial smoothing and magnitude clamping (`2.0` LR px) produce a smooth dense field upsampled at true tile centers and warped to `mov` coordinates.
2. **Per-tile robustness and lucky selection:** Post-local-alignment residuals per tile are compared against an edge-aware tolerance (`2.5 * noise_sigma + 0.35 * mean_grad`) with global median brightness normalization to produce Wronski-style weights (suppressing moving objects). Optional `local_select` (e.g. `0.7`) keeps the best `X%` frames per tile (AutoStakkert-style, always retaining the reference frame).
3. **Deposition integration:** Both `demosaic_drizzle` (`_scatter_fast` / `_scatter_accumulate`) and `cfa_drizzle` (`_deposit_plane`) accept per-frame `(local_du, local_dv)` and per-pixel `local_weights`, depositing each sample at `global H + local residual`.

---

## 2. Synthetic Evaluation (2x, `lr=256`, `n=9`, `scripts/eval_wp5.py synth`)

Extended `synthesize_target_burst` with a smooth parallax-like non-rigid warp (`0.35–0.80` LR px amplitude) plus a moving bright patch in 3 of 9 frames.

| Burst | Method | GT PSNR (dB) | MTF50 (cyc/LR px) | Ghost Energy | Moving Patch RMSE |
|---|---|---:|---:|---:|---:|
| Plain rigid | `master` | 30.223 | 0.3619 | 0.001555 | — |
| Plain rigid | `local_align` | **30.322** (+0.10) | **0.3622** | 0.001555 | — |
| Plain rigid | `local_select=0.7` | 30.003 (-0.22) | 0.3613 | 0.001555 | — |
| Plain rigid | `master` + `cfa_drizzle` | 34.642 | 0.3093 | — | — |
| Plain rigid | `local_align` + `cfa_drizzle` | 34.336 (-0.31) | 0.3095 | — | — |
| Non-rigid + moving | `master` | 28.449 | 0.1365 | 0.004644 | 0.10812 |
| Non-rigid + moving | `local_align` | **29.958** (+1.51) | **0.3406** | 0.004644 | **0.03559** (-67%) |
| Non-rigid + moving | `local_select=0.7` | 29.772 (+1.32) | 0.3401 | 0.004644 | **0.03522** (-67%) |
| Non-rigid + moving | `master` + `cfa_drizzle` | 28.682 | 0.1034 | — | 0.14523 |
| Non-rigid + moving | `local_align` + `cfa_drizzle` | **32.851** (+4.17) | **0.3010** | — | **0.04324** (-70%) |

---

## 3. Real Bursts, 2x, 1024×1024 ROI (`scripts/eval_wp5.py real`)

Mean held-out gain in dB (min in brackets). `0919` (8) and `own9` (9) sweep all frames; `samsung14` (14), `pixel4a20` (20), and `pixel4a9` (9) use 5 evenly spaced held-out frames.

| Burst | DCB `master` | DCB `local_align` | DCB `local_sel70` | CFA `master_def` | CFA `local_def` | CFA `master_cfa` | CFA `local_cfa` |
|---|---:|---:|---:|---:|---:|---:|---:|
| HDR+ 0919 (8) | **+1.441** (+0.932) | +1.383 (**+0.992**) | — | **-3.188** (-4.333) | -3.224 (-4.366) | **+0.986** (+0.740) | +0.791 (+0.475) |
| own9 (9) | **+0.052** (+0.027) | -0.128 (-0.695) | — | **-2.877** (-3.162) | -2.996 (-3.580) | **-0.038** (-0.190) | -0.334 (-1.149) |
| Samsung 14 (5/14) | **+1.021** (+0.619) | +1.015 (+0.653) | +1.018 (**+0.687**) | **-8.258** (-8.940) | -8.260 (-8.947) | **-5.271** (-5.854) | -5.307 (-5.899) |
| Pixel 4a 20 (5/20) | **+1.581** (+0.645) | +1.558 (+0.648) | +1.570 (**+0.788**) | **+0.788** (+0.207) | +0.737 (+0.226) | **+1.043** (+0.711) | +0.924 (+0.582) |
| Pixel 4a first 9 (5/9) | **+1.186** (+0.962) | +1.138 (+1.003) | — | **+0.815** (+0.451) | +0.753 (+0.331) | **+0.850** (+0.598) | +0.774 (+0.613) |

**Key questions:**
- **Samsung 14:** Local alignment does **not** close the `-5.3` / `-8.3 dB` CFA-space gap (`-5.271 -> -5.307 dB`); the gap is not caused by local misalignment.
- **20 vs 9 frames (Pixel 4a):** On the 5-frame held-out sweep, 20 frames beat 9 frames on DCB mean (`+1.581` vs `+1.186 dB` master; `+1.558` vs `+1.138 dB` local), and `local_select=0.7` raises the worst held-out frame from `+0.645` to `+0.788 dB`, though `local_align` does not beat `master` on mean gain.

---

## 4. Runtime and Memory (`scripts/eval_wp5.py bench20`, 1024 ROI, 2x, 20 frames)

- `master`: **14.60 s**, peak RSS **1775.4 MiB**
- `local_align=True`: **43.74 s** (`3.0×`, of which `23.20 s` in local alignment), peak RSS **1954.8 MiB** (`+179.4 MiB` / `+10.1%`)

---

## 5. Visual Panels (`docs/wp5/*.png`, pixel-aligned 2×2, all `< 1.2 MB`)

- [`hdrplus_0919_2x.png`](wp5/hdrplus_0919_2x.png) (1.07 MB): `master` and `local_align` both eliminate the single-frame moiré and resolve clean louvers; visually tied (CFA drizzle shows a fine grid pattern).
- [`own9_2x.png`](wp5/own9_2x.png) (1.07 MB): Auto-selected crop contains a walking person. `master` smears the person into a dotted ghost trail; `local_align` suppresses the ghost trail cleanly. `local_align + cfa_drizzle` shows colour zippering on the single-frame-only moving person.
- `samsung14_2x.png` (panel not included: the Samsung 14 burst is CC BY-NC-SA 4.0 (non-commercial); the numbers above are measurements only): Scaffolding against sky; `master` and `local_align` are sharp and visually indistinguishable without double edges.
- `pixel4a20_2x.png` (not published: source burst has no stated license): Perforated facade; `master` and `local_align` both remove rainbow moiré cleanly and look visually identical; `local_align + cfa_drizzle` separates perforations slightly more crisply.

---

## 6. Default Decision

**Keep `local_align=False` (opt-in).** While `local_align=True` delivers strong gains on synthetic non-rigid parallax + moving-object bursts (`+1.51` / `+4.17 dB`, `3×` less ghosting) and removes the moving-person ghost trail on `own9`, it does not win on held-out metrics on the real bursts (regressing `-0.18 dB` mean on `own9`, exceeding the `-0.1 dB` threshold) and adds `3×` runtime overhead. It remains available opt-in for scenes with parallax or moving subjects.

Image licenses: `wp5/hdrplus_0919_2x.png` HDR+ (Hasinoff et al., 2016) CC-BY-SA; `wp5/samsung14_2x.png` Bhat et al. 2021 (deep-rep) CC BY-NC-SA 4.0, non-commercial; both shared under the same license.
