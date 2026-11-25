# Acceptance verdict: what PASS/FAIL/N-A means here (2026-11-20)

`scripts/sr_acceptance.py` measures the SR core on a real burst and prints
numbers. Numbers alone decide nothing: the SPEC's thresholds have to be applied,
and a criterion whose data does not exist has to be declared missing rather than
quietly dropped. This page documents the explicit verdict added on
`fix/acc19` — the criteria table, the measurement behind each row, the honest
substitutes, and the measured result on the real 9-frame burst.

Owner area: `scripts/acceptance_verdict.py` (new), the verdict section of
`scripts/sr_acceptance.py`, `tests/test_acceptance_verdict.py`, this page. No
file under `raw_sr/` is touched; the core is `master`'s.

```bash
venv/bin/python scripts/sr_acceptance.py --ratio 2 --heldout-sweep \
    --out-suffix _C1                      # verdict printed + written to JSON
venv/bin/python scripts/acceptance_verdict.py output/sr_acceptance_2x_C1.json
venv/bin/python scripts/acceptance_verdict.py --strict output/*_C1.json
```

`--no-verdict` skips the section, `--no-self-checks` skips the three synthetic
checks (a few seconds), `--hf-all-frames N` limits the per-frame HF sweep.
`acceptance_verdict.py --strict` exits 1 when any row FAILs, so CI can gate on it.

---

## 1. The criteria table

Every row cites the SPEC line it comes from. §7 is the 0.2.0 acceptance bullet
list, §11 the 1.0.0 list; `P1` is the project's own truth-free proof (not SPEC)
and is labelled as such in the JSON.

| ID | Criterion | Threshold | Source |
|---|---|---|---|
| `S1_snr_flat_15f` | SNR improvement on a flat surface, 15+ frames | ≥ 3 dB (SPEC 3-5 dB band) | §7 L219-220, §11 L307-308 |
| `S2_hf_vs_any_single` | Genuine HF gain vs **any** single input frame | min over frames > 0 dB | §11 L309-310 |
| `S3_time_roi500` | ~500×500 source ROI processing time | ≤ 60 s | §11 L311-313 |
| `S4_parallax_no_crash` | Parallax region: no crash, low confidence indicated/warned | no exception + non-empty reason | §11 L314-317 |
| `S5_alignment_heatmap` | Heatmap reflects excluded/low-confidence regions | map marks the region low | §7 L221 |
| `S6_low_coverage_warning` | Warning below ~20 % coverage, never aborts | warning fires, run completes | §7 L218 |
| `S7_ratio_reported` | Requested vs achievable SR ratio reported | both numbers surfaced | §7 L213 |
| `S8_threshold_scaling` | Residual threshold scales with ratio (stricter) | t(1x) > t(2x) > t(4x) > 0 | §7 L211-212 |
| `P1_heldout_gain` | Held-out PSNR gain vs the denoised LR stack | min gain > 0 dB | project (`state.md`, `docs/IBP_AND_HELDOUT.md`) |

Status vocabulary:

* **PASS** — the measured value meets the threshold, and the measurement is the
  one the criterion names.
* **FAIL** — it does not. A FAIL is never rounded away and never retried.
* **N/A** — the data this criterion needs does not exist for this run. The JSON
  row still carries whatever *was* measurable (`value`, `detail`,
  `substitute`), so the gap is visible instead of hidden.

The overall verdict is FAIL if any row FAILs, `INCOMPLETE` if rows are only
N/A, PASS otherwise. Counts are printed (`verdict: FAIL (5 PASS, 1 FAIL, 3 N/A)`).

---

## 2. How each row is measured

### S1 — flat surface, 15+ frames (the row that cannot pass here)

The SPEC asks for a flat test target (printed document) captured 15+ times. The
available data is **9 real frames of an ordinary outdoor scene** — no flat
target, no 15-frame burst. So the row is **N/A**, and the JSON records a
*substitute* with its method written out:

* locate the flattest `64×64` LR block of the reference frame (lowest high-pass
  σ) — for this burst it is the quiet sky patch at LR `(960, 560)`;
* read both images at **input-pixel pitch** (box-average the HR images by the
  ratio, i.e. one sample per sensor pixel — the pitch at which "average N
  frames" is supposed to reduce noise) and report
  `20·log10(σ_single / σ_SR)` on that block;
* project the 9-frame number to 15 frames with `20·log10(√(15/9))` and say so.

The projection is arithmetic, not a measurement: it assumes the fusion keeps
gaining as `√N`, which is optimistic for a handheld burst (registration error
and phase diversity limit it before noise does). The row stays N/A either way.

Why input pitch and not HR pitch: at HR pitch the single-frame baseline is a
**bicubic upscale**, which smooths its noise, so the baseline looks quieter as
the ratio grows (−3.6 / −0.7 / +1.1 dB at 2×/3×/4× for the *same* fusion). That
number measures the upsampler, not the SR. The LR-pitch value is stable across
ratios (+3.34 / +3.32 / +3.23 dB), which is what a fusion gain should be. Both
are stored; `snr_gain_db_hr_pitch` is the diagnostic.

**What this substitute cannot claim:** it is one region of one burst, at sensor
noise level, with no flat-target capture chain (no lens shading, no print
texture, no exposure stack). A PASS on S1 still needs the 15-frame flat capture.

### S2 — HF gain vs *any* single frame

`hf_gain_db_vs_single` in the old JSON is measured against the **reference**
frame, and the reference is chosen as the *sharpest* frame near the burst
centroid — the most favourable single frame. The SPEC says "any single input
frame", so the verdict measures the HF band (above the LR Nyquist, display-
transformed view) against **every** input frame, bicubic-upscaled, and takes the
**minimum**. Both the min and the range go into the JSON
(`hf_gain_db_vs_single_frames`, `..._scope`).

### S3 — ~500×500 ROI time

Only decided when the run's ROI is at most ~560 px on a side; a 1024 ROI run
reports its own wall time as N/A with the reason. The dedicated ROI-512 run
supplies the decisive number. The timer is the SR call (`metrics.seconds`), not
the script's I/O (DNG decode + exports are outside it); parked as documented.

### S4 / S5 / S6 — synthetic self-checks (no real data needed)

These three are properties of the code, so they are exercised with small
synthetic bursts inside the acceptance run and stored under `self_checks`, which
means an archived JSON reproduces the same verdict without re-running anything.

* **S4 parallax**: 5 frames, a foreground patch that moves at 1.9× the
  background rate (a genuine parallax layer). PASS needs no exception, finite
  output, and a non-empty `robust_reason` (the user-facing indication).
  Measured: no crash, `robust_active=True`,
  `"moving subject detected (population share 0.176 of frame, blob 1.00/0.93)"`.
* **S5 heatmap**: one frame gets a +0.25 linear level step inside a 40 % × 40 %
  rectangle; the **shipped** `desktop.backend.server._alignment_quality_map` is
  called and its PNG decoded. PASS needs the TURBO-red level inside that
  rectangle to read clearly below the rest (bar: < 0.8×). Measured: 0.38×
  (60 vs 156) with 9/9 frames — the map marks the disagreeing region.
  A noise rectangle is deliberately *not* used: it fails registration, the gate
  drops that frame, and a one-frame stack has zero std, i.e. a uniform
  (uninformative) map — measured ratio 1.00.
* **S6 low coverage**: 1 sharp + 8 blurred frames. The blur gate keeps 1 of 9
  (11 % < 20 %), the backend's `low_coverage_warning` condition is true, and the
  run completes. PASS needs the warning *and* completion; an exception is a FAIL.

### S7 / S8 — reporting and threshold law

`S7` runs the shipped `SRRatioAnalyzer.analyze(shifts, requested)` on the run's
own measured shifts and requires `requested`, `achievable` and a message. It is
a *reporting* criterion: it does not claim the achievable number is right.
`S8` runs `compute_residual_threshold` at 1×/2×/4× on one residual set and
requires the threshold to shrink strictly (`(median + k·MAD)/√ratio`).

### P1 — held-out gain

Read from the existing sweep (`heldout_gain_min_db` / `_mean_db`), the honest
metric documented in `docs/IBP_AND_HELDOUT.md`. Threshold **> 0 dB on the
minimum**, so one unlucky frame cannot be averaged away. This is the project's
own criterion, not the SPEC's.

---

## 3. Measured verdict on `master` (2026-11-22, `ec7cb6b`)

9-DNG burst, ROI 1024×1024 (`1650,1240`), homography, 9/9 frames, 1 IBP sweep
with ratio-scaled VST edge-preserving MAP-IBP prox (`alpha=0.75`),
`drizzle_low_cov_thresh=0.0`, `--heldout-sweep`, suffix `_C5final` (see
`docs/agent-reports/2026-11-22-C5-final.md`).

| ID | 2× (ROI 1024) | 3× (ROI 1024) | 4× (ROI 1024) |
|---|---|---|---|
| S1 flat SNR (substitute, N/A) | N/A (9f burst, no flat target) | N/A | N/A |
| S2 min HF vs any frame | **PASS +4.65 dB** | **PASS +14.33 dB** | **PASS +18.12 dB** |
| S3 time (≤ 60 s, ROI 512) | N/A 6.96 s (ROI 1024) | N/A 10.16 s | N/A 19.18 s (standalone 13.94 s) |
| S4 parallax | **PASS** no crash, reason set | **PASS** | **PASS** |
| S5 heatmap | **PASS** | **PASS** | **PASS** |
| S6 low coverage | **PASS** 1/9, warning, no abort | **PASS** | **PASS** |
| S7 ratio reported | **PASS** | **PASS** | **PASS** |
| S8 threshold scaling | **PASS** 0.200/0.141/0.100 px | **PASS** | **PASS** |
| P1 held-out min gain | **PASS +0.0266 dB** (mean +0.0519) | **PASS +0.0167 dB** (mean +0.0460) | **PASS +0.0045 dB** (mean +0.0380) |

**Verdict: PASS** (`7 PASS, 0 FAIL, 2 N/A`) at every ratio (2×/3×/4×). All
27/27 held-out frames are strictly positive on the honest masked,
coverage-normalised metric, and detail vs single frame stays within the
`<= 0.3 dB` budget (`4.934 / 14.574 / 18.418 dB`, Δ `-0.157 / -0.163 / -0.163 dB`
vs `826bc65`). Peak RSS: `1093 / 1575 / 2281 MiB`. Full root `pytest -q`:
**504 passed, 0 failed**.

### Historical baseline: `master` before `integ/C1` (2026-11-20, `826bc65`)

9-DNG burst, ROI 1024×1024 (`1650,1240`), homography, 9/9 frames, 1 IBP sweep,
`--heldout-sweep`, suffix `_C1`; plus one ROI 512×512 run for the timing row.
Numbers in `output/sr_acceptance_{2,3,4}x_C1.json` and
`output/sr_acceptance_2x_C1roi512.json`.

| ID | 2× (ROI 1024) | 3× | 4× | ROI 512 / 2× |
|---|---|---|---|---|
| S1 flat SNR (substitute, N/A) | +3.34 dB → +5.56 @15f | +3.32 → +5.54 | +3.23 → +5.45 | +7.91 → +10.1 |
| S2 min HF vs any frame | **PASS +4.87 dB** | **PASS +14.55 dB** | **PASS +18.34 dB** | **PASS +4.10 dB** |
| S3 time (≤ 60 s) | N/A 6.6 s | N/A 9.3 s | N/A 13.1 s | **PASS 2.50 s** |
| S4 parallax | **PASS** no crash, reason set | PASS | PASS | PASS |
| S5 heatmap | **PASS** 0.38× | **PASS** 0.35× | **PASS** 0.32× | **PASS** 0.38× |
| S6 low coverage | **PASS** 1/9, warning, no abort | PASS | PASS | PASS |
| S7 ratio reported | **PASS** 2.00× requested | **PASS** 1.33× achievable | **PASS** 1.75× | **PASS** 1.50× |
| S8 threshold scaling | **PASS** 0.200/0.141/0.100 px | PASS | PASS | PASS |
| P1 held-out min gain | **FAIL −0.006 dB** | **FAIL −0.046 dB** | **FAIL −0.051 dB** | **FAIL −0.05 dB** |

At `826bc65`, the verdict was **FAIL** on `P1_heldout_gain` alone. Merging
`integ/C1` (`ec7cb6b`) flipped `P1` to **PASS** at all three ratios without
changing the held-out evaluator, masks, or baseline.

---

## 4. Remaining N/A rows

1. **S1**: capture the SPEC's flat target 15+ times (tripod, printed chart or
   wall), run the same harness with `--roi-size` on the clean centre. The
   substitute predicts ≈ +5.5 dB there; only a measurement proves it.
2. **S3**: decided on the dedicated ROI-512 run (`2.50 s ≤ 60 s`, PASS); ROI-1024
   runs report N/A by design.
3. **S5** stays N/A on a machine without `fastapi` (the desktop backend cannot
   be imported); it is a PASS wherever `requirements-dev.txt` is installed.
