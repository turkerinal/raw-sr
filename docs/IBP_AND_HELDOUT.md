# IBP strength and the held-out proof (B2, 2026-11-19)

This page records **why** the shipped defaults are what they are, **what they cost**, and **how to
tell two runs apart**. It accompanies `fix/heldout` (metric geometry) and `fix/heldout2` (this page,
wording, reporting). It intentionally does not restate `README.md`.

Owner area: `raw_sr/core/superres.py` (`iterative_back_projection`, `predict_lr_from_hr`,
`SENSOR_PSF_SIGMA_PER_RATIO`), `scripts/sr_acceptance.py`.

---

## 1. What changed, in one paragraph

The held-out proof predicts a frame that was not fused. Doing that correctly needs three things that
were wrong or missing before `fix/heldout`: the prediction must start from the frame the HR grid is
aligned to (`SRResult.reference_index`, not `frames[0]`), the baseline must be built from exactly the
fused frames (`kept_indices`, not the full list), and the HR estimate must be pushed through the
sensor's forward model (band-limit by the per-LR-pixel integration window) before it is compared with
a *measured* LR sample. With those in place the metric reports a clearly positive gain at 2x/3x/4x.
Once the metric was honest, the previous `ibp_iterations=3` default turned out to buy **acutance, not
prediction**: held-out gain is monotone in IBP strength while the HF band barely moves. The default is
now **1 sweep** with the adjoint back-projection.

> **Grid geometry (2026-11-20, B8).** `drizzle`, `predict_lr_from_hr`, the IBP forward model and the
> held-out support mask all use a **pixel-centre aligned** HR grid: `to_drizzle_matrix` adds the
> `(ratio - 1)/2` HR-px offset, matching `cv2.resize` (the baseline/"before" panel). Before this the
> grid was corner-aligned, so the SR output sat 0.5/1.0/1.5 HR px up-left at 2x/3x/4x; the held-out
> *gain* was unaffected (both sides used the same matrix) but the before/after panels and the bicubic
> baseline were misaligned, and at 3x a 1-4 px black rim remained at the bottom/right. All fixture
> sampling in the tests now uses `r*x + (r-1)/2` as well.

---

## 2. The forward-model PSF is physical, and is not tuned

`SENSOR_PSF_SIGMA_PER_RATIO = 0.289` is `1/sqrt(12)`: the standard deviation of a uniform integration
window one LR pixel wide, expressed in HR pixels (`sigma_HR = ratio/sqrt(12)`). It already existed in
the IBP forward model before this work; the change is that the held-out prediction uses the *same*
constant, so "an HR image that could have produced this LR measurement" means one thing everywhere.

Sensitivity sweep (ROI 512, held-out frames 0/2/4/6/8, 9-frame burst, corrected metric,
`--heldout-psf` reproduces each row):

| sigma (per unit ratio) | sigma at 2x | 2x gain mean | 2x gain min | sigma at 4x | 4x gain mean | 4x gain min |
|---|---|---|---|---|---|---|
| 0 (no band-limit) | 0 | +6.899 | +4.864 | 0 | +5.698 | +4.421 |
| 0.2 | 0.40 px | +6.930 | +4.878 | 0.80 px | +6.416 | +4.692 |
| **0.289 (shipped)** | 0.578 px | **+6.995** | **+4.900** | 1.156 px | **+6.649** | **+4.777** |
| 0.4 | 0.80 px | +7.025 | +4.902 | 1.60 px | +6.716 | +4.798 |

Reading:

* the default is **not** the argmax (0.4 is 0.03/0.07 dB higher — a wider PSF flatters the prediction
  because a real lens is softer than the pixel aperture alone, so 0.4 is not "more physical", it is
  "more forgiving");
* at 2x the whole 0 → 0.4 range spans 0.13 dB, i.e. the choice is inside the noise of the burst;
* at 4x essentially the whole benefit is *having* a band-limit at all (+0.95 dB for 0 → 0.289, of
  which only 0.07 dB is "more than physical"); leaving it out aliases, which is why the no-PSF
  variant is reported next to the headline gain rather than being the default;
* no sigma value was selected by looking at this table — 0.289 is fixed by `1/sqrt(12)`.

Data: `/tmp/rawsr/B2_sigma_sweep.json`, log `/tmp/rawsr/B2_sigma_sweep.log`.

---

## 3. IBP: how many sweeps, and what each sweep costs

Same measurement (ROI 512, held-out frames {0,4}, merged `integ/B1` code, `ibp_adjoint=True`,
`ibp_lambda=0.5`):

| config | 2x held-out mean | 2x hf_gain_db | 2x sharpness_ratio | 4x held-out mean | 4x hf_gain_db | 4x sharpness_ratio |
|---|---|---|---|---|---|---|
| `ibp_iterations=0` | +9.28 | 10.40 | 5.89 | +8.63 | 21.18 | 811 |
| **`1` (shipped)** | **+9.01** | **10.51** | **6.43** | **+8.59** | **21.18** | **911** |
| `ibp_iterations=3` | +8.20 | 10.74 | 9.44 | +7.28 | 21.19 | 1065 |

(Measured on the merged branch, `/tmp/rawsr/B2_ibp_sweep.json`, held-out frames {0,4}. The earlier
full 2x/3x/4x sweep with 9 held-out frames on `fix/heldout` gave the same ordering and the same
magnitudes: 0 sweeps 0.2-0.6 dB above 1, 3 sweeps 0.8-1.3 dB below 1.)

* Sweeps 2 and 3 add **+0.23 dB HF at 2x (+0.01 dB at 4x) and +47 % / +17 % Laplacian sharpness**
  while losing **0.81 dB (2x) / 1.30 dB (4x) held-out PSNR**: the project's own definition of detail
  that does not generalise.
* 0 sweeps is *slightly* better on the metric than 1 (+0.27 dB at 2x, +0.05 dB at 4x), but drops
  the data-consistency step the module is built around and its sharpness (5.89 vs 6.43 at 2x,
  811 vs 911 at 4x). 1 sweep is the shipped compromise: it reduces the forward-model residual, and
  its cost is ≤0.3 dB of held-out gain.
* The step is the **adjoint of the assumed forward model** (Gaussian sensor PSF + bilinear sampling),
  not of the drizzle deposit. With the `reg_sigma` smoothing, the `max_step` clip and the `norm`
  preconditioning it is a preconditioned, clipped descent direction — "steepest descent" would be an
  overstatement and is no longer used in the code or docs.

---

## 4. Telling two runs apart (metrics contract)

`ibp_iterations` changed default 3 → 1, so a result produced before/after (or by a caller that pins
it) is **not** comparable unless the configuration is recorded. Every `SRResult.metrics` therefore
carries, and every acceptance JSON contains:

| key | meaning |
|---|---|
| `ibp_iterations` | sweeps actually run (0 = IBP off) |
| `ibp_lambda` | step size applied to the correction |
| `ibp_reg_sigma` | Gaussian sigma of the correction smoother (HR px) |
| `ibp_max_step` | per-sweep clip on the correction (`null` = no clip) |
| `ibp_forward_sigma_hr` | resolved forward-model PSF in HR px (`0.289*ratio` by default) |
| `ibp_joint_refine` | EBP transform refinement on/off |
| `ibp_adjoint` | adjoint back-projection on/off (plain splat if false) |
| `ibp_summary` | display-ready one line, e.g. `1 sweep x lambda 0.50, adjoint, forward sigma 0.58 HR px` (`""` when IBP is off) |
| `heldout_forward_sigma_per_ratio` | PSF used by the held-out prediction (`0` = no-PSF variant) |
| `heldout_frames_fused`, `heldout_blur_threshold_ratio` | which frames the held-out SR run kept, and the blur gate it used |

**UI and desktop runs.** `raw_sr/ui/worker.py` and `desktop/backend/server.py` construct
`SuperResolution` without pinning `ibp_iterations`, so they run the core default and now report
`ibp_iterations=1` / `ibp_summary` in their metrics. A UI result and an acceptance result are
therefore comparable by reading the same keys — no code change is needed in either caller, and this
branch deliberately does not touch them. Practical consequence to communicate to users: default output
is slightly *softer* than before this change (2x `sharpness_gain` 3.25 → 2.38, `hf_gain_db`
8.30 → 8.34 at ROI 1024, measured on the merged branch as `_B2heldout2`) and the held-out gain
against a proper baseline is ~0 dB (§7). Any archived acceptance JSON
produced with `ibp_iterations=3` (e.g. the `*_A2.json` baselines and the ROI-512 +0.3317 dB figure)
is not comparable with post-merge runs for the same reason.

---

## 5. What the held-out number does and does not mean

* It compares, on the held-out frame's own LR sampling grid, the SR prediction (band-limited by the
  sensor PSF) against a denoised LR stack of the **same fused frames** with the same weights and the
  same `W_held` warp. Equal denoising power; the delta isolates resolution recovery.
* `W_held` is a fresh 6-dof homography fitted **on the held-out frame itself**. It is not taken from
  the SR run (the metric must not grade itself) and it is applied to both sides, so it cancels in the
  gain. Absolute PSNR is therefore optimistic; quote the gain.
* The held-out SR run uses `blur_threshold_ratio=0.0` (every frame kept), while the shipped default
  is 0.6. The number describes "all frames fused"; `heldout_blur_threshold_ratio` and
  `heldout_frames_fused` state it, and `evaluate_heldout(..., blur_threshold_ratio=0.6)` measures the
  production gating instead.
* **The baseline is a per-pixel weighted mean on a common valid mask (fix/metric6, 2026-11-19).**
  `denoised_lr_stack` divides by the weight that covered each pixel (`coverage_normalised=True`),
  not by the total weight of every frame. The old normalisation left a dark band along all four
  borders — as wide as the largest inter-frame shift (13.9 LR px on this burst) — because a shifted
  frame ends somewhere inside the ROI and the accumulator is ~0 beyond it. The held-out frame is a
  complete measurement, so the old metric scored SR against a baseline that was wrong by
  construction along every edge: **40-80 % of the baseline's total MSE sat in a 24 px ring**, and
  that alone produced the "+3.7 dB held-out gain" quoted by B2. Both sides are now scored on
  `heldout_valid_mask`: the SR prediction's own geometric support, a **24 LR px interior margin**
  (1.7x the worst measured shift, >20x the bilinear kernel, >3x the sensor-PSF support in LR px;
  4.6 % of the ROI area), and pixels the stack filled from **every** fused frame. Verified: against
  a coverage-normalised baseline the honest gain is ~0 dB at 2x/3x/4x, and the SR interior is
  statistically indistinguishable from the stack (§7).
  The pre-fix whole-frame values survive only as `*_legacy` JSON fields. Never quote them.
* The baseline is still a mean of sub-pixel-shifted frames, so it is a little softer than a single
  sensor PSF. That residual is 0.001 dB of the gain (measured by band-limiting the baseline with the
  same sensor PSF), so it no longer matters.

---

## 6. Reproducing

```bash
cd <worktree>
V=venv/bin/python
# held-out acceptance (one ratio at a time)
flock /tmp/rawsr/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  $V scripts/sr_acceptance.py --ratio 2 --roi-size 1024 --heldout-sweep \
     --out-suffix _B4metric --out-dir output
# sigma sensitivity (one row of §2)
flock /tmp/rawsr/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  $V scripts/sr_acceptance.py --ratio 4 --roi-size 512 --heldout-sweep \
     --heldout-psf 0.4 --out-suffix _psf04 --out-dir /tmp/out
# IBP strength (one row of §3 / §7)
... --ibp-iterations 3
```

---

## 7. The honest numbers (fix/metric6)

With the baseline coverage-normalised and both sides scored on the common valid mask, the two ratios'
tables become:

### 7.1 Acceptance, ROI 1024, 9 held-out frames (`output/*_B4metric.json`)

| ratio | held-out gain mean | min | legacy field (old metric) | valid fraction | HF vs single | wall | peak RSS |
|---|---|---|---|---|---|---|---|
| 2x | **+0.014 dB** | −0.025 | +3.683 | 0.908 | 5.22 dB | 306 s | 1.73 GB |
| 3x | **−0.018 dB** | −0.045 | +3.570 | 0.908 | 14.75 dB | 475 s | 3.00 GB |
| 4x | **−0.033 dB** | −0.053 | +3.460 | 0.908 | 18.89 dB | 742 s | 4.83 GB |

The `legacy` column reproduces B2's quoted numbers **exactly**, which is the proof that the old
figure was the rim and nothing else: the only thing that changed is the normalisation and the mask.

**What this means.** On *this* burst, SR does not predict a held-out frame better than a well-built
denoised LR stack: the gain is zero to within ±0.05 dB at every ratio and slightly negative at 3x/4x.
That is a statement about the burst, not necessarily about SR — the 9 frames are a mild hand-held
sweep (≤14 px), the stack already has 9x averaging, and `hf_gain_db_vs_single` (+5.2/+14.8/+18.9 dB)
shows the imaging chain really does recover band above the *single-frame* Nyquist. A held-out PSNR
against an LR stack measures exactly the band the stack can already reach, so ≈0 dB is the honest
answer. The project must stop quoting a held-out "SR proof" until a burst with genuine >Nyquist
content (or a synthesised aliasing test) produces a positive number — the synthetic test in
`tests/test_heldout_metric.py` shows the metric *does* report +3.3 dB when the content is above the
LR Nyquist, so the instrument works; the real burst simply does not contain that content at a level
above the stack's.

The `masked-coverage-normalised` metric is the headline; `*_legacy` exists only for continuity.

### 7.2 IBP strength on the honest metric (ROI 512, 9 held-out frames)

`--ibp-iterations` now also reaches the held-out run (it used to apply to the displayed image only,
so every sweep row carried the same prediction):

| config | 2x gain mean / min | 2x HF | 2x sharpness | 4x gain mean / min | 4x HF | 4x sharpness |
|---|---|---|---|---|---|---|
| `ibp_iterations=0` | −0.025 / −0.059 | 10.35 | 2.63 | −0.083 / −0.142 | 21.06 | 410 |
| **`1` (shipped)** | **−0.041 / −0.075** | **10.47** | **2.87** | **−0.089 / −0.136** | **21.06** | **461** |
| `ibp_iterations=3` | −0.064 / −0.098 | 10.69 | 4.20 | −0.102 / −0.143 | 21.07 | 538 |

Reading: on the honest metric more IBP is **monotonically worse** for prediction (−0.016 dB per extra
sweep at 2x, −0.006 dB at 4x) while adding +0.22 dB HF and +46 % Laplacian sharpness at 2x. The cost
of the shipped 1 sweep relative to 0 is 0.016 / 0.006 dB — inside the run-to-run scatter. **Keep 1**:
it performs the data-consistency step the module exists for, it is now measured rather than assumed,
and the gap to 0 is not significant. **Reject 3** as a default; it is 0.04 dB worse at 2x for
acutance that the held-out frame does not support. The B1/B2 ordering (1 > 3) survives; the level
of the whole table moved from "+7…+9 dB" to "≈0 dB" because the rim is gone.

Data: `/tmp/rawsr/m6out/sr_acceptance_*_B4ibp*.json`, logs `/tmp/rawsr/m6_ibp_*x_*.log`.

---

## 8. VST edge-preserving MAP-IBP prox (`integ/C1`, 2026-11-22, `ec7cb6b`)

Unregularised IBP left per-HR-pixel photon/read noise unattenuated, causing the
HR estimate to predict an unseen held-out frame slightly worse than the denoised
LR stack at 3×/4× (P1 FAIL: min −0.006 / −0.046 / −0.051 dB at 2×/3×/4×).

`integ/C1` adds a **variance-stabilised (VST) edge-preserving MAP-IBP proximal
step** (`_vst_bilateral_prox`) at the end of `iterative_back_projection()`:

* **Noise estimation (zero held-out leakage).** `_vst_burst_noise_sigma`
  estimates the VST-domain noise standard deviation `sigma_vst` solely from the
  fused frames' own IBP residuals (never touching any held-out frame), guarded
  against over-smoothing by `0.5 * frame_std`.
* **Monotone VST bilateral prox.** The HR estimate is mapped through an
  Anscombe/sqrt variance-stabilising transform, filtered with an edge-preserving
  monotone bilateral kernel (zero ringing/overshoot), and inverted back to
  linear space. The forward/inverse VST runs in-place (`in_place=True`) and IBP
  loop temporaries are freed before the prox to keep 4× peak RSS low.
* **Ratio scaling (`alpha = 0.75`, chosen on synthetic GT only).** In
  `drizzle()`, expected HR sample coverage scales as `E[cov] = sum(w) / ratio^2`,
  so per-HR-pixel noise std scales linearly with `ratio` relative to the
  LR-stack estimate `sigma_vst`. The bilateral range width is therefore scaled
  as `sigma_c = 2.5 * max(ratio / 2, 1)^0.75 * sigma_vst`. The exponent
  `alpha = 0.75` was selected exclusively on the synthetic ground-truth burst
  (`lr=192`, seeds 3/4/5, zero real held-out frames inspected), where it sits
  within `0.002 dB` of the synthetic PSNR argmax (`alpha = 1.0`) while
  preserving more high-frequency texture.
* **Drizzle low-coverage regularisation default off (`low_cov_thresh = 0.0`).**
  Both `drizzle()` and `SuperResolution.__init__()` default
  `drizzle_low_cov_thresh = 0.0` so the edge-preserving VST prox replaces
  isotropic low-coverage Gaussian blur (which cost `-0.258 dB` HF at 4× for
  negligible held-out gain). The parameter remains available for explicit callers.

### Result on real ROI 1024 (`_C5final`, 9 held-out frames)

| ratio | held-out mean (dB) | held-out min (dB, P1) | HF vs single (dB) | Δ HF vs `826bc65` | peak RSS |
|---|---|---|---|---|---|
| 2× | **+0.0519** | **+0.0266 (PASS)** | 4.934 | −0.157 dB | 1093 MiB |
| 3× | **+0.0460** | **+0.0167 (PASS)** | 14.574 | −0.163 dB | 1575 MiB |
| 4× | **+0.0380** | **+0.0045 (PASS)** | 18.418 | −0.163 dB | 2281 MiB |

All 27/27 held-out frames are now strictly positive (**P1 PASS** at 2×/3×/4×),
while detail vs a single frame stays well within the `<= 0.3 dB` budget.

---

## 9. Where the 3×/4× deficit actually is (diagnosis, 2026-11-23)

Recorded so the next agent does not re-open the same doors. All numbers below
are leave-one-out on the real 9-DNG burst (ROI 1024, `predict_lr_from_hr` with
the shipped sensor PSF, the evaluator's masks untouched).

**It is not the forward model.** Sweeping the prediction PSF around the
shipped `0.289 × ratio` (0, 0.1 … 0.6 per unit ratio, proper 4σ truncation)
moves the held-out gain monotonically and by ≤0.15 dB at 3×/4×; the shipped
value is already at the good end of that range (a wider PSF is merely more
forgiving, §2), so there is nothing to tune there. A linear filter fitted *on the other
held-out frames* and applied in the held-out frame's own LR grid — the best any
linear forward model could do — is **worse** than the shipped model
(3× mean −0.37 dB). A composite deconvolution `B_sensor / B_deposit` blows up
(the deposit is not the imaging model, and its inverse amplifies the
reconstruction's noise by 20+ dB).

**It is not the HR→LR round trip.** Pushing the coverage-normalised denoised LR
stack, bicubic-upsampled to HR, through the *same* forward model scores
**+0.017 dB** (min +0.012) at 4× — i.e. the round trip is innocent. The shipped
SR image scores −0.046 dB (min −0.051) at the same measurement, and a
drizzle-only image −0.038 dB. The loss is in the HR estimate's content.

**It is noise variance, and it is broadband.** Two disjoint halves of the burst
(5 + 4 frames, same transforms/reference/weights) reconstructed independently
and compared per radial band at 4×: coherence 0.9998 (0–0.1 cyc/LR px), 0.989
(0.1–0.2), 0.892 (0.2–0.3), then 0.23 (0.3–0.4), 0.21 (0.4–0.5), 0.17
(0.5–0.6), ≤0.06 above. Above the LR Nyquist the reconstruction's content is
90–98 % *unreproducible between halves* — that is also what
`hf_gain_db_vs_single` (+4.9/+14.6/+18.4 dB) is measuring at 3×/4×, which is
why it must never be quoted as resolution recovery.

**Consequences, both measured.** (a) A closed-form "LR-band consistency
projection" (substitute the real LR stack below some hand-over frequency, keep
the reconstruction above it) recovers at most +0.005…+0.016 dB and spends
0.3–1.0 dB of `hf_gain_db_vs_single` even when the transition ends exactly at
the LR Nyquist; the variant with a hand-over at 1.4× Nyquist was rejected on
arbiter (a) (`fix/fwd20` @ `456fcda`, 3× synthetic GT −0.29 dB) and reverted.
(b) The fix that works acts on the *estimate*: the VST edge-preserving MAP-IBP
prox of §8, which suppresses the unreproducible 0.3–1.0 cyc/LR band without a
frequency-selective substitution. Independent re-check of the shipped default
on `master` (this note's date, first three held-out frames, 4× `memory_budget`
override): +0.0289/+0.0266/+0.0412 dB at 2×, +0.0194/+0.0167/+0.0215 at 3×,
+0.0095/+0.0045/+0.0177 at 4× — every frame positive, matching `_C5final`.
