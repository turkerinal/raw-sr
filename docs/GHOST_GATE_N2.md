# The ghost gate at N=2 (pair detector D5 + reference-only fusion)

Last revised 2026-11-20 (branch `fix/xf22`). Owner area: the auto ghost gate
(`_ghost_gate_*`, `_auto_robust_fusion`) and the robust-fusion path in
`drizzle`. Registration/IBP and the drizzle deposit parameters belong to other
branches and were not touched.

---

## 1. Why two frames are special

Every detector before D5 reasons about a frame's deviation from the
**across-frame median**. With `N = 2` that median *is* the mean, which has two
consequences:

* the measured deviation is **half** the real frame-to-frame disagreement;
* both frames sit at that same distance from the reference, so the Tukey
  biweight gives them the **same** weight — and the weighted average is then
  literally the plain average.

Measured on the flooded synthetic burst below, forced `robust_fusion=True`
differed from `robust_fusion=False` by **< 1e-4** per pixel: at N=2 the robust
re-weighting pass was inert. The xfail pinned in `fix/sign14` was therefore not
a threshold that needed nudging; the statistic D3 was reading was the wrong
one.

## 2. D5: the pair disagreement

`_ghost_gate_pair` / `_ghost_gate_pair_at` compare the two frames **directly**
on the same gain-normalized HR stack D3 uses:

    ratio = mean |Sg[0] - Sg[1]|  /  (p95 - p5 of the reference |level|)   (over common coverage)
    share = D2 tile-mask share

`ratio` shares D3's scale (it is exactly 2x D3's `mean(|d|/2)`), so the existing
contrast normalization keeps its meaning. A pair alone is only trusted when a
*large* part of the frame disagrees as one population
(`share >= _GHOST_GATE_PAIR_SHARE = 0.10`), because with no redundancy a single
contrasty static crop can look like a layer.

Measured (N=2, one pair, CPU scatter, same fixtures as the ghost-gate suites):

| case (N=2)                            | pair ratio | D2 share | verdict |
|---------------------------------------|-----------:|---------:|---------|
| real burst, 12 ROIs, 512 and 1024      | 0.003-0.082| 0.009-0.223 | OFF |
| synthetic static, +-5% exposure drift/step | 0.010  | 0.000    | OFF     |
| synthetic static, noise 0.012-0.020    | 0.030-0.050| 0.000    | OFF     |
| synthetic static, contrast 0.02-0.10   | 0.085-0.351| 0.000    | OFF     |
| flooded moving subject (57% of frame)  | 0.137      | 0.366    | **ON**  |

The cuts (ratio 0.10, share 0.10) sit in the gap: worst static ratio 0.082 is
1.2x below the cut, the flooded case is 1.4x above it; the worst static share
that also had a high ratio was 0.223 but its ratio was 0.006, and every
low-contrast/noisy static that reaches ratio 0.35 has share **0.000** (nothing
stands out spatially).

D5 runs **only** for `len(frames) == 2` (`_ghost_gate_pair` returns `None`
otherwise), and it runs *last*, after D1-D4, so no burst with three or more
frames changes at all.

## 3. Reference-only fusion where the pair disagrees

`drizzle(..., reference_pos=<position of SRResult.reference_index>)` replaces
the gate's region with a **single-frame** render of the reference frame
(`_reference_only_region`) instead of re-weighting. Rationale: with two
observations and no third to break the tie, down-weighting cannot remove the
ghost; taking one observation can.

* The reference is the frame the HR grid is aligned to and the frame a held-out
  prediction is defined against, so the subject appears once, where the
  reference has it.
* The region alpha is the gate's own disagreement mask dilated by **one** tile
  (the weight path dilates by two; replacing that much of the frame with
  single-frame content costs HF far outside the contradiction — measured
  7.72 -> 1.58 dB on a 34-px subject).
* The render is a bicubic resample, not a scatter deposit: at 2x a single
  frame's `pixfrac = 0.5` lattice leaves 3 of 4 HR pixels uncovered. Inside the
  region the result is therefore single-frame resolution, which is honest — the
  two frames contradict each other there, so there is no valid sub-pixel
  information left to recover.
* IBP (a different owner's area) still refines the whole image against both
  frames, so the region is not the reference render bit for bit; it does stay
  much closer to the reference than to the other frame's displaced copy (test
  `test_n2_fallback_keeps_the_reference_not_the_average`).

Effect on the flooded synthetic burst (trail error vs ground truth, the swept
band where plain fusion shows both copies of the subject): plain `0.06302`,
auto `0.05793` (−8.1 %). This is a modest improvement because the reference
copy of a 760-px subject still sits inside the trail band; the point of the
change is that the ghost is a *single* copy, not a superposition.

## 4. D4 no longer decides N=2

`_auto_robust_fusion` requires at least **two** non-self registration residuals
for the D4 branch (`len(residuals) >= 2`, i.e. N >= 3). With one residual the
"median" is that single number and a failed ECC lock is indistinguishable from
motion — measured on the real burst at ROI (0,0) 512, where the 2-frame fit
misses by **85.7 px** and D4 used to declare the static crop "moving" (and,
under the new fallback, would have replaced it with single-frame content). N=2
motion is D5's job, and D5 uses stack statistics that do not depend on the fit
succeeding.

## 5. Contract

* Static real burst at N=2: AUTO resolves **OFF** and is **bit-identical** to
  `robust_fusion=False` (`hf_gain_db`, `sharpness_ratio`, `hr` all equal) over
  the ROI grid in `tests/test_superres_ghost_gate_b2.py`
  (`test_real_static_burst_at_n2_is_off_and_byte_identical`).
* Static synthetic N=2 (exposure drift/step, low contrast, heavy noise): OFF.
* Moving subject at N=2 (small, 220-px dominant, flooded): ON; the flooded case
  is carried by D5 and the reason reads
  `moving subject detected (2-frame pair disagreement ... keeps only the
  reference frame)`.
* `robust_reason` is surfaced by the desktop result panel
  (`ghostText` in `ResultPanel.tsx`), so the N=2 behaviour is visible in the
  UI with no extra warning state: N=2 is now decided, not a caveat.
* No xfail remains for the gate: `grep -rn xfail tests` is empty.
