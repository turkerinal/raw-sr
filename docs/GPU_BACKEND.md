# CuPy/CUDA GPU Backend (`raw_sr.core.gpu`)

Classical (non-neural) GPU acceleration for `raw_sr` using CuPy (`cupy-cuda13x[ctk]`).

## Design

Earlier GPU attempts (OpenCV OpenCL T-API warp routing and `pyopencl` scatter)
transferred buffers per operation and measured slower than CPU. The CuPy backend
keeps data **resident in VRAM across an entire stage** (upload once, execute all
iterations/passes on GPU, download once):

1. **Drizzle scatter (`scatter_frames_gpu`)**: keeps the `(oh, ow, 3)` RGB
   accumulator and `(oh, ow)` coverage table resident in VRAM across the burst.
   A custom CUDA `RawKernel` runs one thread per LR pixel with `double`
   homography projection and `float32` `atomicAdd`, supporting `rweights` and
   `local_shifts`.
2. **Iterative Back-Projection (`iterative_back_projection_gpu`)**: keeps `hr`,
   `corr`, and `norm` resident in VRAM across all frames and sweeps. Uses custom
   CUDA `RawKernel`s for bilinear `warpPerspective` (forward simulation, adjoint
   back-projection, footprint) and the VST bilateral proximal step, plus
   separable `convolve1d` Gaussian blurs.
3. **Enhancement (`richardson_lucy_tv_gpu`, `starlet_sharpen_gpu`)**: keeps
   luminance resident in VRAM across all RL-TV iterations (separable PSF
   convolutions, TV divergence, morphological deringing, edge-aware blending)
   and a-trous B3-spline starlet scales + MAD thresholding.

## API & Auto Calibration

- `SuperResolution(..., use_gpu=None | True | False)`
- `enhance_image(..., use_gpu=None | True | False)`
- `richardson_lucy_tv(..., use_gpu=...)`, `starlet_sharpen(..., use_gpu=...)`,
  `iterative_back_projection(..., use_gpu=...)`

Modes:
- `False`: pure CPU (bit-identical to pre-GPU baseline).
- `True`: forces GPU when CuPy + CUDA device are present; falls back to CPU on
  VRAM guard refusal or `OutOfMemoryError`.
- `None` (default, auto): enabled per stage only when CuPy + CUDA device are
  present, output size >= 512x512, and a one-time warm calibration measures GPU
  >= 5% faster than CPU.

`SRResult.metrics` reports `gpu_drizzle`, `gpu_ibp`, and `gpu_stages`.
`EnhanceResult.details` reports `gpu_active` (`1.0` or `0.0`).

## VRAM & OOM Safety

Every GPU entry point checks `gpu_fits_vram(required_bytes)` before allocating
and catches `OutOfMemoryError` / CUDA runtime errors, releasing CuPy pool blocks
and falling back to CPU without crashing. Peak measured VRAM on a 4 GB laptop
GPU is ~1.3 GB at 2x (2048x2048) and ~2.3 GB at 4x (4096x4096).

## What Stayed on CPU and Why

- **RAW loading (`rawpy`/LibRaw)**: I/O + C demosaic library; CPU-only.
- **Global ECC registration**: serial iterative `cv2.findTransformECC` on small
  1024x1024 green planes; already parallelised across CPU workers.
- **Local alignment (`local_align`, opt-in)**: many tiny 48x48 per-tile
  phase-correlation + LK solves; launch-bound on GPU, left on CPU.
- **Export**: 16-bit TIFF write (~0.3-1.0 s), I/O-bound.
- **Adaptive PSF estimation & safety check**: small patch/tile operations;
  the heavy RL-TV + starlet core runs on GPU.

## Independent re-check (own9, 1024 ROI, min of 2 runs; laptop timings vary up to 2x run to run)
- `SuperResolution.run`: 2x CPU 6.20 s -> GPU 5.53 s (~10%); 4x CPU 14.8 s vs GPU 16.2 s (no gain).
  The GPU stages (drizzle, IBP) are a small share of wall time; RAW load and registration (CPU) dominate.
  Bit-level agreement confirmed: max diff 5.5e-6 (2x) / 9.2e-6 (4x), PSNR > 135 dB.
- `enhance_image`: wp2 2x 0.93 -> 0.30 s, 4x 7.83 -> 1.84 s (4.3x); adaptive 2x 0.92 -> 0.61 s,
  4x 18.9 -> 12.9 s. Max diff 7e-4 (wp2), 1.5e-6 (adaptive).
- The report's 4x end-to-end 34.1 -> 18.6 s was not reproduced for `run()`; the clear win is enhance.
  Next lever: parallel/faster RAW decode and registration (CPU side), not more GPU kernels.
