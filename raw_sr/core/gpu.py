"""Optional CuPy/CUDA GPU acceleration backend for classical raw_sr stages.

Design principles
-----------------
1. **GPU-resident stages**: each accelerated stage uploads its inputs ONCE,
   executes all iterative / multi-pass operations on the device, and downloads
   the result ONCE. Per-op host<->device round-trips are avoided.
2. **Safe VRAM guards & OOM fallback**: every GPU entry point checks free VRAM
   before allocating and catches ``cupy.cuda.memory.OutOfMemoryError`` (and any
   CUDA runtime failure), releasing cached pool blocks and returning ``None`` so
   the caller falls back transparently to the bit-identical CPU path.
3. **Honest auto mode**: ``use_gpu=None`` enables a stage on the GPU only when
   CuPy + a CUDA device are present AND a one-time warm calibration measures the
   GPU path faster than CPU for that stage.
4. **Classical signal processing only**: no neural networks; every kernel
   implements the exact mathematical operator of the CPU reference.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Device discovery & VRAM management
# --------------------------------------------------------------------------- #

_CUPY_CHECKED = False
_CUPY_MOD: Any = None
_CUPYX_NDIMAGE: Any = None
_DEVICE_INFO: Optional[Dict[str, Any]] = None

# One-time calibration cache: stage -> bool (True = GPU faster than CPU)
_STAGE_BENEFICIAL: Dict[str, bool] = {}

# Minimum output pixels for auto mode to consider GPU (below this, launch
# overhead exceeds compute savings).
_STAGE_MIN_PIXELS: Dict[str, int] = {
    "enhance": 512 * 512,
    "ibp": 512 * 512,
    "drizzle": 512 * 512,
}


def _import_cupy():
    """Lazily import cupy and cupyx.scipy.ndimage; return (cp, ndimage) or (None, None)."""
    global _CUPY_CHECKED, _CUPY_MOD, _CUPYX_NDIMAGE, _DEVICE_INFO
    if _CUPY_CHECKED:
        return _CUPY_MOD, _CUPYX_NDIMAGE
    _CUPY_CHECKED = True
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cp_ndimage

        n_dev = int(cp.cuda.runtime.getDeviceCount())
        if n_dev <= 0:
            return None, None
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name", b"CUDA GPU")
        name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        _DEVICE_INFO = {
            "name": name,
            "vram_total_mb": round(total_b / (1024 * 1024), 1),
            "vram_free_mb": round(free_b / (1024 * 1024), 1),
            "cupy_version": getattr(cp, "__version__", "unknown"),
        }
        _CUPY_MOD = cp
        _CUPYX_NDIMAGE = cp_ndimage
    except Exception:
        _CUPY_MOD = None
        _CUPYX_NDIMAGE = None
        _DEVICE_INFO = None
    return _CUPY_MOD, _CUPYX_NDIMAGE


def gpu_available() -> bool:
    """Return True if CuPy is installed and at least one CUDA device is usable."""
    cp, _ = _import_cupy()
    return cp is not None


def gpu_device_info() -> Optional[Dict[str, Any]]:
    """Return basic CUDA device info dict, or None if unavailable."""
    cp, _ = _import_cupy()
    if cp is None or _DEVICE_INFO is None:
        return None
    info = dict(_DEVICE_INFO)
    try:
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        info["vram_free_mb"] = round(free_b / (1024 * 1024), 1)
        info["vram_total_mb"] = round(total_b / (1024 * 1024), 1)
    except Exception:
        pass
    return info


def free_gpu_memory() -> None:
    """Release unused blocks from CuPy's default memory pool."""
    cp, _ = _import_cupy()
    if cp is None:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def gpu_fits_vram(required_bytes: int, safety_fraction: float = 0.78) -> bool:
    """Return True if ``required_bytes`` fits comfortably in available VRAM."""
    cp, _ = _import_cupy()
    if cp is None:
        return False
    try:
        free_b, _ = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        avail = int(free_b) + int(pool.free_bytes())
        return int(required_bytes) <= int(avail * safety_fraction)
    except Exception:
        return False


def is_oom_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a CUDA/CuPy out-of-memory error."""
    cp, _ = _import_cupy()
    if cp is not None:
        oom_cls = getattr(cp.cuda.memory, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cudaerroroutofmemory" in msg


def reset_gpu_calibration() -> None:
    """Clear cached per-stage GPU calibration results (for tests)."""
    _STAGE_BENEFICIAL.clear()


def _bench_median(fn, repeats: int = 3) -> float:
    fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def gpu_stage_beneficial(stage: str) -> bool:
    """One-time cached calibration: True if GPU is >=5% faster than CPU for ``stage``."""
    if stage in _STAGE_BENEFICIAL:
        return _STAGE_BENEFICIAL[stage]
    cp, cp_nd = _import_cupy()
    if cp is None:
        _STAGE_BENEFICIAL[stage] = False
        return False

    try:
        if stage == "enhance":
            # Benchmark separable 1D convolution loop (RL-TV / starlet workload) on 1024x1024
            img_cpu = np.random.default_rng(0).random((1024, 1024), dtype=np.float32)
            k_cpu = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0

            def _cpu():
                x = img_cpu
                for _ in range(6):
                    x = cv2.sepFilter2D(x, cv2.CV_32F, k_cpu, k_cpu, borderType=cv2.BORDER_REPLICATE)
                return x

            def _gpu():
                x = cp.asarray(img_cpu)
                k = cp.asarray(k_cpu)
                for _ in range(6):
                    x = cp_nd.convolve1d(x, k, axis=1, mode="nearest")
                    x = cp_nd.convolve1d(x, k, axis=0, mode="nearest")
                out = cp.asnumpy(x)
                cp.cuda.Stream.null.synchronize()
                return out

            t_cpu = _bench_median(_cpu)
            t_gpu = _bench_median(_gpu)
            win = t_gpu * 1.05 < t_cpu

        elif stage == "ibp":
            # Benchmark Gaussian blur + bilateral/warp on 1024x1024x3
            hr_cpu = np.random.default_rng(1).random((1024, 1024, 3), dtype=np.float32)

            def _cpu():
                x = cv2.GaussianBlur(hr_cpu, (5, 5), 0.6, borderType=cv2.BORDER_REPLICATE)
                return cv2.bilateralFilter(x, 7, 0.05, 1.5, borderType=cv2.BORDER_REPLICATE)

            def _gpu():
                x = cp.asarray(hr_cpu)
                x = _gaussian_blur_rgb_gpu(cp, cp_nd, x, 0.6)
                x = _bilateral_rgb_gpu(cp, x, 7, 0.05, 1.5)
                out = cp.asnumpy(x)
                cp.cuda.Stream.null.synchronize()
                return out

            t_cpu = _bench_median(_cpu)
            t_gpu = _bench_median(_gpu)
            win = t_gpu * 1.05 < t_cpu

        elif stage == "drizzle":
            # Benchmark scatter of 4 frames of 512x512 -> 1024x1024
            src = np.random.default_rng(2).random((512, 512, 3), dtype=np.float32)
            M = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)

            def _cpu():
                from .superres import _scatter_accumulate
                acc = np.zeros((1024, 1024, 3), dtype=np.float32)
                cov = np.zeros((1024, 1024), dtype=np.float32)
                for _ in range(4):
                    _scatter_accumulate(src, M, 1024, 1024, 0.5, 2, splat="bilinear",
                                        out_acc=acc, out_wt=cov)
                return acc, cov

            def _gpu():
                acc = cp.zeros((1024, 1024, 3), dtype=cp.uint64)
                cov = cp.zeros((1024, 1024), dtype=cp.uint64)
                for _ in range(4):
                    _scatter_frame_into_gpu(cp, src, M, acc, cov, 0.5, 2, 1.0)
                inv_s = np.float64(1.0 / 1073741824.0)
                out = ((cp.asnumpy(acc) * inv_s).astype(np.float32),
                       (cp.asnumpy(cov) * inv_s).astype(np.float32))
                cp.cuda.Stream.null.synchronize()
                return out

            t_cpu = _bench_median(_cpu)
            t_gpu = _bench_median(_gpu)
            win = t_gpu * 1.05 < t_cpu
        else:
            win = False
    except Exception:
        win = False
    finally:
        free_gpu_memory()

    _STAGE_BENEFICIAL[stage] = bool(win)
    return bool(win)


def resolve_use_gpu(use_gpu: Optional[bool], stage: str, pixels: int) -> bool:
    """Resolve tri-state ``use_gpu`` (None=auto, True=force, False=CPU) for a stage."""
    if use_gpu is False:
        return False
    if not gpu_available():
        return False
    if use_gpu is True:
        return True
    # Auto mode: require minimum size and calibration win
    if pixels < _STAGE_MIN_PIXELS.get(stage, 512 * 512):
        return False
    return gpu_stage_beneficial(stage)


# --------------------------------------------------------------------------- #
# Custom CUDA RawKernels (lazy-compiled)
# --------------------------------------------------------------------------- #

_WARP_PERSP_RGB_KERNEL = None
_WARP_PERSP_1CH_KERNEL = None
_BILATERAL_RGB_KERNEL = None
_SCATTER_BILINEAR_KERNEL = None


def _get_warp_persp_rgb_kernel(cp):
    global _WARP_PERSP_RGB_KERNEL
    if _WARP_PERSP_RGB_KERNEL is None:
        _WARP_PERSP_RGB_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void warp_persp_rgb_bilinear(
            const float* __restrict__ src,
            int sh, int sw,
            float* __restrict__ dst,
            int dh, int dw,
            double h00, double h01, double h02,
            double h10, double h11, double h12,
            double h20, double h21, double h22
        ) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= dw || y >= dh) return;

            double denom = h20 * x + h21 * y + h22;
            double inv_d = (fabs(denom) > 1e-12) ? (1.0 / denom) : 0.0;
            double sx = (h00 * x + h01 * y + h02) * inv_d;
            double sy = (h10 * x + h11 * y + h12) * inv_d;

            int out_idx = (y * dw + x) * 3;
            int x0 = (int)floor(sx);
            int y0 = (int)floor(sy);
            float fx = (float)(sx - x0);
            float fy = (float)(sy - y0);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            float r = 0.0f, g = 0.0f, b = 0.0f;
            if (x0 >= 0 && x0 < sw && y0 >= 0 && y0 < sh) {
                int i = (y0 * sw + x0) * 3;
                r += w00 * src[i]; g += w00 * src[i + 1]; b += w00 * src[i + 2];
            }
            if (x1 >= 0 && x1 < sw && y0 >= 0 && y0 < sh) {
                int i = (y0 * sw + x1) * 3;
                r += w10 * src[i]; g += w10 * src[i + 1]; b += w10 * src[i + 2];
            }
            if (x0 >= 0 && x0 < sw && y1 >= 0 && y1 < sh) {
                int i = (y1 * sw + x0) * 3;
                r += w01 * src[i]; g += w01 * src[i + 1]; b += w01 * src[i + 2];
            }
            if (x1 >= 0 && x1 < sw && y1 >= 0 && y1 < sh) {
                int i = (y1 * sw + x1) * 3;
                r += w11 * src[i]; g += w11 * src[i + 1]; b += w11 * src[i + 2];
            }
            dst[out_idx] = r;
            dst[out_idx + 1] = g;
            dst[out_idx + 2] = b;
        }
        ''', 'warp_persp_rgb_bilinear')
    return _WARP_PERSP_RGB_KERNEL


def _get_warp_persp_1ch_kernel(cp):
    global _WARP_PERSP_1CH_KERNEL
    if _WARP_PERSP_1CH_KERNEL is None:
        _WARP_PERSP_1CH_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void warp_persp_ones_bilinear(
            int sh, int sw,
            float* __restrict__ dst,
            int dh, int dw,
            double h00, double h01, double h02,
            double h10, double h11, double h12,
            double h20, double h21, double h22
        ) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= dw || y >= dh) return;

            double denom = h20 * x + h21 * y + h22;
            double inv_d = (fabs(denom) > 1e-12) ? (1.0 / denom) : 0.0;
            double sx = (h00 * x + h01 * y + h02) * inv_d;
            double sy = (h10 * x + h11 * y + h12) * inv_d;

            int x0 = (int)floor(sx);
            int y0 = (int)floor(sy);
            float fx = (float)(sx - x0);
            float fy = (float)(sy - y0);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float val = 0.0f;
            if (x0 >= 0 && x0 < sw && y0 >= 0 && y0 < sh) val += (1.0f - fx) * (1.0f - fy);
            if (x1 >= 0 && x1 < sw && y0 >= 0 && y0 < sh) val += fx * (1.0f - fy);
            if (x0 >= 0 && x0 < sw && y1 >= 0 && y1 < sh) val += (1.0f - fx) * fy;
            if (x1 >= 0 && x1 < sw && y1 >= 0 && y1 < sh) val += fx * fy;
            dst[y * dw + x] = val;
        }
        ''', 'warp_persp_ones_bilinear')
    return _WARP_PERSP_1CH_KERNEL


def _get_bilateral_rgb_kernel(cp):
    global _BILATERAL_RGB_KERNEL
    if _BILATERAL_RGB_KERNEL is None:
        _BILATERAL_RGB_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void bilateral_rgb_replicate(
            const float* __restrict__ src,
            float* __restrict__ dst,
            int H, int W,
            int radius,
            float inv_2sigma_c2,
            float inv_2sigma_s2
        ) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= W || y >= H) return;

            int c_idx = (y * W + x) * 3;
            float cr = src[c_idx];
            float cg = src[c_idx + 1];
            float cb = src[c_idx + 2];

            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_w = 0.0f;
            int r2_max = radius * radius;
            for (int dy = -radius; dy <= radius; ++dy) {
                int ny = y + dy;
                if (ny < 0) ny = 0;
                else if (ny >= H) ny = H - 1;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int r2 = dx * dx + dy * dy;
                    if (r2 > r2_max) continue;
                    int nx = x + dx;
                    if (nx < 0) nx = 0;
                    else if (nx >= W) nx = W - 1;
                    int n_idx = (ny * W + nx) * 3;
                    float nr = src[n_idx];
                    float ng = src[n_idx + 1];
                    float nb = src[n_idx + 2];
                    // OpenCV bilateralFilter uses L1 colour distance for 3-channel float32
                    float dc = fabsf(nr - cr) + fabsf(ng - cg) + fabsf(nb - cb);
                    float w = expf(-(float)r2 * inv_2sigma_s2 - dc * dc * inv_2sigma_c2);
                    sum_r += w * nr;
                    sum_g += w * ng;
                    sum_b += w * nb;
                    sum_w += w;
                }
            }
            float inv_w = (sum_w > 1e-12f) ? (1.0f / sum_w) : 1.0f;
            dst[c_idx] = sum_r * inv_w;
            dst[c_idx + 1] = sum_g * inv_w;
            dst[c_idx + 2] = sum_b * inv_w;
        }
        ''', 'bilateral_rgb_replicate')
    return _BILATERAL_RGB_KERNEL


def _get_scatter_bilinear_kernel(cp):
    global _SCATTER_BILINEAR_KERNEL
    if _SCATTER_BILINEAR_KERNEL is None:
        _SCATTER_BILINEAR_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void scatter_bilinear_kernel(
            const float* __restrict__ src,
            int sh, int sw,
            unsigned long long* __restrict__ acc_q,
            unsigned long long* __restrict__ cov_q,
            int oh, int ow,
            double m00, double m01, double m02,
            double m10, double m11, double m12,
            double m20, double m21, double m22,
            const double* __restrict__ offs,
            int n_sub,
            double tap_weight,
            const float* __restrict__ rweight,
            int has_rw,
            const float* __restrict__ local_du,
            const float* __restrict__ local_dv,
            int has_local
        ) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= sw || y >= sh) return;

            int p_idx = y * sw + x;
            double rw = has_rw ? (double)rweight[p_idx] : 1.0;
            double base_w = tap_weight * rw;
            if (base_w <= 0.0) return;

            double r = fmax((double)src[p_idx * 3], 0.0);
            double g = fmax((double)src[p_idx * 3 + 1], 0.0);
            double b = fmax((double)src[p_idx * 3 + 2], 0.0);
            double du_loc = has_local ? (double)local_du[p_idx] : 0.0;
            double dv_loc = has_local ? (double)local_dv[p_idx] : 0.0;
            const double SCALE = 1073741824.0; // 2^30 fixed-point for deterministic atomicAdd

            for (int sy = 0; sy < n_sub; ++sy) {
                double yy = (double)y + offs[sy];
                for (int sx = 0; sx < n_sub; ++sx) {
                    double xx = (double)x + offs[sx];
                    double denom = m20 * xx + m21 * yy + m22;
                    if (fabs(denom) < 1e-12) continue;
                    double u = (m00 * xx + m01 * yy + m02) / denom + du_loc;
                    double v = (m10 * xx + m11 * yy + m12) / denom + dv_loc;

                    int u0 = (int)floor(u);
                    int v0 = (int)floor(v);
                    double fu = u - (double)u0;
                    double fv = v - (double)v0;

                    double w00 = base_w * (1.0 - fu) * (1.0 - fv);
                    double w10 = base_w * fu * (1.0 - fv);
                    double w01 = base_w * (1.0 - fu) * fv;
                    double w11 = base_w * fu * fv;

                    if (u0 >= 0 && u0 < ow && v0 >= 0 && v0 < oh) {
                        int idx = v0 * ow + u0;
                        atomicAdd(&cov_q[idx], (unsigned long long)llround(w00 * SCALE));
                        atomicAdd(&acc_q[idx * 3], (unsigned long long)llround(r * w00 * SCALE));
                        atomicAdd(&acc_q[idx * 3 + 1], (unsigned long long)llround(g * w00 * SCALE));
                        atomicAdd(&acc_q[idx * 3 + 2], (unsigned long long)llround(b * w00 * SCALE));
                    }
                    int u1 = u0 + 1;
                    if (u1 >= 0 && u1 < ow && v0 >= 0 && v0 < oh) {
                        int idx = v0 * ow + u1;
                        atomicAdd(&cov_q[idx], (unsigned long long)llround(w10 * SCALE));
                        atomicAdd(&acc_q[idx * 3], (unsigned long long)llround(r * w10 * SCALE));
                        atomicAdd(&acc_q[idx * 3 + 1], (unsigned long long)llround(g * w10 * SCALE));
                        atomicAdd(&acc_q[idx * 3 + 2], (unsigned long long)llround(b * w10 * SCALE));
                    }
                    int v1 = v0 + 1;
                    if (u0 >= 0 && u0 < ow && v1 >= 0 && v1 < oh) {
                        int idx = v1 * ow + u0;
                        atomicAdd(&cov_q[idx], (unsigned long long)llround(w01 * SCALE));
                        atomicAdd(&acc_q[idx * 3], (unsigned long long)llround(r * w01 * SCALE));
                        atomicAdd(&acc_q[idx * 3 + 1], (unsigned long long)llround(g * w01 * SCALE));
                        atomicAdd(&acc_q[idx * 3 + 2], (unsigned long long)llround(b * w01 * SCALE));
                    }
                    if (u1 >= 0 && u1 < ow && v1 >= 0 && v1 < oh) {
                        int idx = v1 * ow + u1;
                        atomicAdd(&cov_q[idx], (unsigned long long)llround(w11 * SCALE));
                        atomicAdd(&acc_q[idx * 3], (unsigned long long)llround(r * w11 * SCALE));
                        atomicAdd(&acc_q[idx * 3 + 1], (unsigned long long)llround(g * w11 * SCALE));
                        atomicAdd(&acc_q[idx * 3 + 2], (unsigned long long)llround(b * w11 * SCALE));
                    }
                }
            }
        }
        ''', 'scatter_bilinear_kernel')
    return _SCATTER_BILINEAR_KERNEL


def _gaussian_1d_kernel(sigma: float, ksize: Optional[int] = None) -> np.ndarray:
    """1D Gaussian kernel matching OpenCV's getGaussianKernel(k, sigma)."""
    k = int(ksize) if ksize is not None else (int(np.ceil(3.0 * sigma)) | 1)
    half = k // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    return g.astype(np.float32)


def _gaussian_blur_rgb_gpu(cp, cp_nd, img_gpu, sigma: float):
    """Separable Gaussian blur on (H, W, 3) float32 GPU array with BORDER_REPLICATE."""
    k = cp.asarray(_gaussian_1d_kernel(sigma))
    tmp = cp_nd.convolve1d(img_gpu, k, axis=1, mode="nearest")
    return cp_nd.convolve1d(tmp, k, axis=0, mode="nearest")


def _bilateral_rgb_gpu(cp, img_gpu, d: int, sigma_c: float, sigma_s: float):
    """Bilateral filter on (H, W, 3) float32 GPU array with BORDER_REPLICATE."""
    H, W, _ = img_gpu.shape
    radius = max(1, d // 2)
    out = cp.empty_like(img_gpu)
    kern = _get_bilateral_rgb_kernel(cp)
    block = (16, 16)
    grid = ((W + block[0] - 1) // block[0], (H + block[1] - 1) // block[1])
    inv_2sc2 = float(0.5 / (sigma_c * sigma_c))
    inv_2ss2 = float(0.5 / (sigma_s * sigma_s))
    kern(grid, block, (
        cp.ascontiguousarray(img_gpu), out,
        np.int32(H), np.int32(W), np.int32(radius),
        np.float32(inv_2sc2), np.float32(inv_2ss2),
    ))
    return out


# --------------------------------------------------------------------------- #
# GPU-resident Stage 1: Enhance (RL-TV + Starlet + Scene Mask)
# --------------------------------------------------------------------------- #

def _tv_divergence_gpu(cp, u, eps: float = 1e-4):
    """GPU float32 TV divergence matching enhance._tv_divergence."""
    gx = cp.empty_like(u)
    gy = cp.empty_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gx[:, -1] = 0.0
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    gy[-1, :] = 0.0

    norm = cp.sqrt(gx * gx + gy * gy + cp.float32(eps * eps))
    gx /= norm
    gy /= norm

    div = cp.empty_like(u)
    div[:, 0] = gx[:, 0]
    div[:, 1:] = gx[:, 1:] - gx[:, :-1]
    div[0, :] += gy[0, :]
    div[1:, :] += gy[1:, :] - gy[:-1, :]
    return div


def _sobel_mag_gpu(cp, cp_nd, img_gpu, scale: float = 0.125):
    """3x3 Sobel gradient magnitude on GPU matching cv2.Sobel(ksize=3) * scale."""
    k_diff = cp.array([-1.0, 0.0, 1.0], dtype=cp.float32) * cp.float32(scale)
    k_smooth = cp.array([1.0, 2.0, 1.0], dtype=cp.float32)
    # Note: convolve1d flips kernel, so pass -k_diff for correlation;
    # cv2.Sobel default border is BORDER_REFLECT_101 == scipy mode='mirror'.
    gx = cp_nd.convolve1d(cp_nd.convolve1d(img_gpu, -k_diff, axis=1, mode="mirror"),
                          k_smooth, axis=0, mode="mirror")
    gy = cp_nd.convolve1d(cp_nd.convolve1d(img_gpu, k_smooth, axis=1, mode="mirror"),
                          -k_diff, axis=0, mode="mirror")
    return cp.hypot(gx, gy)


def richardson_lucy_tv_gpu(
    luma: np.ndarray,
    psf_1d: np.ndarray,
    max_iters: int = 10,
    tv_weight: float = 0.002,
    noise_sigma: Optional[float] = None,
    discrepancy_tau: float = 1.05,
    dering_tolerance: float = 0.04,
    dering_radius: int = 2,
    max_overshoot_frac: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, int, str]]:
    """GPU-resident TV-regularized Richardson-Lucy deconvolution.

    Returns ``(deconvolved_luma, iters_run, stop_reason)`` or ``None`` on OOM /
    device error so the caller falls back to CPU.
    """
    cp, cp_nd = _import_cupy()
    if cp is None:
        return None

    H, W = luma.shape[:2]
    # ~10 float32 planes resident during RL-TV
    if not gpu_fits_vram(H * W * 4 * 12):
        return None

    try:
        eps = cp.float32(1e-5)
        obs = cp.clip(cp.asarray(luma, dtype=cp.float32), eps, cp.float32(1.0))
        est = obs.copy()
        k = cp.asarray(psf_1d, dtype=cp.float32)
        tv_w = cp.float32(max(tv_weight, 0.0))

        local_min = local_max = local_range = None
        if (dering_tolerance > 0.0 and dering_radius > 0) or (
            max_overshoot_frac is not None and max_overshoot_frac > 0.0
        ):
            ksz = 2 * max(int(dering_radius), 1) + 1
            se_cpu = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            se = cp.asarray(se_cpu > 0)
            local_min = cp_nd.grey_erosion(obs, footprint=se, mode="nearest")
            local_max = cp_nd.grey_dilation(obs, footprint=se, mode="nearest")
            local_range = cp.maximum(local_max - local_min, cp.float32(1e-4))

        iters_run = 0
        stop_reason = "max_iters"
        prev_rms = float("inf")

        for it in range(1, max_iters + 1):
            conv = cp_nd.convolve1d(est, k, axis=1, mode="nearest")
            conv = cp_nd.convolve1d(conv, k, axis=0, mode="nearest")

            if noise_sigma is not None and noise_sigma > 0.0 and it > 1:
                diff = obs - conv
                rms = float(cp.sqrt(cp.mean(diff * diff, dtype=cp.float64)))
                if rms <= discrepancy_tau * noise_sigma:
                    stop_reason = "discrepancy"
                    break
                if it >= 3 and (prev_rms - rms) < 1e-4 * prev_rms:
                    stop_reason = "converged"
                    break
                prev_rms = rms

            rel = obs / cp.maximum(conv, eps)
            corr = cp_nd.convolve1d(rel, k, axis=1, mode="nearest")
            corr = cp_nd.convolve1d(corr, k, axis=0, mode="nearest")

            if tv_w > 0.0:
                div = _tv_divergence_gpu(cp, est)
                denom = cp.maximum(cp.float32(1.0) - tv_w * div, cp.float32(0.5))
                est_next = est * (corr / denom)
            else:
                est_next = est * corr

            cp.clip(est_next, cp.float32(0.0), cp.float32(1.0), out=est_next)

            if max_overshoot_frac is not None and max_overshoot_frac > 0.0 and it >= 3 and local_max is not None:
                over = cp.maximum(est_next - local_max, local_min - est_next)
                strong = local_range > cp.float32(0.05)
                if bool(cp.any(strong)):
                    ov_frac = float(cp.percentile((over[strong] / local_range[strong]), 98.0))
                    if ov_frac > max_overshoot_frac:
                        stop_reason = "overshoot_monitor"
                        break

            est = est_next
            iters_run = it

        if dering_tolerance > 0.0 and dering_radius > 0 and iters_run > 0 and local_min is not None:
            tol = cp.float32(dering_tolerance)
            lo = cp.maximum(cp.float32(0.0), local_min - tol * (local_max - local_min))
            hi = cp.minimum(cp.float32(1.0), local_max + tol * (local_max - local_min))
            cp.clip(est, lo, hi, out=est)

        if noise_sigma is not None and noise_sigma > 0.0 and iters_run > 0:
            gmag = _sobel_mag_gpu(cp, cp_nd, obs, scale=0.125)
            g_thr = cp.float32(max(2.5 * noise_sigma, 1e-5))
            weight = cp.clip((gmag - cp.float32(0.5) * g_thr) / g_thr, cp.float32(0.0), cp.float32(1.0))
            kg = cp.asarray(_gaussian_1d_kernel(1.0, ksize=5))
            weight = cp_nd.convolve1d(cp_nd.convolve1d(weight, kg, axis=1, mode="nearest"),
                                      kg, axis=0, mode="nearest")
            est = obs + weight * (est - obs)

        out = cp.asnumpy(est).astype(np.float32, copy=False)
        return out, iters_run, stop_reason
    except Exception:
        free_gpu_memory()
        return None


def starlet_sharpen_gpu(
    luma: np.ndarray,
    gains: Sequence[float],
    k_sigma: float = 3.0,
    noise_sigma: Optional[float] = None,
    shadow_protect: float = 0.03,
    highlight_protect: float = 0.06,
) -> Optional[np.ndarray]:
    """GPU-resident a-trous B3-spline starlet sharpening. Returns None on OOM/error."""
    cp, cp_nd = _import_cupy()
    if cp is None:
        return None

    n_scales = len(gains)
    if n_scales == 0 or all(abs(g - 1.0) < 1e-6 for g in gains):
        return luma.astype(np.float32, copy=True)

    H, W = luma.shape[:2]
    if not gpu_fits_vram(H * W * 4 * (n_scales + 6)):
        return None

    b3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    noise_factors = (0.8907, 0.2007, 0.0856, 0.0413, 0.0205)

    try:
        luma_g = cp.asarray(luma, dtype=cp.float32)
        c = luma_g.copy()
        details = []
        for j in range(n_scales):
            step = 1 << j
            if step == 1:
                k_cpu = b3
            else:
                k_cpu = np.zeros(4 * step + 1, dtype=np.float32)
                k_cpu[0::step] = b3
            k = cp.asarray(k_cpu)
            c_next = cp_nd.convolve1d(c, k, axis=1, mode="nearest")
            c_next = cp_nd.convolve1d(c_next, k, axis=0, mode="nearest")
            details.append(c - c_next)
            c = c_next

        prot = cp.ones_like(luma_g, dtype=cp.float32)
        if shadow_protect > 0.0:
            prot *= cp.clip(luma_g / cp.float32(shadow_protect), cp.float32(0.0), cp.float32(1.0))
        if highlight_protect > 0.0:
            prot *= cp.clip(
                (cp.float32(1.0) - luma_g) / cp.float32(highlight_protect),
                cp.float32(0.0), cp.float32(1.0),
            )

        out = c
        for j, (w, g) in enumerate(zip(details, gains)):
            if abs(g - 1.0) < 1e-6:
                out += w
                continue
            mad_sigma = float(cp.median(cp.abs(w))) * 1.4826
            if noise_sigma is not None and noise_sigma > 0.0:
                nf = noise_factors[min(j, len(noise_factors) - 1)]
                layer_sigma = max(mad_sigma, float(noise_sigma) * nf)
            else:
                layer_sigma = mad_sigma

            thr = cp.float32(max(k_sigma * layer_sigma, 0.0))
            if thr > 0.0:
                abs_w = cp.abs(w)
                excess = cp.sign(w) * cp.maximum(abs_w - thr, cp.float32(0.0))
            else:
                excess = w
            out += w + cp.float32(g - 1.0) * prot * excess

        cp.clip(out, cp.float32(0.0), cp.float32(1.0), out=out)
        return cp.asnumpy(out).astype(np.float32, copy=False)
    except Exception:
        free_gpu_memory()
        return None


# --------------------------------------------------------------------------- #
# GPU-resident Stage 2: Iterative Back-Projection (IBP + VST bilateral prox)
# --------------------------------------------------------------------------- #

def iterative_back_projection_gpu(
    hr: np.ndarray,
    frames: Sequence,
    aligns: Sequence,
    ratio: float,
    weights: Optional[Sequence[float]] = None,
    iterations: int = 1,
    lam: float = 0.5,
    reg_sigma: float = 0.5,
    max_step: Optional[float] = 0.1,
    forward_sigma: Optional[float] = None,
    coverage: Optional[np.ndarray] = None,
    forward_sigma_shrink: float = 1.0,
    adjoint: bool = True,
    prox_noise_scale: float = 2.5,
    prox_space_lr: float = 0.75,
) -> Optional[np.ndarray]:
    """GPU-resident IBP + VST bilateral proximal step.

    Keeps ``hr``, ``corr``, and ``norm`` resident in VRAM across all frames and
    iterations; uploads each LR frame once per sweep and downloads ``hr`` once
    at the end. Returns ``None`` on OOM or unsupported mode (caller falls back
    to CPU).
    """
    cp, cp_nd = _import_cupy()
    if cp is None:
        return None
    if iterations <= 0 or len(frames) == 0:
        return hr.astype(np.float32, copy=True)

    oh, ow = hr.shape[:2]
    h, w = frames[0].rgb.shape[:2]
    # Resident VRAM: hr (3) + hr_fwd (3) + corr (3) + up (3) + norm (1) + m (1) + cov (1) + scratch (~3) = ~18 float32 planes
    req_bytes = oh * ow * 4 * 18 + h * w * 4 * 12
    if not gpu_fits_vram(req_bytes):
        return None

    from .superres import to_drizzle_matrix, SENSOR_PSF_SIGMA_PER_RATIO, _vst_burst_noise_sigma

    if weights is None:
        weights = [1.0] * len(frames)
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)

    matrices = [to_drizzle_matrix(al, ratio).astype(np.float64) for al in aligns]
    matrices_inv = [np.linalg.inv(M).astype(np.float64) for M in matrices]

    warp_rgb = _get_warp_persp_rgb_kernel(cp)
    warp_1ch = _get_warp_persp_1ch_kernel(cp)
    block = (16, 16)
    grid_lr = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])
    grid_hr = ((ow + block[0] - 1) // block[0], (oh + block[1] - 1) // block[1])

    try:
        hr_g = cp.asarray(hr, dtype=cp.float32)
        cov_g = cp.asarray(coverage, dtype=cp.float32) if coverage is not None else None
        norm_g = None
        norm_safe_g = None
        sim_g = cp.empty((h, w, 3), dtype=cp.float32)
        up_g = cp.empty((oh, ow, 3), dtype=cp.float32)
        m_g = cp.empty((oh, ow), dtype=cp.float32)

        last_vst_mads: List[float] = []
        for it in range(int(iterations)):
            sigma_it = float(forward_sigma) * (float(forward_sigma_shrink) ** it)
            if sigma_it and sigma_it > 0:
                hr_fwd_g = _gaussian_blur_rgb_gpu(cp, cp_nd, hr_g, sigma_it)
            else:
                hr_fwd_g = hr_g

            corr_g = cp.zeros((oh, ow, 3), dtype=cp.float32)
            norm_pending = norm_g is None
            if norm_pending:
                norm_g = cp.zeros((oh, ow), dtype=cp.float32)

            collect_vst = (
                it == int(iterations) - 1
                and bool(reg_sigma and reg_sigma > 0
                         and prox_noise_scale and prox_noise_scale > 0
                         and prox_space_lr and prox_space_lr > 0)
            )
            if collect_vst:
                last_vst_mads = []

            hr_fwd_c = cp.ascontiguousarray(hr_fwd_g)
            for k, (fr, M, M_inv, wt) in enumerate(zip(frames, matrices, matrices_inv, weights)):
                # Note: OpenCV warpPerspective(src, M, dsize) maps dst(x,y) <- src(inv(M) * (x,y)).
                # So forward sim uses M_inv in cv2 -> inverse of M_inv is M!
                # And back-projection uses M in cv2 -> inverse of M is M_inv!
                # Wait: verify! cv2.warpPerspective(src, H, dsize) without WARP_INVERSE_MAP
                # computes dst(x,y) = src(H^{-1}(x,y)).
                # Forward: cv2.warpPerspective(hr_fwd, M_inv, (w, h)) -> samples hr_fwd at M(x_lr, y_lr).
                warp_rgb(grid_lr, block, (
                    hr_fwd_c, np.int32(oh), np.int32(ow),
                    sim_g, np.int32(h), np.int32(w),
                    np.float64(M[0, 0]), np.float64(M[0, 1]), np.float64(M[0, 2]),
                    np.float64(M[1, 0]), np.float64(M[1, 1]), np.float64(M[1, 2]),
                    np.float64(M[2, 0]), np.float64(M[2, 1]), np.float64(M[2, 2]),
                ))

                fr_g = cp.asarray(fr.rgb, dtype=cp.float32)
                if collect_vst:
                    g_obs = fr_g[:, :, 1]
                    g_sim = sim_g[:, :, 1]
                    valid = g_sim > cp.float32(1e-6)
                    if bool(cp.any(valid)):
                        dv = (cp.sqrt(cp.maximum(g_obs[valid], cp.float32(0.0)) + cp.float32(1e-4))
                              - cp.sqrt(cp.maximum(g_sim[valid], cp.float32(0.0)) + cp.float32(1e-4)))
                        last_vst_mads.append(float(cp.median(cp.abs(dv))))

                res_g = cp.ascontiguousarray(fr_g - sim_g)
                # Back-projection: cv2.warpPerspective(residual, M, (ow, oh)) -> samples residual at M_inv(x_hr, y_hr)
                warp_rgb(grid_hr, block, (
                    res_g, np.int32(h), np.int32(w),
                    up_g, np.int32(oh), np.int32(ow),
                    np.float64(M_inv[0, 0]), np.float64(M_inv[0, 1]), np.float64(M_inv[0, 2]),
                    np.float64(M_inv[1, 0]), np.float64(M_inv[1, 1]), np.float64(M_inv[1, 2]),
                    np.float64(M_inv[2, 0]), np.float64(M_inv[2, 1]), np.float64(M_inv[2, 2]),
                ))
                warp_1ch(grid_hr, block, (
                    np.int32(h), np.int32(w),
                    m_g, np.int32(oh), np.int32(ow),
                    np.float64(M_inv[0, 0]), np.float64(M_inv[0, 1]), np.float64(M_inv[0, 2]),
                    np.float64(M_inv[1, 0]), np.float64(M_inv[1, 1]), np.float64(M_inv[1, 2]),
                    np.float64(M_inv[2, 0]), np.float64(M_inv[2, 1]), np.float64(M_inv[2, 2]),
                ))
                wt_f = cp.float32(wt)
                if norm_pending:
                    norm_g += m_g * wt_f
                corr_g += up_g * (m_g * wt_f)[:, :, None]

            if norm_safe_g is None:
                norm_safe_g = cp.maximum(norm_g, cp.float32(1e-6))
            corr_g /= norm_safe_g[:, :, None]

            if adjoint and sigma_it and sigma_it > 0:
                corr_g = _gaussian_blur_rgb_gpu(cp, cp_nd, corr_g, sigma_it)
            if reg_sigma and reg_sigma > 0:
                corr_g = _gaussian_blur_rgb_gpu(cp, cp_nd, corr_g, float(reg_sigma))
            if max_step is not None and max_step > 0:
                cp.clip(corr_g, cp.float32(-max_step), cp.float32(max_step), out=corr_g)

            hr_g += cp.float32(lam) * corr_g
            cp.clip(hr_g, cp.float32(0.0), cp.float32(1.0), out=hr_g)
            if cov_g is not None:
                hr_g[cov_g < cp.float32(1e-6)] = cp.float32(0.0)
            del hr_fwd_g, corr_g

        del norm_g, norm_safe_g, sim_g, up_g, m_g

        # MAP-IBP VST bilateral proximal step on GPU
        if (reg_sigma and reg_sigma > 0
                and prox_noise_scale and prox_noise_scale > 0
                and prox_space_lr and prox_space_lr > 0):
            sigma_vst = _vst_burst_noise_sigma(frames, weights, residual_mads=last_vst_mads)
            ratio_scale = max(float(ratio) / 2.0, 1.0) ** 0.75
            sigma_c = float(prox_noise_scale) * ratio_scale * float(sigma_vst)
            sigma_s = float(prox_space_lr) * float(ratio)
            if sigma_c > 1e-7 and sigma_s > 1e-6:
                d = min(max(3, int(np.ceil(2.0 * sigma_s)) * 2 + 1), 11)
                eps_vst = cp.float32(1e-4)
                y_g = cp.sqrt(cp.maximum(hr_g, cp.float32(0.0)) + eps_vst)
                yf_g = _bilateral_rgb_gpu(cp, y_g, d, sigma_c, sigma_s)
                del y_g
                hr_g = cp.clip(yf_g * yf_g - eps_vst, cp.float32(0.0), cp.float32(1.0))
                del yf_g
                if cov_g is not None:
                    hr_g[cov_g < cp.float32(1e-6)] = cp.float32(0.0)

        out = cp.asnumpy(hr_g).astype(np.float32, copy=False)
        return out
    except Exception:
        free_gpu_memory()
        return None


# --------------------------------------------------------------------------- #
# GPU-resident Stage 3: Drizzle Scatter Deposition
# --------------------------------------------------------------------------- #

def _scatter_frame_into_gpu(
    cp,
    src_cpu: np.ndarray,
    M: np.ndarray,
    acc_g,
    cov_g,
    pixfrac: float,
    subsamples: int,
    fweight: float,
    rweight_cpu: Optional[np.ndarray] = None,
    local_du_cpu: Optional[np.ndarray] = None,
    local_dv_cpu: Optional[np.ndarray] = None,
) -> None:
    """Deposit one LR frame into GPU-resident ``acc_g`` and ``cov_g`` tables."""
    sh, sw = src_cpu.shape[:2]
    oh, ow = cov_g.shape[:2]
    n_sub = max(int(subsamples), 1)
    p = float(pixfrac) if (pixfrac is not None and pixfrac > 0) else 1.0
    offs = ((np.arange(n_sub, dtype=np.float64) + 0.5) / n_sub - 0.5) * p
    tap_w = float(fweight) / float(n_sub * n_sub)

    M64 = np.asarray(M, dtype=np.float64)
    if abs(M64[2, 2]) > 1e-12:
        M64 = M64 / M64[2, 2]

    src_g = cp.ascontiguousarray(cp.asarray(src_cpu, dtype=cp.float32))
    offs_g = cp.asarray(offs, dtype=cp.float64)
    dummy = cp.empty((1,), dtype=cp.float32)

    has_rw = rweight_cpu is not None
    rw_g = cp.ascontiguousarray(cp.asarray(rweight_cpu, dtype=cp.float32)) if has_rw else dummy
    has_loc = local_du_cpu is not None and local_dv_cpu is not None
    du_g = cp.ascontiguousarray(cp.asarray(local_du_cpu, dtype=cp.float32)) if has_loc else dummy
    dv_g = cp.ascontiguousarray(cp.asarray(local_dv_cpu, dtype=cp.float32)) if has_loc else dummy

    kern = _get_scatter_bilinear_kernel(cp)
    block = (16, 16)
    grid = ((sw + block[0] - 1) // block[0], (sh + block[1] - 1) // block[1])
    kern(grid, block, (
        src_g, np.int32(sh), np.int32(sw),
        acc_g, cov_g, np.int32(oh), np.int32(ow),
        np.float64(M64[0, 0]), np.float64(M64[0, 1]), np.float64(M64[0, 2]),
        np.float64(M64[1, 0]), np.float64(M64[1, 1]), np.float64(M64[1, 2]),
        np.float64(M64[2, 0]), np.float64(M64[2, 1]), np.float64(M64[2, 2]),
        offs_g, np.int32(n_sub), np.float64(tap_w),
        rw_g, np.int32(1 if has_rw else 0),
        du_g, dv_g, np.int32(1 if has_loc else 0),
    ))


def scatter_frames_gpu(
    frames: Sequence,
    aligns: Sequence,
    ratio: float,
    weights: Sequence[float],
    oh: int,
    ow: int,
    pixfrac: float,
    subsamples: int,
    splat: str = "bilinear",
    rweights: Optional[Sequence[Optional[np.ndarray]]] = None,
    local_shifts: Optional[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """GPU-resident drizzle scatter across all frames.

    Keeps ``acc (oh, ow, 3)`` and ``cov (oh, ow)`` in VRAM for the entire burst
    using 64-bit fixed-point ``atomicAdd`` (bit-for-bit deterministic across
    runs) and downloads them once at the end. Returns ``None`` on unsupported
    splat, insufficient VRAM, or CUDA error.
    """
    if splat != "bilinear":
        return None
    cp, _ = _import_cupy()
    if cp is None or not frames:
        return None

    sh, sw = frames[0].rgb.shape[:2]
    # Resident tables: acc (3*8) + cov (1*8) + one LR frame
    req_bytes = oh * ow * 8 * 4 + sh * sw * 4 * 8
    if not gpu_fits_vram(req_bytes):
        return None

    from .superres import to_drizzle_matrix

    try:
        acc_g = cp.zeros((oh, ow, 3), dtype=cp.uint64)
        cov_g = cp.zeros((oh, ow), dtype=cp.uint64)
        r_f = float(ratio)
        for j, (fr, al, wt) in enumerate(zip(frames, aligns, weights)):
            M = to_drizzle_matrix(al, ratio)
            rw = rweights[j] if rweights is not None else None
            du = (local_shifts[0][j] * r_f) if local_shifts is not None else None
            dv = (local_shifts[1][j] * r_f) if local_shifts is not None else None
            _scatter_frame_into_gpu(
                cp, fr.rgb, M, acc_g, cov_g,
                pixfrac=pixfrac, subsamples=subsamples,
                fweight=float(wt), rweight_cpu=rw,
                local_du_cpu=du, local_dv_cpu=dv,
            )
        inv_s = cp.float64(1.0 / 1073741824.0)
        acc = cp.asnumpy((acc_g * inv_s).astype(cp.float32))
        cov = cp.asnumpy((cov_g * inv_s).astype(cp.float32))
        return acc, cov
    except Exception:
        free_gpu_memory()
        return None
