# raw_sr Desktop — Architecture & Frozen Interface Contract

Modern desktop UI for `raw_sr` (RAW multi-frame super-resolution).

**Stack:** Electron (main/preload) + React + TypeScript + Vite (renderer) +
Python **FastAPI** backend that wraps the existing, unchanged `raw_sr.core`
super-resolution engine.

The Python core (`raw_sr/core/*`) is **not** modified by this work. The old
PySide6 UI (`raw_sr/ui/*`, `raw_sr/main.py`) is left in place as a fallback but
is no longer the primary UI.

```
┌──────────────────────── Electron app ────────────────────────┐
│  main process (Node)                                         │
│    • spawns the Python backend on a free port                │
│    • native open/save dialogs (IPC)                          │
│    • BrowserWindow                                           │
│  renderer (React + TS)  ──HTTP──►  127.0.0.1:<port>          │
│                                      FastAPI backend         │
│                                        • loads RAW, jobs     │
│                                        • calls raw_sr.core   │
└──────────────────────────────────────────────────────────────┘
```

## Directory layout (ownership)

```
desktop/
  ARCHITECTURE.md            # this file (frozen contract)
  package.json               # electron + vite + react  [ORCHESTRATOR]
  electron.vite.config.ts    # [ORCHESTRATOR]
  tsconfig*.json             # [ORCHESTRATOR]
  electron-builder.yml       # [AGENT-B] packaging
  src/
    shared/ipc.ts            # shared IPC + DTO types     [ORCHESTRATOR]
    main/index.ts            # Electron main             [AGENT-B]
    preload/index.ts         # preload / contextBridge   [AGENT-B]
    renderer/
      index.html
      src/
        main.tsx             # [ORCHESTRATOR]
        App.tsx              # wiring + layout            [ORCHESTRATOR]
        styles/global.css    # design tokens / theme      [ORCHESTRATOR]
        api/types.ts         # DTOs shared with backend    [ORCHESTRATOR]
        api/client.ts        # HTTP client                 [ORCHESTRATOR]
        ui/                  # design-system primitives    [ORCHESTRATOR]
        features/
          files/             # names-only list + validation [AGENT-C]
          roi/               # zoomable ROI/search editing  [AGENT-D]
          process/           # SR ratio + controls         [AGENT-F]
          result/            # alignment, before/after, view [AGENT-E]
  backend/
    requirements.txt         # fastapi, uvicorn           [ORCHESTRATOR]
    __init__.py
    server.py                # FastAPI app                [AGENT-A]
    jobs.py                  # job manager/threads        [AGENT-A]
    imaging.py               # ndarray -> PNG data URL     [AGENT-A]
    session.py               # loaded-frame cache          [AGENT-A]
```

**Rule:** each agent edits ONLY the files it owns. Never touch another agent's
files. The ORCHESTRATOR-owned files are the integration surface: their exported
signatures are frozen — build against them, do not change them.

---

## Python backend HTTP API (base `http://127.0.0.1:<port>`)

All requests/responses JSON unless noted. Errors: `{ "detail": "..." }` with a
non-2xx status.

### `GET /api/health`
`200 { "status": "ok", "version": "1.0.0" }`

### `GET /api/capabilities`
Additive, non-breaking hardware probe (added with the OpenCL work):
`200 { "opencl": boolean, "cpu_count": number, "opencl_device": string|null,
        "opencl_vendor": string|null, "engine_status": EngineStatus }`

- `opencl` — whether an OpenCL device is available to OpenCV T-API
  (`cv2.ocl.haveOpenCL()`, cached at first call; brand-agnostic).
- `cpu_count` — host logical core count.
- `opencl_device` / `opencl_vendor` — name/vendor of the device OpenCV
  actually resolved for T-API (`cv2.ocl.Device.getDefault()`, same first-call
  cache; `null` when no device or an OpenCV binding without the class).
- `engine_status` — why the **fusion engine** will run on the GPU or the CPU,
  computed once at backend startup and cached (additive):

```ts
interface EngineStatus {
  python_executable: string;      // interpreter the backend runs under
  pyopencl: boolean;              // importable in that interpreter?
  pyopencl_error: string | null;  // import/probe failure detail, else null
  opencl_platforms: string[];     // pyopencl platform names (e.g. ["NVIDIA CUDA"])
  gpu_device: string | null;      // fastest GPU device pyopencl sees, else null
  nvidia_smi: boolean;            // nvidia-smi answers AND the ICD is registered
  pinned_device: string | null;   // OPENCV_OPENCL_DEVICE value the pin wrote
  gate_decision: 'gpu' | 'cpu';   // auto (no-override) fusion/scatter verdict
  gate_reason: string;            // one line, e.g. 'pyopencl not importable in <exe>',
                                  // 'no OpenCL GPU device (platforms: …)',
                                  // 'calibration: CPU faster',
                                  // 'GPU scatter faster on <device>'
}
```

This exists because the auto gate is silent and machine-specific: pyopencl may
be missing in the interpreter Electron picked (`RAW_SR_PYTHON`), no GPU device
may exist, or the one-time scatter-vs-CPU calibration (`gpu_scatter_beneficial`,
a wall-clock comparison with a 5% margin) may measure the CPU faster — the
documented outcome on some laptop GPUs, and *load-sensitive* (transient GPU
contention at launch can flip it, and the verdict is cached for the process's
lifetime). `engine_status` turns that invisible decision into a nameable cause;
the renderer's GPU (OpenCL) control shows `gate_reason` + `python_executable`
in its tooltip when the verdict is CPU.

The renderer's "GPU (OpenCL)" Auto/On/Off control shows a ✓/— hint from this
endpoint (device name in the tooltip) . The selection maps to `use_opencl` on
`POST /api/process`:
`auto → null` (core auto-detects), `on → true`, `off → false`. The backend
forwards the kwarg to `SuperResolution` only when the request set it non-None
**and** the installed core signature accepts it (`inspect.signature` guard);
`null` is never passed, so older cores and absent-GPU machines behave exactly
as before. The choice is persisted in `localStorage['rawsr.gpuMode']`
(default `'auto'`).

The renderer's "Ghost suppression" Auto/On/Off control sits beside it and maps
to `robust_fusion` on `POST /api/process`:
`auto → null` (the core detects a moving subject / parallax and decides),
`on → true` (always robust Tukey weights), `off → false` (plain fusion). Like
`use_opencl`, the backend forwards the kwarg to `SuperResolution` only when the
request set it non-None and the installed core accepts it; older cores behave
as before. Persisted in `localStorage['rawsr.ghostMode']` (default `'auto'`).

**Device pinning.** OpenCV reads `OPENCV_OPENCL_DEVICE` exactly once, at the
first UMat use in the process. At module import the backend therefore pins
`'NVIDIA CUDA:GPU:0'` iff `nvidia-smi -L` answers and (off Linux, or
`/etc/OpenCL/vendors/nvidia.icd` exists) and the variable is unset — a
deterministic device choice on hybrid-graphics laptops, inherited by spawned
core workers. A user-set value always wins. The gate matters: OpenCV's
`selectOpenCLDevice` *disables* OpenCL when the requested device does not
exist, so the pin is only ever written after a positive NVIDIA probe. Note
this machine-class finding (measured): on an RTX 3050 Laptop GPU the routed
upsample warp is slower than the CPU (0.09x at 1024², 0.63–0.82x at 4096²)
even with the right device selected, so honest auto-calibration correctly
keeps such laptops on the CPU — `opencl_device` exists so the UI can say
*which* device that verdict was measured against.

### `POST /api/session/files`
Body: `{ "paths": string[] }`
Validates RAW files, decodes a small **thumbnail** and a larger **preview**, and
flags blurred frames. Rejected (non-RAW / undecodable) files are listed
separately, never fail the whole request. More than 24 paths is rejected with
`400` (burst cap — see *Resource guardrails*).

`200`:
```ts
{
  frames: Array<{
    id: string;            // stable index as string, e.g. "0"
    path: string;
    name: string;          // basename
    width: number;         // full sensor width
    height: number;
    thumbnail: string;     // data:image/png;base64,... (max dimension 160)
    preview: string;       // data:image/png;base64,... (max dimension 1024)
    sharpness: number;     // Laplacian variance (display transformed)
    blurry: boolean;       // sharpness < 0.5 * best
  }>;
  rejected: Array<{ path: string; reason: string }>;
}
```
RAW extensions accepted: `.dng .cr2 .cr3 .nef .nrw .arw .srf .sr2 .raf .rw2 .orf .pef .raw .tif .tiff`.

### `POST /api/session/preview`
Body: `{ "path": string, "max_dim": number }`
For re-rendering a preview at a requested size (rarely needed; `session/files`
already returns previews). `max_dim` is validated to `16..2048` — the UI never
asks above 1024, and the bound caps how large one decode-cache entry can grow
(see *Resource guardrails*). Out-of-range `max_dim` returns `400`.
`200 { "preview": string, "width": number, "height": number }`

### `POST /api/analyze`
Body:
```ts
{ "paths": string[],
  target_roi?: [number, number, number, number],   // x,y,w,h (source px)
  search_area?: [number, number, number, number],
  sr_ratio: number,
  k_threshold?: number }                            // default 3.0
```
Runs **fast** analysis on downscaled previews (motion + achievable SR + usable
frame count). Used for live feedback while the user edits the ROI.

`200`:
```ts
{
  motion: {
    // ALL shift values below are in SOURCE pixels (scaled up from the
    // analysis preview), so the ROI panel can use them directly.
    avg_shift: number; max_shift: number; sufficient: boolean;
    message: string;
    shifts: Array<[number, number]>;
  };
  sr_ratio: {
    requested: number; achievable: number;
    coverage_fraction: number; message: string;
  };
  usable_frames: number;      // frames usable inside the ROI (motion-suitable)
  total_frames: number;
  per_frame_usable: boolean[]; // aligned with `paths`
}
```

### `POST /api/process`
Body:
```ts
{ "paths": string[],
  target_roi?: [number, number, number, number],
  sr_ratio: number,
  registration?: "homography" | "translation",   // default "homography"
  exclude?: string[],                             // frame ids the user un-flagged
  use_opencl?: boolean | null,                     // default null = auto (GPU via OpenCL when a device exists)
  robust_fusion?: boolean | null,                  // default null = auto (suppress ghosting only when a moving subject is detected)
  memory_budget_bytes?: number | null }            // default null = 60% of MemAvailable, clamped 2-6 GiB
```
Starts a background job; returns immediately.
`200 { "job_id": string }`

**413 — memory guard.** Before decoding, the endpoint estimates the HR peak
(≈160 B per HR pixel) for the resolved ROI at the requested ratio. Above the
budget it refuses with
`413 { "detail": "Not enough memory for this request: HR output … Reduce the SR ratio or the ROI and try again, or raise the memory budget (memory_budget_bytes)." }`
and starts no job. A run that passes the pre-check but later raises
`MemoryError` inside the core ends as `state: "error"` with a readable message
and `error_kind: "memory"`; a run between 75 % and 100 % of the budget
completes and carries `report.memory_warning`.

**409 — single flight.** A process job keeps every decoded frame resident, so
two concurrent process jobs would double the peak. While a process job is
running, a second `POST /api/process` returns
`409 { "detail": "A process job is already running" }`. The guard is released
when the in-flight job reaches any terminal state (done / error / cancelled) —
including cancellation that races job startup — and is visible-free at the
moment the job's terminal state becomes observable, so a client that reacts to
`done` can start the next run without retrying. `/api/dev/synthetic` (test
helper) is not guarded. The renderer already surfaces the `detail` as a toast.

### `GET /api/jobs/{job_id}`
```ts
{ "state": "queued" | "running" | "done" | "error" | "cancelled",
  "progress": number,      // 0..100
  "message": string,
  "error": string | null,
  "error_kind": "memory" | null }   // additive: HR-size guard refused the run
```

### `POST /api/jobs/{job_id}/cancel`
`200 { "cancelled": true }`

### `GET /api/jobs/{job_id}/result`
Only valid when state == "done".
```ts
{
  report: {
    used_frames: number; total_frames: number; ratio: number;
    hr_shape: [number, number, number];
    roi: [number, number, number, number]; roi_clipped: boolean;
    metrics: {
      hf_gain_db: number; hf_gain_db_linear: number;
      sharpness_ratio: number; mean_shift_px: number;
      max_shift_px: number; registration: string;
      hf_gain_db_vs_single: number | null;                  // honest detail vs ONE frame
      sharpness_gain_vs_single: number | null;              // Laplacian ratio vs that frame
      frame_noise_sigma: number[] | null;                   // per kept frame (linear)
      fusion_weight_spread: number | null;                  // max/min weights, 1.0 = uniform
      ibp_iterations?: number; mean_align_score?: number;   // extra info
      ibp_summary?: string;                                 // IBP config, '' = off
      drizzle_pixfrac?: number;                             // deposit footprint (LR px)
      drizzle_subsamples?: number;                          // sub-samples per axis
      reference_index?: number;                             // frame `hr` is aligned to
      opencl_scatter: "cpu" | "gpu" | "unavailable";        // fusion engine
      opencl_device: string | null;                         // GPU name, if used
      opencl_reason: string;                                // why GPU or CPU
      robust_active: boolean;                               // ghost suppression on?
      robust_reason: string;                                // why on/off (auto/on/off)
    };
    low_coverage_warning: boolean;
    memory_warning: string | null;   // non-fatal HR-size heads-up, null if comfortable
  };
  images: {
    fused: string;      // data URL, view-transformed preview of the HR result
    before: string;     // data URL, reference frame upscaled to HR size (baseline)
    width: number; height: number;   // of the preview (display) image
  };
  alignment: {           // "did the frames align, and how well?"
    score: number;       // 0..1 overall alignment success
    level: "ok" | "warn" | "bad";
    message: string;     // short human-readable verdict
    mean_zncc: number;   // mean per-frame zero-normalized cross-correlation
    min_zncc: number;
    per_frame: Array<{
      id: string;        // frame id, as returned by /api/session/files
      shift: [number, number];
      zncc: number;      // 0..1 alignment quality for this frame
      residual: number;  // alignment residual in px
      used: boolean;     // kept after quality filtering
    }>;
    map: string;         // data URL: low-res per-pixel alignment-quality heatmap
  };
  view_transform: { gamma: number; brightness: number; contrast: number };
}
```

> **Fusion-engine keys (additive):** `opencl_scatter` names the engine that
> deposited the fusion — `"gpu"` ran on the GPU (`opencl_device` carries its
> name), `"cpu"` means the CPU measured faster or GPU use was turned off, and
> `"unavailable"` means no OpenCL device (or a core build predating the keys).
> `opencl_reason` is a one-line explanation of the verdict recorded per job
> (`'pyopencl not importable in <exe>'`, `'no OpenCL GPU device …'`,
> `'calibration: CPU faster'`, `'GPU accelerated on <device>'`,
> `'forced by request'`, …); it mirrors `engine_status.gate_reason` from
> `/api/capabilities` but reflects *this* run. The result panel renders one
> compact “Fusion engine: …” line from these when present; every other metrics
> field is unchanged.

> **Ghost-suppression keys (additive):** `robust_active` (bool) says whether
> the robust Tukey weights were used for this run, and `robust_reason` is the
> one-line why (`'auto: moving subject detected'`, `'forced on by request'`,
> `'forced off by request'`, `'auto: plain fusion …'`, `'unavailable'` for a
> core build predating the keys). They mirror the renderer's Ghost suppression
> Auto/On/Off control and are recorded per job; the result panel renders one
> compact “Ghost suppression: ON/OFF — <reason>” line from them, and every other
> metrics field is unchanged.

> **Refinement / deposit keys (additive):** `ibp_summary` is the core's
> display-ready IBP configuration (e.g. `1 sweep x lambda 0.50, adjoint,
> forward sigma 0.58 HR px`; `""` when IBP is off), and `ibp_iterations` is the
> sweep count that actually ran (default changed 3 → 1 — compare runs only with
> this key in hand). `drizzle_pixfrac` is the deposit footprint in LR pixels and
> `drizzle_subsamples` the sub-sample lattice per axis; `reference_index` names
> the frame `hr` is aligned to (also the held-out baseline). A core that predates
> them yields `""`/`null`, and the panel omits the corresponding line rather than
> printing a configuration that did not run. The result panel renders
> “Refinement (IBP): …” and “Drizzle deposit: N sub-samples/axis · pixfrac …
> LR px” from these. See `docs/IBP_AND_HELDOUT.md`.
>
> **Honest detail keys (additive):** `hf_gain_db_vs_single` and
> `sharpness_gain_vs_single` measure the HR result against **one frame**
> bicubically upscaled (the `scripts/sr_acceptance.py` convention), not against
> the denoised LR stack that `hf_gain_db` / `sharpness_ratio` use. The stack is
> already an average, so those two read as *acutance*; only the single-frame
> pair is evidence of recovered resolution. The result panel's headline tile
> uses the single-frame pair, and the stack pair is shown under *Advanced* as
> “Detail vs denoised stack (acutance)”. Both single-frame keys are `null` for
> a build without the measuring helpers; the panel then prints “unavailable”.

> **Fusion-weight keys (additive):** `frame_noise_sigma` is the per-kept-frame
> noise estimate (linear units, same order as `kept_indices`) that the fusion
> weight divided by, and `fusion_weight_spread` is max/min of the normalised
> weights (`1.0` = uniform). Both are `null` for a core that predates them. The
> result panel shows them compactly under *Advanced* (σ range + count, spread).

> **Memory guard keys (additive):**
> `report.memory_warning` is the core's HR-size `RuntimeWarning` (a run between
> 75 % and 100 % of the budget that still completed), `null` when the run was
> comfortable; the panel renders it as a non-fatal note above the comparison.
> A run *above* the budget never starts: `POST /api/process` refuses it with
> **413** and an actionable `detail`, and a job-time `MemoryError` becomes a
> readable `error` with `error_kind: "memory"` (never a traceback). The
> optional request field `memory_budget_bytes` raises the ceiling; the default
> is 60 % of `MemAvailable`, clamped 2–6 GiB.

> **Coverage map:** the desktop backend never labels `res.coverage` as an “SNR
> map”; the legacy PySide UI (`raw_sr/ui/worker.py:142`) does (`snr_map` is the
> drizzle coverage array). The desktop's bottom heatmap is the *alignment
> quality* map (`alignment.map`), which is a different quantity.

> **Removed:** the four confidence layers (coverage / snr / registration /
> combined) were dropped — they added cost without answering "did the frames
> align?". They are replaced by `alignment` above (one map instead of four).

> **Retention:** only the **most recent completed processed job** keeps its
> full-resolution result arrays in memory. Older completed jobs still answer
> this endpoint in full (report + display images + alignment are always kept);
> see *Resource guardrails*.

### Core interface additions (`raw_sr.core.superres`)
The SR engine now exposes alignment quality so the backend can report it
without recomputation. `SRResult` gains:
```ts
kept_indices: number[];     // indices into the input `frames` list (kept frames)
align_scores: number[];     // ZNCC in [0,1] per kept frame (aligned to kept_indices)
align_residuals: number[];  // alignment residual in px per kept frame
```
The core also gains genuine classical clarity improvements (iterative
back-projection, reduced post-fusion blur, sharper demosaic option) — still no
AI and no fabricated detail (SPEC §1).

### `POST /api/jobs/{job_id}/render`
Re-applies a view transform to the preview (non-destructive; master untouched).
Body `{ "gamma": number, "brightness": number, "contrast": number }`
`200 { "fused": string, "before": string, "width": number, "height": number }` (data URLs)

> `before` and `fused` MUST both be returned so the comparison stays consistent
> once a view-transform slider moves. `images.before` is the **honest reference
> frame**: it is rendered with the *same* demosaic + preview pipeline the app
> uses elsewhere and gets **no extra sharpening**; it must look like the original
> RAW frame, not like a processed derivative.

### `POST /api/export`
Body:
```ts
{ "job_id": string,
  "kind": "master" | "baked",
  "output_path": string,                 // chosen via Electron save dialog
  "view_transform"?: { gamma: number; brightness: number; contrast: number } }
```
`200 { "output_path": string }`
`master` is 16-bit linear TIFF; `baked` applies the view transform.
Export needs the full HR array, which is retained only for the most recent
completed processed job. Exporting an **older, already-evicted** job returns
`409` with a `detail` explaining that the result was evicted and processing
should be re-run; the job's result/render endpoints keep working.

### `POST /api/quality`
Body `{ "paths": string[], "threshold_ratio"?: number }`
Recomputes blur flags after a user override. `200 { "flags": boolean[], "scores": number[] }`

---

## Resource guardrails (memory-audit fixes)

Static memory audit findings, enforced at the backend boundary. All limits
live in `desktop/backend/server.py` / `jobs.py`.

- **Frame cap (`MAX_FRAMES = 24`).** The product is a 2–20-frame burst tool;
  24 keeps headroom. Every endpoint that takes `paths` (`/api/session/files`,
  `/api/analyze`, `/api/quality`, `/api/process`) rejects an over-cap list
  with `400 Too many frames (N > 24)` via one shared helper, before any
  decode. (Each loaded frame pins its full decoded base — the "F-pin" — so an
  uncapped list meant an uncapped request-time peak.)
- **Preview `max_dim` bounded to `16..2048`.** The UI asks for ≤1024; the
  previous 8192 bound let one canonical decode-cache entry grow to hundreds of
  MB (worst-case ~2048²·3ch·fp32 ≈ 50 MB now; typical ≤12 MB). Violations are
  validation `400`s.
- **Single-flight `/api/process`.** See the endpoint section above: one
  process job at a time, `409` otherwise, guard released on every terminal
  path via `JobManager.create(..., on_finish=...)`.
- **Frames freed after the core run.** `/api/process` releases the full-res
  `frames` list as soon as `SuperResolution.run` returns — before/report
  phases keep only the detached `RawFrame.green` copies plus `before_rgb`,
  which is a fresh preview-pipeline decode (`session.load_reference_linear`)
  or, in the fallback, an explicit `np.array(frames[0].rgb)` **copy**. The
  stored job therefore never pins a full decoded frame. (During `run` itself
  the F-pin of N frames remains until the deferred ROI-detach change lands in
  `raw_sr.core.superres` — out of desktop scope.)
- **Heavy-result eviction (`JobManager`, `keep_heavy_results=1`).** A stored
  result owns `hr` (HR float32) + `coverage` + `before_rgb` ≈ 345 MB/job.
  The manager keeps these for only the K most recent **completed process**
  jobs (LRU, K configurable, default 1). Evicted jobs — and completed
  non-result jobs, which occupy no slot — keep their report, display arrays,
  alignment and base64 images, so `GET /api/jobs/{id}/result` and `/render`
  keep rendering recent history; only the heavy **export** path notices an
  eviction, with a graceful `409`. Cancelled/errored jobs drop heavy arrays
  immediately (they never serve a result).

---

## Electron IPC (`window.desktop`, via preload contextBridge)

```ts
interface DesktopBridge {
  backendPort(): Promise<number>;
  openRawFiles(): Promise<string[]>;          // native open dialog, RAW filter
  saveFile(defaultName: string): Promise<string | null>;
  showItemInFolder(path: string): Promise<void>;
  isElectron: true;
}
```
`src/shared/ipc.ts` declares `DesktopBridge` and `window.desktop` typing.
Frontend must degrade gracefully if `window.desktop` is missing (plain browser
dev): `openRawFiles` falls back to a hidden `<input type="file">` is NOT
required; instead show a message. `saveFile` falls back to a prompt.

---

## Design system (ORCHESTRATOR-owned, frozen exports)

`src/renderer/src/ui/index.ts` re-exports:

- `Button` — `{ variant?: 'primary'|'secondary'|'ghost'|'danger'; size?: 'sm'|'md'; disabled?; onClick?; children }`
- `IconButton` — `{ title: string; onClick?; children }`
- `Card` — `{ title?: string; actions?: ReactNode; children; className? }`
- `Slider` — `{ label: string; min: number; max: number; step: number; value: number; onChange(v:number): void; format?(v:number):string }`
- `Toggle` — `{ checked: boolean; onChange(b:boolean): void; label?: string }`
- `Tabs` — `{ tabs: {id:string;label:string}[]; active: string; onChange(id:string):void }`
- `ProgressBar` — `{ value: number; label?: string; indeterminate?: boolean }`
- `Spinner`
- `Badge` — `{ tone?: 'neutral'|'warn'|'ok'|'bad'; children }`
- `NumberField` — `{ label: string; value: number; min?; max?; step?; onChange(v:number):void }`
- `Field` — `{ label: string; hint?: string; children }`
- `EmptyState` — `{ title: string; hint?: string; icon?: ReactNode }`
- `Toast`/`useToast` — `toast(message: string, tone?: 'info'|'error'|'ok')`

Theme tokens live in `styles/global.css` as CSS variables (`--bg`, `--panel`,
`--text`, `--muted`, `--accent`, `--border`, `--radius`, spacing). Dark,
modern, minimal. Components use tokens, never hard-coded colors.

---

## Frontend state (ORCHESTRATOR-owned: `App.tsx`)

`App` owns all state and passes props to feature components. Feature components
are **presentational + local interaction only**; they call callbacks from props.
Frozen feature props:

```ts
// features/files/FilePanel.tsx
// Renders file NAMES only (no thumbnails). Kept/dropped status is shown under
// each name in the same cell once processing has finished.
interface FilePanelProps {
  frames: FrameInfo[]; rejected: RejectedFile[]; loading: boolean;
  excluded: Set<string>;            // frames the user un-flagged (blur override)
  selectedId?: string | null;        // frame currently shown in the ROI panel
  perFrame?: Record<string, { used: boolean; zncc: number }> | null;  // from result.alignment
  onAddFiles(): void;
  onRemoveFile(id: string): void;
  onToggleBlur(id: string): void;    // user override of auto blur flag
  onSelectFrame?(id: string): void;  // open this frame in the ROI panel
}

// features/roi/ROIPanel.tsx
interface ROIPanelProps {
  preview: string | null; width: number; height: number;
  targetRoi: Rect | null; searchArea: Rect | null;
  motion: MotionInfo | null;         // per-frame shifts -> reset placement + auto search area
  onChange(target: Rect | null, search: Rect | null): void;
  usableFrames: number; totalFrames: number;   // live "N/total frames usable"
}

// features/process/ProcessPanel.tsx
// NOTE: the progress bar now lives in the static-scene warning slot in App
// (the warning text is hidden while a job runs and replaced by progress; on
// success a green "completed" line is shown, then the whole slot is hidden).
interface ProcessPanelProps {
  srRatio: number; onSrRatioChange(v: number): void;
  achievable: number | null;                   // live requested-vs-achievable
  motion: MotionInfo | null;
  busy: boolean; progress: number; progressMessage: string;
  canProcess: boolean;
  onProcess(): void; onCancel(): void;
}

// features/result/ResultPanel.tsx
// Layout order: Alignment (top, once done) -> before/after side-by-side (each
// independently zoomable) -> metrics -> view transform -> Quality map (bottom).
interface ResultPanelProps {
  result: ResultPayload | null;
  viewTransform: ViewTransform;
  onViewTransformChange(v: ViewTransform): void;
  onExport(kind: 'master' | 'baked'): void;
  busy: boolean;
}

// ui/ZoomPanView.tsx (used by the result before/after panes; the ROI panel
// implements its own zoom/pan because its editable overlays must transform
// with the image)
interface ZoomPanViewProps {
  src: string; alt?: string;
  minZoom?: number; maxZoom?: number;   // default 1..12 (1 = fit)
  ctrlToZoom?: boolean;                 // when true, wheel zooms only while Ctrl is held
  controls?: boolean;                   // show the Fit / 1:1 toolbar (default true)
}
```

`Rect = [x, y, w, h]`. All DTO types come from `api/types.ts` (frozen).

---

## Launch diagnostics (`run.sh`)

`run.sh` prints a short launch summary after the venv is ready, before any
dependency-install or Electron launch work:

```
==> Launch summary
    python:   <interpreter>              # RAW_SR_PYTHON if set, else repo venv
    pyopencl: OK | MISSING (<error>)     # MISSING also prints the pip fix command
    opencl:   <device> | none (…)
```

It is diagnostic only — a missing pyopencl warns and the app still runs on the
CPU. Set `RAW_SR_PREFLIGHT_ONLY=1` to print this summary and exit without
launching (used by the backend test suite and handy for "is my GPU seen?").

---

## Verification bar

- `npm run typecheck` clean.
- `npm run build` succeeds.
- Backend: `python -m pytest desktop/backend/tests -q` (AGENT-A adds tests) and
  `GET /api/health` responds.
- End-to-end (ORCHESTRATOR): `desktop/scripts/e2e-smoke.mjs` boots the backend
  against a synthetic/mock burst, drives the API, asserts every SPEC §10 feature
  path returns data. The synthetic DNG-free path uses `scripts/smoke.py`-style
  synthetic frames exposed via a `/api/dev/synthetic` helper (AGENT-A).
