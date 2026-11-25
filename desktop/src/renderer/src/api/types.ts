/**
 * DTOs shared with the Python FastAPI backend.
 * Frozen — mirrors desktop/ARCHITECTURE.md. Do not change without updating both.
 */

export type Rect = [number, number, number, number] // x, y, w, h (source pixels)

export interface FrameInfo {
  id: string
  path: string
  name: string
  width: number
  height: number
  /** data:image/png;base64,... (max dimension 160) */
  thumbnail: string
  /** data:image/png;base64,... (max dimension 1024) */
  preview: string
  sharpness: number
  blurry: boolean
}

export interface RejectedFile {
  path: string
  reason: string
}

export interface FilesResponse {
  frames: FrameInfo[]
  rejected: RejectedFile[]
}

export interface MotionInfo {
  /** all shift values are in SOURCE pixels (scaled from the analysis preview) */
  avg_shift: number
  max_shift: number
  sufficient: boolean
  message: string
  shifts: Array<[number, number]>
}

export interface SrRatioInfo {
  requested: number
  achievable: number
  coverage_fraction: number
  message: string
}

export interface AnalyzeResponse {
  motion: MotionInfo
  sr_ratio: SrRatioInfo
  usable_frames: number
  total_frames: number
  per_frame_usable: boolean[]
}

export interface JobStatus {
  state: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  progress: number
  message: string
  error: string | null
  /** additive: 'memory' when the HR-size guard refused the run */
  error_kind?: string | null
}

export interface Metrics {
  hf_gain_db: number
  hf_gain_db_linear: number
  sharpness_ratio: number
  mean_shift_px: number
  max_shift_px: number
  registration: string
  /**
   * High-frequency band above the *single-frame* Nyquist, versus that frame
   * bicubically upscaled — the honest resolution figure (the acceptance-harness
   * convention). `hf_gain_db` above is measured against the denoised LR stack,
   * i.e. acutance relative to an already-averaged baseline.
   */
  hf_gain_db_vs_single?: number | null
  /** Laplacian-variance ratio versus one bicubic-upscaled frame */
  sharpness_gain_vs_single?: number | null
  /** per-kept-frame noise estimate the fusion weight used (linear units) */
  frame_noise_sigma?: number[] | null
  /** max/min of the normalised fusion weights; 1.0 = uniform weighting */
  fusion_weight_spread?: number | null
  /** number of IBP iterations that ran */
  ibp_iterations?: number
  /** display-ready IBP configuration line ('' when IBP is off) */
  ibp_summary?: string
  /** drizzle deposit footprint in LR pixels */
  drizzle_pixfrac?: number
  /** drizzle sub-samples per axis (2/3/3 at 2x/3x/4x by default) */
  drizzle_subsamples?: number
  /** index into the processed frames the HR grid is aligned to */
  reference_index?: number
  /** mean per-frame alignment score (ZNCC) over kept frames */
  mean_align_score?: number
  /** fusion the core ran: 'map' (2x default with a CUDA GPU) or 'demosaic_drizzle' etc. */
  fusion?: string
  /** why the core chose that fusion (e.g. 'auto: 2x with CUDA GPU') */
  fusion_reason?: string
}

export interface Report {
  used_frames: number
  total_frames: number
  ratio: number
  hr_shape: [number, number, number]
  roi: Rect
  roi_clipped: boolean
  metrics: Metrics
  low_coverage_warning: boolean
  /**
   * Non-fatal HR-size memory heads-up from the core's guard (the run still
   * completed). null/undefined when the run was comfortably inside the budget.
   */
  memory_warning?: string | null
}

export interface ViewTransform {
  gamma: number
  brightness: number
  contrast: number
}

export interface AlignmentFrame {
  id: string
  shift: [number, number]
  zncc: number
  residual: number
  used: boolean
}

export interface Alignment {
  score: number
  level: 'ok' | 'warn' | 'bad'
  message: string
  mean_zncc: number
  min_zncc: number
  per_frame: AlignmentFrame[]
  /** data URL: low-res per-pixel alignment-quality heatmap */
  map: string
}

export interface ResultPayload {
  report: Report
  images: {
    /** super-resolved result (data URL) */
    fused: string
    /** honest reference frame, rendered like the app preview (no extra sharpening) */
    before: string
    width: number
    height: number
  }
  alignment: Alignment
  view_transform: ViewTransform
}

export interface ProcessRequest {
  paths: string[]
  target_roi?: Rect
  sr_ratio: number
  registration?: 'homography' | 'translation'
  exclude?: string[]
  /** GPU acceleration via OpenCL (brand-agnostic). null/undefined = auto-detect. */
  use_opencl?: boolean | null
  /** Robust (Tukey) ghost suppression. null/undefined = auto (detect subject). */
  robust_fusion?: boolean | null
}

export interface ExportRequest {
  job_id: string
  kind: 'master' | 'baked'
  output_path: string
  view_transform?: ViewTransform
}

export interface RenderResponse {
  fused: string
  before: string
  width: number
  height: number
}
