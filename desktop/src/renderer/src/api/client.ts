/**
 * HTTP client for the Python FastAPI backend.
 * Frozen — the exported function signatures are the integration surface.
 */
import type {
  AnalyzeResponse,
  ExportRequest,
  FilesResponse,
  JobStatus,
  ProcessRequest,
  Rect,
  RenderResponse,
  ResultPayload,
  ViewTransform
} from './types'

let cachedBase: string | null = null

async function resolveBase(): Promise<string> {
  if (cachedBase) return cachedBase
  const fallback = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://127.0.0.1:8765'
  if (typeof window !== 'undefined' && window.desktop) {
    try {
      const port = await window.desktop.backendPort()
      cachedBase = `http://127.0.0.1:${port}`
      return cachedBase
    } catch {
      /* fall through to fallback */
    }
  }
  cachedBase = fallback
  return cachedBase
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const base = await resolveBase()
  const res = await fetch(`${base}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = (await res.json()) as { detail?: string }
      if (body?.detail) detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export const api = {
  health: () => request<{ status: string; version: string }>('/api/health'),

  addFiles: (paths: string[]) =>
    request<FilesResponse>('/api/session/files', {
      method: 'POST',
      body: JSON.stringify({ paths })
    }),

  analyze: (body: {
    paths: string[]
    target_roi?: Rect | null
    search_area?: Rect | null
    sr_ratio: number
    k_threshold?: number
  }) => request<AnalyzeResponse>('/api/analyze', { method: 'POST', body: JSON.stringify(body) }),

  process: (body: ProcessRequest) =>
    request<{ job_id: string }>('/api/process', { method: 'POST', body: JSON.stringify(body) }),

  jobStatus: (jobId: string) => request<JobStatus>(`/api/jobs/${jobId}`),

  cancel: (jobId: string) =>
    request<{ cancelled: boolean }>(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),

  result: (jobId: string) => request<ResultPayload>(`/api/jobs/${jobId}/result`),

  render: (jobId: string, viewTransform: ViewTransform) =>
    request<RenderResponse>(`/api/jobs/${jobId}/render`, {
      method: 'POST',
      body: JSON.stringify(viewTransform)
    }),

  export: (body: ExportRequest) =>
    request<{ output_path: string }>('/api/export', { method: 'POST', body: JSON.stringify(body) }),

  quality: (paths: string[], thresholdRatio?: number) =>
    request<{ flags: boolean[]; scores: number[] }>('/api/quality', {
      method: 'POST',
      body: JSON.stringify({ paths, threshold_ratio: thresholdRatio })
    })
}

export { resolveBase }
