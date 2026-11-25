import { useCallback, useEffect, useRef, useState } from 'react'
import { api, resolveBase } from './api/client'
import type {
  AnalyzeResponse,
  FrameInfo,
  JobStatus,
  Rect,
  RejectedFile,
  ResultPayload,
  ViewTransform
} from './api/types'
import { Badge, Button, ProgressBar, useToast } from './ui'
import { FilePanel } from './features/files/FilePanel'
import { ROIPanel } from './features/roi/ROIPanel'
import { ProcessPanel } from './features/process/ProcessPanel'
import { ResultPanel } from './features/result/ResultPanel'

const DEFAULT_VIEW: ViewTransform = { gamma: 2.2, brightness: 0, contrast: 1 }

// ---- GPU (OpenCL) preference -------------------------------------------------
// Persisted user choice; mapped to the request's `use_opencl` field:
//   'auto' -> null (backend/core auto-detects), 'on' -> true, 'off' -> false
type GpuMode = 'auto' | 'on' | 'off'
const GPU_MODES: GpuMode[] = ['auto', 'on', 'off']
const GPU_MODE_STORAGE_KEY = 'rawsr.gpuMode'

function loadGpuMode(): GpuMode {
  try {
    const saved = window.localStorage.getItem(GPU_MODE_STORAGE_KEY)
    if (saved === 'on' || saved === 'off') return saved
  } catch {
    /* storage unavailable (privacy mode) */
  }
  return 'auto'
}

// ---- Ghost suppression (robust fusion) preference ----------------------------
// Persisted user choice; mapped to the request's `robust_fusion` field:
//   'auto' -> null (core detects a moving subject / parallax and decides),
//   'on' -> true (always robust weights), 'off' -> false (plain fusion)
type GhostMode = 'auto' | 'on' | 'off'
const GHOST_MODES: GhostMode[] = ['auto', 'on', 'off']
const GHOST_MODE_STORAGE_KEY = 'rawsr.ghostMode'

function loadGhostMode(): GhostMode {
  try {
    const saved = window.localStorage.getItem(GHOST_MODE_STORAGE_KEY)
    if (saved === 'on' || saved === 'off') return saved
  } catch {
    /* storage unavailable (privacy mode) */
  }
  return 'auto'
}

interface Capabilities {
  opencl: boolean
  cpuCount: number
  /** Name of the OpenCL device the backend selected (null = none/older backend). */
  device: string | null
  /** Auto-gate verdict from engine_status: whether the fusion/scatter engine
   * will use the GPU ('gpu') or the CPU ('cpu'). null = older backend. */
  engineDecision: 'gpu' | 'cpu' | null
  /** One-line why for engineDecision (e.g. 'calibration: CPU faster',
   * 'pyopencl not importable in <exe>'). null = older backend. */
  engineReason: string | null
  /** Interpreter the backend runs under; shown when the engine is on the CPU. */
  pythonExecutable: string | null
  /** Whether pyopencl is importable in that interpreter. */
  pyopencl: boolean | null
}

export function App() {
  const { toast } = useToast()

  // Files
  const [frames, setFrames] = useState<FrameInfo[]>([])
  const [rejected, setRejected] = useState<RejectedFile[]>([])
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [excluded, setExcluded] = useState<Set<string>>(new Set())

  // ROI + ratio
  const [targetRoi, setTargetRoi] = useState<Rect | null>(null)
  const [searchArea, setSearchArea] = useState<Rect | null>(null)
  const [srRatio, setSrRatio] = useState(2)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Analysis feedback
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null)

  // Processing
  const [busy, setBusy] = useState(false)
  const [lastJobId, setLastJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [progressMessage, setProgressMessage] = useState('Ready.')
  const cancelRef = useRef(false)

  // GPU (OpenCL) preference, persisted; sent as `use_opencl` on process requests.
  const [gpuMode, setGpuMode] = useState<GpuMode>(loadGpuMode)
  useEffect(() => {
    try {
      window.localStorage.setItem(GPU_MODE_STORAGE_KEY, gpuMode)
    } catch {
      /* storage unavailable */
    }
  }, [gpuMode])
  const capabilities = useCapabilities()

  // Ghost suppression preference, persisted; sent as `robust_fusion` on process.
  const [ghostMode, setGhostMode] = useState<GhostMode>(loadGhostMode)
  useEffect(() => {
    try {
      window.localStorage.setItem(GHOST_MODE_STORAGE_KEY, ghostMode)
    } catch {
      /* storage unavailable */
    }
  }, [ghostMode])

  // Result
  const [result, setResult] = useState<ResultPayload | null>(null)
  const [viewTransform, setViewTransform] = useState<ViewTransform>(DEFAULT_VIEW)

  // Status slot that replaces the static-scene warning while a job runs.
  const [phase, setPhase] = useState<'idle' | 'complete' | 'error' | 'cancelled' | 'hidden'>(
    'idle'
  )
  const [phaseMessage, setPhaseMessage] = useState('')
  const completeTimerRef = useRef<number | null>(null)

  const activeFrames = frames.filter((f) => !excluded.has(f.id))
  const activePaths = activeFrames.map((f) => f.path)

  // The selected frame drives the ROI preview.
  const selectedFrame = frames.find((f) => f.id === selectedId) ?? frames[0] ?? null

  // Keep the selection valid as frames are added / removed.
  useEffect(() => {
    if (!frames.length) {
      setSelectedId(null)
      return
    }
    setSelectedId((prev) => (prev && frames.some((f) => f.id === prev) ? prev : frames[0].id))
  }, [frames])

  useEffect(
    () => () => {
      if (completeTimerRef.current) window.clearTimeout(completeTimerRef.current)
    },
    []
  )

  // Per-frame kept/dropped status is driven by the finished result.
  const perFrame: Record<string, { used: boolean; zncc: number }> | null = result
    ? Object.fromEntries(
        result.alignment.per_frame.map((a) => [a.id, { used: a.used, zncc: a.zncc }])
      )
    : null

  // ---- file selection ------------------------------------------------------
  const addFiles = useCallback(async () => {
    if (!window.desktop) {
      toast('Native file dialog is only available in the desktop app.', 'error')
      return
    }
    const paths = await window.desktop.openRawFiles()
    if (!paths.length) return
    setLoadingFiles(true)
    try {
      const res = await api.addFiles(paths)
      setFrames((prev) => {
        const seen = new Set(prev.map((f) => f.path))
        return [...prev, ...res.frames.filter((f) => !seen.has(f.path))]
      })
      setRejected((prev) => [...prev, ...res.rejected])
      if (res.rejected.length) {
        toast(`${res.rejected.length} file(s) rejected — not RAW or unreadable.`, 'error')
      }
    } catch (e) {
      toast(`Failed to load files: ${(e as Error).message}`, 'error')
    } finally {
      setLoadingFiles(false)
    }
  }, [toast])

  const removeFile = useCallback((id: string) => {
    setFrames((prev) => prev.filter((f) => f.id !== id))
  }, [])

  const toggleBlur = useCallback((id: string) => {
    setExcluded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const selectFrame = useCallback((id: string) => setSelectedId(id), [])

  // ---- live analysis (debounced) ------------------------------------------
  useEffect(() => {
    if (!activePaths.length) {
      setAnalysis(null)
      return
    }
    const handle = window.setTimeout(async () => {
      try {
        const res = await api.analyze({
          paths: activePaths,
          target_roi: targetRoi,
          search_area: searchArea,
          sr_ratio: srRatio
        })
        setAnalysis(res)
      } catch (e) {
        toast(`Analysis failed: ${(e as Error).message}`, 'error')
      }
    }, 350)
    return () => window.clearTimeout(handle)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames, excluded, targetRoi, searchArea, srRatio])

  // ---- processing ----------------------------------------------------------
  const process = useCallback(async () => {
    if (!activePaths.length) return
    if (completeTimerRef.current) {
      window.clearTimeout(completeTimerRef.current)
      completeTimerRef.current = null
    }
    setPhase('idle')
    setPhaseMessage('')
    setBusy(true)
    setResult(null)
    setProgress(0)
    setProgressMessage('Starting...')
    cancelRef.current = false
    try {
      const { job_id } = await api.process({
        paths: activePaths,
        target_roi: targetRoi ?? undefined,
        sr_ratio: srRatio,
        registration: 'homography',
        exclude: Array.from(excluded),
        use_opencl: gpuMode === 'auto' ? null : gpuMode === 'on',
        robust_fusion: ghostMode === 'auto' ? null : ghostMode === 'on'
      })
      setLastJobId(job_id)
      for (;;) {
        if (cancelRef.current) {
          await api.cancel(job_id)
        }
        const st: JobStatus = await api.jobStatus(job_id)
        setProgress(st.progress)
        setProgressMessage(st.message)
        if (st.state === 'done') {
          const payload = await api.result(job_id)
          setResult(payload)
          setViewTransform(payload.view_transform)
          setProgressMessage('Processing complete.')
          setPhase('complete')
          setPhaseMessage('Processing complete')
          completeTimerRef.current = window.setTimeout(() => {
            setPhase('hidden')
            completeTimerRef.current = null
          }, 3500)
          break
        }
        if (st.state === 'error') {
          // The backend classifies an HR-size refusal as error_kind='memory':
          // surface it as a fixable out-of-memory, not as a crash.
          const memory = st.error_kind === 'memory'
          const detail = st.error ?? 'Processing failed'
          setProgressMessage(memory ? 'Not enough memory.' : 'Error.')
          setPhase('error')
          setPhaseMessage(`${memory ? 'Not enough memory' : 'Processing failed'}: ${detail}`)
          toast(`${memory ? 'Not enough memory' : 'Processing error'}: ${detail}`, 'error')
          break
        }
        if (st.state === 'cancelled') {
          setProgressMessage('Cancelled.')
          setPhase('cancelled')
          setPhaseMessage('Processing cancelled.')
          break
        }
        await new Promise((r) => window.setTimeout(r, 400))
      }
    } catch (e) {
      toast(`Processing error: ${(e as Error).message}`, 'error')
      setProgressMessage('Error.')
      setPhase('error')
      setPhaseMessage(`Processing failed: ${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }, [activePaths, targetRoi, srRatio, excluded, gpuMode, ghostMode, toast])

  const cancel = useCallback(() => {
    cancelRef.current = true
    setProgressMessage('Cancelling…')
  }, [])

  // ---- live view transform -------------------------------------------------
  const onViewTransformChange = useCallback(
    async (vt: ViewTransform) => {
      setViewTransform(vt)
      if (!result || !lastJobId) return
      try {
        const rendered = await api.render(lastJobId, vt)
        setResult((prev) =>
          prev
            ? {
                ...prev,
                images: {
                  ...prev.images,
                  fused: rendered.fused,
                  before: rendered.before,
                  width: rendered.width,
                  height: rendered.height
                }
              }
            : prev
        )
      } catch (e) {
        toast(`Render failed: ${(e as Error).message}`, 'error')
      }
    },
    [result, lastJobId, toast]
  )

  // ---- export --------------------------------------------------------------
  const doExport = useCallback(
    async (kind: 'master' | 'baked') => {
      if (!result || !lastJobId) return
      if (!window.desktop) {
        toast('Export is only available in the desktop app.', 'error')
        return
      }
      const suggested = kind === 'master' ? 'sr_master.tiff' : 'sr_baked.tiff'
      const output = await window.desktop.saveFile(suggested)
      if (!output) return
      try {
        await api.export({
          job_id: lastJobId,
          kind,
          output_path: output,
          view_transform: kind === 'baked' ? viewTransform : undefined
        })
        toast(`Saved ${kind} to ${output}`, 'ok')
        await window.desktop.showItemInFolder(output)
      } catch (e) {
        toast(`Export failed: ${(e as Error).message}`, 'error')
      }
    },
    [result, lastJobId, viewTransform, toast]
  )

  const health = useHealth()

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">SR</div>
          <div>
            <h1>RAW Multi-Frame Super-Resolution</h1>
            <small>Classical multi-frame SR — no AI</small>
          </div>
        </div>
        <div className="spacer" />
        <GpuModeControl mode={gpuMode} onChange={setGpuMode} capabilities={capabilities} />
        <GhostModeControl mode={ghostMode} onChange={setGhostMode} />
        <Badge tone={health === 'ok' ? 'ok' : health === 'down' ? 'bad' : 'neutral'}>
          {health === 'ok' ? 'Engine ready' : health === 'down' ? 'Engine offline' : 'Connecting…'}
        </Badge>
        <Button variant="primary" onClick={addFiles} disabled={loadingFiles}>
          Add RAW Files…
        </Button>
      </header>

      {busy || phase !== 'hidden' ? (
        <div className="warning-slot" style={{ margin: '8px 20px 0' }}>
          {busy ? (
            <ProgressBar
              value={progress}
              label={progressMessage}
              indeterminate={busy && progress === 0}
            />
          ) : phase === 'complete' ? (
            <div
              className="warning-banner"
              style={{
                background: 'rgba(56, 193, 114, .1)',
                borderColor: 'rgba(56, 193, 114, .4)',
                color: 'var(--ok)'
              }}
            >
              <span aria-hidden="true">✓</span>
              <span>{phaseMessage}</span>
            </div>
          ) : phase === 'error' ? (
            <div
              className="warning-banner"
              style={{
                background: 'rgba(239, 83, 80, .09)',
                borderColor: 'rgba(239, 83, 80, .4)',
                color: '#f1a3a1'
              }}
            >
              <span aria-hidden="true">⚠</span>
              <span>{phaseMessage}</span>
            </div>
          ) : phase === 'cancelled' ? (
            <div className="warning-banner" style={{ color: '#f4d58a' }}>
              <span aria-hidden="true">⚠</span>
              <span>{phaseMessage}</span>
            </div>
          ) : (
            <div className="warning-banner">
              <span>⚠</span>
              <span>
                Designed for <strong>static scenes</strong> photographed with a moving camera. It does
                not produce reliable results for moving subjects (vehicles, people, animals, water,
                foliage in wind).
              </span>
            </div>
          )}
        </div>
      ) : null}

      <div className="app-body">
        <div className="col col--left stack">
          <FilePanel
            frames={frames}
            rejected={rejected}
            loading={loadingFiles}
            excluded={excluded}
            selectedId={selectedId}
            perFrame={perFrame}
            onAddFiles={addFiles}
            onRemoveFile={removeFile}
            onToggleBlur={toggleBlur}
            onSelectFrame={selectFrame}
          />
          <ProcessPanel
            srRatio={srRatio}
            onSrRatioChange={setSrRatio}
            achievable={analysis?.sr_ratio.achievable ?? null}
            motion={analysis?.motion ?? null}
            busy={busy}
            progress={progress}
            progressMessage={progressMessage}
            canProcess={activePaths.length > 0 && !busy}
            onProcess={process}
            onCancel={cancel}
          />
        </div>

        <div className="col col--right">
          <ROIPanel
            preview={selectedFrame?.preview ?? null}
            width={selectedFrame?.width ?? 0}
            height={selectedFrame?.height ?? 0}
            targetRoi={targetRoi}
            searchArea={searchArea}
            motion={analysis?.motion ?? null}
            onChange={(t, s) => {
              setTargetRoi(t)
              setSearchArea(s)
            }}
            usableFrames={analysis?.usable_frames ?? 0}
            totalFrames={analysis?.total_frames ?? frames.length}
          />
          <ResultPanel
            result={result}
            viewTransform={viewTransform}
            onViewTransformChange={onViewTransformChange}
            onExport={doExport}
            busy={busy}
          />
        </div>
      </div>
    </div>
  )
}

/**
 * Small Auto/On/Off segmented control for brand-agnostic GPU (OpenCL)
 * acceleration. Maps to `use_opencl` on the process request; the ✓/— hint
 * reflects GET /api/capabilities. Uses the Button primitive + theme tokens so
 * it matches the design system without touching frozen feature prop interfaces.
 */
function GpuModeControl({
  mode,
  onChange,
  capabilities
}: {
  mode: GpuMode
  onChange(mode: GpuMode): void
  capabilities: Capabilities | null
}) {
  const hint = (() => {
    const parts: string[] = []
    if (capabilities?.device) parts.push(`OpenCL device: ${capabilities.device}.`)
    parts.push(
      'Brand-agnostic GPU acceleration (OpenCL): the alignment warps and the fusion ' +
        'deposition run on the GPU when they measure faster; Auto decides per run.'
    )
    // When the engine is on the CPU, name the exact reason and the interpreter
    // that produced it — this is the "works for me, not for you" breadcrumb.
    if (capabilities?.engineDecision === 'cpu') {
      parts.push(`Engine: CPU${capabilities.engineReason ? ` — ${capabilities.engineReason}` : ''}.`)
      if (capabilities.pythonExecutable) parts.push(`Python: ${capabilities.pythonExecutable}.`)
    } else if (capabilities?.engineDecision === 'gpu' && capabilities.engineReason) {
      parts.push(`Engine: GPU — ${capabilities.engineReason}.`)
    }
    return parts.join(' ')
  })()
  return (
    <div
      className="row"
      style={{ gap: 8 }}
      title={hint}
    >
      <span className="tiny muted" style={{ whiteSpace: 'nowrap' }}>
        GPU (OpenCL){capabilities ? (capabilities.opencl ? ' ✓' : ' —') : ''}
      </span>
      <div
        role="group"
        aria-label="GPU (OpenCL) mode"
        style={{
          display: 'inline-flex',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-elev)',
          overflow: 'hidden'
        }}
      >
        {GPU_MODES.map((m) => (
          <Button
            key={m}
            size="sm"
            variant={mode === m ? 'primary' : 'ghost'}
            aria-pressed={mode === m}
            onClick={() => onChange(m)}
            style={{ border: 'none', borderRadius: 0, padding: '4px 10px' }}
          >
            {m === 'auto' ? 'Auto' : m === 'on' ? 'On' : 'Off'}
          </Button>
        ))}
      </div>
    </div>
  )
}

/**
 * Small Auto/On/Off segmented control for ghost suppression (robust Tukey
 * fusion). Maps to `robust_fusion` on the process request:
 * auto → null (core detects a moving subject / parallax), on → true (always
 * robust weights), off → false (plain fusion). The tooltip explains the
 * trade-off; matches the GPU control's style.
 */
function GhostModeControl({
  mode,
  onChange
}: {
  mode: GhostMode
  onChange(mode: GhostMode): void
}) {
  const hint =
    'Ghost suppression (robust fusion): across-frame Tukey weights down-weight ' +
    'pixels that disagree with the burst, removing ghost trails from a moving ' +
    'subject or parallax. Auto = only suppress when a moving subject is ' +
    'detected; On = always robust weights; Off = plain fusion. Robust weights ' +
    'can cost some fine detail on a still scene, so Auto stays honest.'
  return (
    <div className="row" style={{ gap: 8 }} title={hint}>
      <span className="tiny muted" style={{ whiteSpace: 'nowrap' }}>
        Ghost suppression
      </span>
      <div
        role="group"
        aria-label="Ghost suppression mode"
        style={{
          display: 'inline-flex',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-elev)',
          overflow: 'hidden'
        }}
      >
        {GHOST_MODES.map((m) => (
          <Button
            key={m}
            size="sm"
            variant={mode === m ? 'primary' : 'ghost'}
            aria-pressed={mode === m}
            onClick={() => onChange(m)}
            style={{ border: 'none', borderRadius: 0, padding: '4px 10px' }}
          >
            {m === 'auto' ? 'Auto' : m === 'on' ? 'On' : 'Off'}
          </Button>
        ))}
      </div>
    </div>
  )
}

/** Hardware capabilities from GET /api/capabilities; null until/unless available. */
function useCapabilities(): Capabilities | null {
  const [caps, setCaps] = useState<Capabilities | null>(null)
  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const base = await resolveBase()
        const res = await fetch(`${base}/api/capabilities`)
        if (!res.ok) return
        const body = (await res.json()) as {
          opencl?: unknown
          cpu_count?: unknown
          opencl_device?: unknown
          engine_status?: unknown
        }
        const engine = (body.engine_status ?? {}) as {
          gate_decision?: unknown
          gate_reason?: unknown
          python_executable?: unknown
          pyopencl?: unknown
        }
        if (alive) {
          setCaps({
            opencl: body.opencl === true,
            cpuCount: typeof body.cpu_count === 'number' ? body.cpu_count : 0,
            device:
              typeof body.opencl_device === 'string' && body.opencl_device
                ? body.opencl_device
                : null,
            engineDecision:
              engine.gate_decision === 'gpu' || engine.gate_decision === 'cpu'
                ? engine.gate_decision
                : null,
            engineReason:
              typeof engine.gate_reason === 'string' && engine.gate_reason
                ? engine.gate_reason
                : null,
            pythonExecutable:
              typeof engine.python_executable === 'string' && engine.python_executable
                ? engine.python_executable
                : null,
            pyopencl: typeof engine.pyopencl === 'boolean' ? engine.pyopencl : null
          })
        }
      } catch {
        /* endpoint unavailable (older backend) — hint stays hidden */
      }
    })()
    return () => {
      alive = false
    }
  }, [])
  return caps
}

function useHealth(): 'ok' | 'down' | 'connecting' {
  const [state, setState] = useState<'ok' | 'down' | 'connecting'>('connecting')
  useEffect(() => {
    let alive = true
    const check = async () => {
      try {
        await api.health()
        if (alive) setState('ok')
      } catch {
        if (alive) setState('down')
      }
    }
    check()
    const t = window.setInterval(check, 5000)
    return () => {
      alive = false
      window.clearInterval(t)
    }
  }, [])
  return state
}
