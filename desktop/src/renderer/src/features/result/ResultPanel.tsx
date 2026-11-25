/**
 * Result panel: alignment verdict, before/after comparison (each pane
 * independently zoomable), run metrics, live non-destructive view transform
 * controls, the per-pixel quality map and export. SPEC §10.
 *
 * Layout order (top → bottom): warnings → Alignment → Comparison → Metrics →
 * View Transform → Quality map.
 *
 * Owned by AGENT-E.
 */
import type { Alignment, ResultPayload, ViewTransform } from '../../api/types'
import { Badge, Button, Card, EmptyState, Slider, ZoomPanView } from '../../ui'
import { AlignmentView } from './AlignmentView'
import { MetricsGrid } from './MetricsGrid'
import './result.css'

export interface ResultPanelProps {
  result: ResultPayload | null
  viewTransform: ViewTransform
  onViewTransformChange: (v: ViewTransform) => void
  onExport: (kind: 'master' | 'baked') => void
  busy: boolean
}

const TONE: Record<Alignment['level'], 'ok' | 'warn' | 'bad'> = {
  ok: 'ok',
  warn: 'warn',
  bad: 'bad'
}

/**
 * Compact fusion-engine text from the additive result-metrics keys
 * (`opencl_scatter`, `opencl_device` — see desktop/ARCHITECTURE.md). Returns
 * null when the backend reported no engine (older backend), so the panel
 * renders no line rather than guessing.
 */
function engineText(metrics: {
  fusion?: unknown
  opencl_scatter?: unknown
  opencl_device?: unknown
}): string | null {
  // The 2x default MAP solver runs on CUDA and never uses the OpenCL drizzle
  // scatter, so opencl_scatter reads 'unavailable' for it; name it instead.
  if (metrics.fusion === 'map') return 'MAP solver (CUDA GPU)'
  if (metrics.opencl_scatter === 'gpu') {
    const device =
      typeof metrics.opencl_device === 'string' && metrics.opencl_device
        ? metrics.opencl_device
        : null
    return device ? `GPU (OpenCL) — ${device}` : 'GPU (OpenCL)'
  }
  if (metrics.opencl_scatter === 'cpu') return 'CPU (OpenCL declined)'
  if (metrics.opencl_scatter === 'unavailable') return 'CPU (OpenCL unavailable)'
  return null
}

/**
 * One-line ghost-suppression verdict from the additive result-metrics keys
 * (`robust_active`, `robust_reason` — see desktop/ARCHITECTURE.md). Returns
 * null when the backend reported no verdict (older backend), so the panel
 * renders no line rather than guessing.
 */
function ghostText(metrics: {
  robust_active?: unknown
  robust_reason?: unknown
}): string | null {
  if (typeof metrics.robust_active !== 'boolean') return null
  const state = metrics.robust_active ? 'ON' : 'OFF'
  const reason =
    typeof metrics.robust_reason === 'string' && metrics.robust_reason
      ? metrics.robust_reason
      : null
  return reason ? `${state} — ${reason}` : state
}

/**
 * One-line IBP (iterative back-projection) configuration from the additive
 * `ibp_summary` / `ibp_iterations` metrics (see docs/IBP_AND_HELDOUT.md).
 * `ibp_summary` is '' when the core ran with IBP off. Returns null when the
 * backend reported neither key (older backend), so no line is rendered.
 */
function refinementText(metrics: {
  ibp_summary?: unknown
  ibp_iterations?: unknown
}): string | null {
  if (typeof metrics.ibp_summary === 'string' && metrics.ibp_summary.trim()) {
    return metrics.ibp_summary.trim()
  }
  if (typeof metrics.ibp_iterations === 'number' && Number.isFinite(metrics.ibp_iterations)) {
    return metrics.ibp_iterations > 0
      ? `${metrics.ibp_iterations} sweep${metrics.ibp_iterations === 1 ? '' : 's'}`
      : 'off (0 sweeps)'
  }
  return null
}

/**
 * Drizzle deposit configuration from the additive `drizzle_subsamples` /
 * `drizzle_pixfrac` metrics. `pixfrac` is the footprint in LR pixels, so the
 * sub-sample lattice is quoted per axis (2/3/3 at 2x/3x/4x by default).
 * Returns null when neither key is present (older backend).
 */
function depositText(metrics: {
  drizzle_subsamples?: unknown
  drizzle_pixfrac?: unknown
}): string | null {
  const parts: string[] = []
  if (
    typeof metrics.drizzle_subsamples === 'number' &&
    Number.isFinite(metrics.drizzle_subsamples)
  ) {
    parts.push(`${metrics.drizzle_subsamples} sub-samples/axis`)
  }
  if (
    typeof metrics.drizzle_pixfrac === 'number' &&
    Number.isFinite(metrics.drizzle_pixfrac)
  ) {
    parts.push(`pixfrac ${metrics.drizzle_pixfrac.toFixed(2)} LR px`)
  }
  return parts.length ? parts.join(' · ') : null
}

/** Non-empty metrics detail line (label + value), or nothing. */
function metricLine(label: string, value: string | null) {
  if (!value) return null
  return (
    <div className="metric" style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
      <span className="metric__label">{label}</span>
      <span className="metric__value" style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
        {value}
      </span>
    </div>
  )
}

function finite(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

function signed(v: unknown, digits = 2): string | null {
  return finite(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(digits)}` : null
}

function num(v: unknown, digits = 2): string | null {
  return finite(v) ? v.toFixed(digits) : null
}

/**
 * Compact per-frame noise estimate: the sigma of the noise the fusion weight
 * divided by (`metrics.frame_noise_sigma`, linear units), shown as a count and
 * a range so a mixed-noise burst is visible at a glance.
 */
function noiseText(metrics: { frame_noise_sigma?: unknown }): string | null {
  const raw = metrics.frame_noise_sigma
  if (!Array.isArray(raw)) return null
  const vals = raw.filter((v): v is number => finite(v) && v >= 0)
  if (!vals.length) return null
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const fmt = (v: number): string => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1))
  return `${vals.length} kept frame${vals.length === 1 ? '' : 's'} · σ ${fmt(lo)}${
    hi > lo ? `–${fmt(hi)}` : ''
  } (linear)`
}

/** Weight spread: max/min of the normalised fusion weights, 1.00 = uniform. */
function spreadText(metrics: { fusion_weight_spread?: unknown }): string | null {
  const v = metrics.fusion_weight_spread
  if (!finite(v) || v <= 0) return null
  return `${v.toFixed(2)}× (1.00 = uniform weighting)`
}

export function ResultPanel({ result, viewTransform, onViewTransformChange, onExport, busy }: ResultPanelProps) {
  if (!result) {
    return (
      <Card title="Result">
        <EmptyState icon="✨" title="No result yet" hint="Process a burst to see the super-resolved image." />
      </Card>
    )
  }

  const { report, images, alignment } = result
  const qualityMap = typeof alignment.map === 'string' ? alignment.map.trim() : ''
  const isMap = report.metrics.fusion === 'map'
  const engine = engineText(
    report.metrics as { fusion?: unknown; opencl_scatter?: unknown; opencl_device?: unknown }
  )
  const ghost = ghostText(report.metrics as { robust_active?: unknown; robust_reason?: unknown })
  const refinement = refinementText(
    report.metrics as { ibp_summary?: unknown; ibp_iterations?: unknown }
  )
  const deposit = depositText(
    report.metrics as { drizzle_subsamples?: unknown; drizzle_pixfrac?: unknown }
  )
  const noise = noiseText(report.metrics as { frame_noise_sigma?: unknown })
  const spread = spreadText(report.metrics as { fusion_weight_spread?: unknown })
  const detailVsSingle = report.metrics.hf_gain_db_vs_single
  const stackDetail = signed(report.metrics.hf_gain_db)
  const stackSharpness = num(report.metrics.sharpness_ratio)
  const memoryWarning =
    typeof report.memory_warning === 'string' && report.memory_warning.trim()
      ? report.memory_warning.trim()
      : null

  return (
    <Card
      title="Result"
      hint={`${images.width}×${images.height}`}
      actions={
        <div className="row">
          <Button size="sm" onClick={() => onExport('master')} disabled={busy}>
            Export Master (Linear)
          </Button>
          <Button size="sm" variant="primary" onClick={() => onExport('baked')} disabled={busy}>
            Export Baked
          </Button>
        </div>
      }
    >
      <div className="stack">
        {report.low_coverage_warning && (
          <div className="warning-banner">
            <span aria-hidden="true">⚠</span>
            <span>
              Low coverage: only <strong>{report.used_frames}</strong> of{' '}
              <strong>{report.total_frames}</strong> frames were usable. Super-resolution gain may be
              reduced.
            </span>
          </div>
        )}

        {report.roi_clipped && (
          <div className="result__notice">
            <span aria-hidden="true">ℹ</span>
            <span>The target ROI extended past the frame bounds and was clipped.</span>
          </div>
        )}

        {memoryWarning && (
          <div className="warning-banner" style={{ color: '#f4d58a' }}>
            <span aria-hidden="true">⚠</span>
            <span>
              <strong>Memory heads-up (not fatal):</strong> {memoryWarning} The result was produced
              — reduce the SR ratio or the ROI next time if the machine felt short on memory.
            </span>
          </div>
        )}

        {/* 1 — Alignment (top, whenever a result exists) */}
        <AlignmentView alignment={alignment} />

        <div className="divider" />

        {/* 2 — Comparison: two panes, each independently zoomable */}
        <div className="stack">
          <div className="row row--between">
            <h3 className="result__section-title">Comparison</h3>
            <span className="small faint">each pane zooms independently</span>
          </div>
          <div className="result__compare">
            <div className="result__pane">
              <h4 className="result__pane-title">Before (reference)</h4>
              <span className="small faint">
                One frame, {Number(report.ratio)}× upsampled for viewing — no sharpening or detail
                added; crisp grain here is sensor noise.
              </span>
              <ZoomPanView src={images.before} alt="Before — reference frame" />
            </div>
            <div className="result__pane">
              <h4 className="result__pane-title">After (super-resolved)</h4>
              <span className="small faint">
                {report.used_frames}/{report.total_frames} frames fused at {Number(report.ratio)}×
                — noise averaged out; extra resolution comes only from real sub-pixel shifts.
              </span>
              <ZoomPanView src={images.fused} alt="After — super-resolved result" />
            </div>
          </div>
        </div>

        <div className="divider" />

        {/* 3 — Metrics */}
        <MetricsGrid report={report} />
        <p className="small faint result__metric-note">
          <strong>Detail vs Single</strong> is the high-frequency energy above one frame&apos;s
          Nyquist versus that same frame bicubically upscaled — real added resolution, not
          sharpening. On a normal handheld burst the stricter held-out test (predict an unused frame
          vs a denoised stack of the rest) reads ≈ 0 dB, so no held-out gain is claimed here.
        </p>
        {metricLine('Fusion engine', engine)}
        {/* IBP and the drizzle deposit belong to the demosaic_drizzle path;
            a MAP result did not run them, so their lines are hidden. */}
        {!isMap && metricLine('Refinement (IBP)', refinement)}
        {!isMap && metricLine('Drizzle deposit', deposit)}
        {metricLine('Ghost suppression', ghost)}

        <details className="result__advanced">
          <summary>Advanced</summary>
          <div className="stack">
            {metricLine('Detail vs denoised stack (acutance)', stackDetail ? `${stackDetail} dB` : null)}
            {metricLine('Sharpness vs denoised stack', stackSharpness ? `${stackSharpness}×` : null)}
            {metricLine('Frame noise (fusion weight)', noise)}
            {metricLine('Weight spread', spread)}
            {detailVsSingle === null || detailVsSingle === undefined ? (
              <div className="small faint">
                Detail vs single unavailable: this backend did not report it.
              </div>
            ) : null}
            <div className="small faint">
              Held-out prediction gain is not computed in-app (it needs a full leave-one-out
              reconstruction). Measured on the real handheld burst it is ≈ 0 dB, and the older
              +3.6 dB figure was a border artefact of the whole-frame metric — see
              docs/IBP_AND_HELDOUT.md §7.
            </div>
          </div>
        </details>

        <div className="divider" />

        {/* 4 — View transform */}
        <div className="stack result__view">
          <div className="row row--between">
            <h3 className="result__section-title">View Transform</h3>
            <span className="small faint">Live · non-destructive</span>
          </div>
          <Slider
            label="Gamma"
            min={1}
            max={4}
            step={0.05}
            value={viewTransform.gamma}
            onChange={(gamma) => onViewTransformChange({ ...viewTransform, gamma })}
            format={(v) => v.toFixed(2)}
          />
          <Slider
            label="Brightness"
            min={-1}
            max={1}
            step={0.02}
            value={viewTransform.brightness}
            onChange={(brightness) => onViewTransformChange({ ...viewTransform, brightness })}
            format={(v) => v.toFixed(2)}
          />
          <Slider
            label="Contrast"
            min={0}
            max={2}
            step={0.02}
            value={viewTransform.contrast}
            onChange={(contrast) => onViewTransformChange({ ...viewTransform, contrast })}
            format={(v) => v.toFixed(2)}
          />
        </div>

        <div className="divider" />

        {/* 5 — Quality map (bottom) */}
        <div className="stack result__quality">
          <div className="row row--between">
            <h3 className="result__section-title">Quality map</h3>
            <div className="row">
              <span className="small faint">per-pixel alignment quality</span>
              <Badge tone={TONE[alignment.level]}>{alignment.level.toUpperCase()}</Badge>
            </div>
          </div>
          {qualityMap ? (
            <div className="canvas-wrap">
              <img src={qualityMap} alt="Per-pixel alignment-quality heatmap" />
            </div>
          ) : (
            <div className="result__notice">
              <span aria-hidden="true">ℹ</span>
              <span>No alignment-quality map was reported for this result.</span>
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}
