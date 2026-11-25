/**
 * Metrics grid for the result report (SPEC §10).
 *
 * Surfaces the registration / quality metrics from `report.metrics` together
 * with the run summary from `report` (frames used, HR shape, ROI).
 *
 * The detail tile is deliberately the figure measured against ONE frame
 * (`hf_gain_db_vs_single`), not the core's `hf_gain_db` (which compares with
 * the denoised LR stack, i.e. acutance relative to an already-averaged
 * baseline). The stack-relative numbers live in the result panel's Advanced
 * section, where they cannot be read as a resolution "proof".
 *
 * Owned by AGENT-E.
 */
import type { Report } from '../../api/types'

export interface MetricsGridProps {
  report: Report
}

function num(v: number | null | undefined, digits = 2): string {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—'
}

function signed(v: number | null | undefined, digits = 2): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: 'pos' }) {
  return (
    <div className="metric">
      <div className="metric__label">{label}</div>
      <div className={`metric__value${tone === 'pos' ? ' metric__value--pos' : ''}`}>{value}</div>
    </div>
  )
}

export function MetricsGrid({ report }: MetricsGridProps) {
  const m = report.metrics
  const [hrH, hrW, hrC] = report.hr_shape
  const [, , roiW, roiH] = report.roi
  const detail = m.hf_gain_db_vs_single

  return (
    <div className="metrics-grid">
      <Metric
        label="Detail vs Single"
        value={`${signed(detail)} dB`}
        tone={typeof detail === 'number' && detail >= 0 ? 'pos' : undefined}
      />
      <Metric label="Sharpness vs Single" value={`${num(m.sharpness_gain_vs_single)}×`} />
      <Metric label="Mean Shift" value={`${num(m.mean_shift_px)} px`} />
      <Metric label="Max Shift" value={`${num(m.max_shift_px)} px`} />
      <Metric label="Registration" value={m.registration || '—'} />
      <Metric label="Frames" value={`${report.used_frames}/${report.total_frames}`} />
      <Metric label="SR Ratio" value={`${num(report.ratio, 1)}×`} />
      <Metric label="HR Size" value={`${hrW}×${hrH}`} />
      <Metric label="Channels" value={String(hrC)} />
      <Metric label="ROI" value={`${roiW}×${roiH}`} />
    </div>
  )
}
