/**
 * Process panel: free-form SR ratio with live requested-vs-achievable feedback,
 * motion sufficiency message, and the Process / Cancel actions. SPEC §10.
 *
 * Progress is no longer rendered here — App owns the progress bar in the
 * static-scene warning slot (see ARCHITECTURE.md).
 *
 * Owned by AGENT-F.
 */
import type { MotionInfo } from '../../api/types'
import { Badge, Button, Card } from '../../ui'
import { MotionFeedback } from './MotionFeedback'
import { SrRatioInput, formatRatio } from './SrRatioInput'
import './process.css'

export interface ProcessPanelProps {
  srRatio: number
  onSrRatioChange: (v: number) => void
  achievable: number | null
  motion: MotionInfo | null
  busy: boolean
  progress: number
  progressMessage: string
  canProcess: boolean
  onProcess: () => void
  onCancel: () => void
}

type Tone = 'neutral' | 'ok' | 'warn' | 'bad'

function achievableTone(fraction: number | null): Tone {
  if (fraction == null) return 'neutral'
  if (fraction >= 0.95) return 'ok'
  if (fraction >= 0.8) return 'warn'
  return 'bad'
}

function achievableNote(fraction: number | null): string {
  if (fraction == null) return 'Measuring achievable resolution…'
  if (fraction >= 0.95) return 'Motion supports the requested ratio.'
  if (fraction >= 0.8) return 'Near the limit — output may be slightly softer than requested.'
  return 'Insufficient subpixel diversity; lower the ratio or add well-shifted frames.'
}

export function ProcessPanel({
  srRatio,
  onSrRatioChange,
  achievable,
  motion,
  busy,
  canProcess,
  onProcess,
  onCancel
}: ProcessPanelProps) {
  const safeRatio = srRatio > 0 ? srRatio : 1
  const supported = achievable != null ? achievable / safeRatio : null
  const tone = achievableTone(supported)

  return (
    <Card title="Super-Resolution">
      <div className="stack">
        <SrRatioInput value={srRatio} onChange={onSrRatioChange} disabled={busy} />

        {/* Requested vs achievable ------------------------------------------- */}
        <div className="sr-feedback">
          <div className="sr-feedback__head">
            <span className="field__label">Requested vs achievable</span>
            {achievable != null && (
              <Badge tone={tone}>achievable ~{formatRatio(achievable)}</Badge>
            )}
          </div>
          {achievable == null ? (
            <span className="tiny muted">{achievableNote(null)}</span>
          ) : (
            <span className="tiny muted">
              Requested {formatRatio(srRatio)} · achievable ~{formatRatio(achievable)}
              {supported != null && ` (${Math.round(supported * 100)}% of request)`} —{' '}
              {achievableNote(supported)}
            </span>
          )}
        </div>

        {/* Motion sufficiency ------------------------------------------------ */}
        <MotionFeedback motion={motion} />

        {/* Actions ----------------------------------------------------------- */}
        <div className="row">
          <Button
            variant="primary"
            onClick={onProcess}
            disabled={!canProcess || busy}
            className="grow"
          >
            {busy ? 'Processing…' : 'Process'}
          </Button>
          <Button variant="danger" onClick={onCancel} disabled={!busy}>
            Cancel
          </Button>
        </div>
      </div>
    </Card>
  )
}
