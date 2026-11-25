/**
 * Motion sufficiency feedback: the engine's verdict plus average/max shift
 * between frames. SPEC §10.
 *
 * Owned by AGENT-F.
 */
import type { MotionInfo } from '../../api/types'
import { Badge } from '../../ui'

export interface MotionFeedbackProps {
  motion: MotionInfo | null
}

export function MotionFeedback({ motion }: MotionFeedbackProps) {
  if (!motion) {
    return (
      <div className="sr-feedback">
        <div className="sr-feedback__head">
          <span className="field__label">Motion sufficiency</span>
          <Badge tone="neutral">no data</Badge>
        </div>
        <span className="tiny muted">Add frames to analyze sub-pixel shift between exposures.</span>
      </div>
    )
  }

  return (
    <div className="sr-feedback">
      <div className="sr-feedback__head">
        <span className="field__label">Motion sufficiency</span>
        <Badge tone={motion.sufficient ? 'ok' : 'warn'}>
          {motion.sufficient ? 'sufficient' : 'insufficient'}
        </Badge>
      </div>
      <span className="tiny muted">{motion.message}</span>
      <div className="sr-feedback__stats">
        <span className="mono tiny">avg shift {motion.avg_shift.toFixed(2)} px</span>
        <span className="mono tiny">max shift {motion.max_shift.toFixed(2)} px</span>
      </div>
    </div>
  )
}
