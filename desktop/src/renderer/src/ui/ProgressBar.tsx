export interface ProgressBarProps {
  value: number
  label?: string
  indeterminate?: boolean
}

export function ProgressBar({ value, label, indeterminate }: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, value))
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="progress">
        <div
          className={`progress__fill ${indeterminate ? 'progress__fill--indeterminate' : ''}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
      {(label || !indeterminate) && (
        <div className="row row--between tiny muted">
          <span>{label ?? ''}</span>
          {!indeterminate && <span className="mono">{clamped}%</span>}
        </div>
      )}
    </div>
  )
}
