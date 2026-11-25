/**
 * AlignmentView — replaces the removed confidence layers (SPEC §10).
 *
 * Answers "did the frames align, and how well?":
 *   • prominent overall `score` + `level` badge + `message` verdict
 *   • `mean_zncc` / `min_zncc`
 *   • a per-frame list with ZNCC bars, residual (px) and shift, styling kept vs
 *     dropped frames
 *
 * The per-pixel alignment-quality heatmap moved to its own "Quality map"
 * section at the bottom of the result panel (ResultPanel).
 *
 * Owned by AGENT-E.
 */
import type { Alignment } from '../../api/types'
import { Badge } from '../../ui'

export interface AlignmentViewProps {
  alignment: Alignment
}

const TONE: Record<Alignment['level'], 'ok' | 'warn' | 'bad'> = {
  ok: 'ok',
  warn: 'warn',
  bad: 'bad'
}

function num(v: number, digits = 3): string {
  return Number.isFinite(v) ? v.toFixed(digits) : '—'
}

function pct(v: number): string {
  return Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—'
}

function shiftText(shift: [number, number]): string {
  const [sx, sy] = shift
  if (!Number.isFinite(sx) || !Number.isFinite(sy)) return '—'
  return `${sx.toFixed(1)}, ${sy.toFixed(1)} px`
}

export function AlignmentView({ alignment }: AlignmentViewProps) {
  const { score, level, message, mean_zncc, min_zncc, per_frame } = alignment
  const frames = Array.isArray(per_frame) ? per_frame : []

  return (
    <div className="stack align">
      <div className="row row--between">
        <h3 className="result__section-title">Alignment</h3>
        <Badge tone={TONE[level]}>{level.toUpperCase()}</Badge>
      </div>

      <div className={`align__summary align__summary--${level}`}>
        <div className="align__score-block">
          <div className="align__score">{pct(score)}</div>
          <div className="align__score-label">alignment score</div>
        </div>
        <div className="align__verdict">
          <div className="align__message">{message || '—'}</div>
          <div className="align__subs">
            <span>
              mean ZNCC <strong className="mono">{num(mean_zncc)}</strong>
            </span>
            <span>
              min ZNCC <strong className="mono">{num(min_zncc)}</strong>
            </span>
          </div>
        </div>
      </div>

      <div className="align__frames">
        <div className="row row--between">
          <h4 className="align__frames-title">Per-frame</h4>
          <span className="small faint">
            {frames.filter((f) => f.used).length}/{frames.length} kept
          </span>
        </div>

        {frames.length === 0 ? (
          <div className="result__notice">
            <span aria-hidden="true">ℹ</span>
            <span>No per-frame alignment data was reported.</span>
          </div>
        ) : (
          <ul className="align__list">
            {frames.map((f, i) => {
              const zncc = Number.isFinite(f.zncc) ? Math.min(1, Math.max(0, f.zncc)) : 0
              const barTone = zncc >= 0.6 ? 'ok' : zncc >= 0.35 ? 'warn' : 'bad'
              return (
                <li
                  key={`${f.id}-${i}`}
                  className={`align__row${f.used ? '' : ' align__row--dropped'}`}
                >
                  <span className="align__id" title={`Frame ${f.id}`}>
                    #{f.id}
                  </span>
                  <span className="align__bar" aria-hidden="true">
                    <span
                      className={`align__bar-fill align__bar-fill--${barTone}`}
                      style={{ width: `${zncc * 100}%` }}
                    />
                  </span>
                  <span className="align__zncc mono">{num(f.zncc)}</span>
                  <span className="align__meta mono" title="alignment residual">
                    ±{num(f.residual, 2)} px
                  </span>
                  <span className="align__meta mono" title="shift (x, y)">
                    {shiftText(f.shift)}
                  </span>
                  <span className={`align__flag${f.used ? ' align__flag--used' : ''}`}>
                    {f.used ? 'kept' : 'dropped'}
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </div>
  )
}
