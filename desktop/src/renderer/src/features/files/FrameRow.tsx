/**
 * Single frame row: file name, optional resolution, compact blur/skip markers,
 * the include/skip override and remove controls, and — once a result exists —
 * the kept/dropped status with its ZNCC under the name. Names only, no images.
 *
 * The row itself is clickable and opens the frame in the ROI panel via
 * `onSelect`; the skip/remove controls swallow the click so acting on them does
 * not also change the selection. Presentational only — state comes from props.
 *
 * Owned by AGENT-C.
 */
import type { KeyboardEvent, MouseEvent } from 'react'
import type { FrameInfo } from '../../api/types'
import { Button, IconButton } from '../../ui'

export interface FrameRowProps {
  frame: FrameInfo
  /** True when the user has skipped this frame (id present in `excluded`). */
  excluded: boolean
  /** True when this frame is the one shown in the ROI panel. */
  selected: boolean
  /** Per-frame result once processing finished; null/undefined before that. */
  result?: { used: boolean; zncc: number } | null
  onRemove: (id: string) => void
  onToggleBlur: (id: string) => void
  /** Open this frame in the ROI panel. */
  onSelect?: (id: string) => void
}

export function FrameRow({
  frame,
  excluded,
  selected,
  result,
  onRemove,
  onToggleBlur,
  onSelect
}: FrameRowProps) {
  const select = () => onSelect?.(frame.id)

  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      select()
    }
  }

  function handleToggle(e: MouseEvent) {
    e.stopPropagation()
    onToggleBlur(frame.id)
  }

  function handleRemove(e: MouseEvent) {
    e.stopPropagation()
    onRemove(frame.id)
  }

  const classes = ['frame-row', selected ? 'is-selected' : '', excluded ? 'is-excluded' : '']
    .filter(Boolean)
    .join(' ')

  const statusText = result ? (result.used ? `kept · ZNCC ${result.zncc.toFixed(2)}` : 'dropped') : '—'
  const statusClass = result ? (result.used ? 'is-kept' : 'is-dropped') : 'is-pending'

  return (
    <div
      className={classes}
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      title={frame.path}
      onClick={select}
      onKeyDown={handleKeyDown}
    >
      <div className="frame-row__cell">
        <div className="frame-row__line">
          <span className="frame-row__name">{frame.name}</span>
          <span className="frame-row__dims mono">
            {frame.width}×{frame.height}
          </span>
          {frame.blurry && (
            <span className="frame-row__blurry" title="Auto-flagged blurry (low sharpness)">
              ⚠ blurry
            </span>
          )}
          {excluded && <span className="frame-row__skipped">skipped</span>}
        </div>

        <div className={`frame-row__status ${statusClass}`}>{statusText}</div>
      </div>

      <div className="frame-row__actions">
        {frame.blurry && (
          <Button
            size="sm"
            variant={excluded ? 'secondary' : 'ghost'}
            className="frame-row__btn"
            title={
              excluded
                ? 'Override the flag and include this frame again'
                : 'Override the flag and skip this frame'
            }
            onClick={handleToggle}
          >
            {excluded ? '↺ Include' : 'Skip'}
          </Button>
        )}

        <IconButton title={`Remove ${frame.name}`} onClick={handleRemove}>
          ×
        </IconButton>
      </div>
    </div>
  )
}
