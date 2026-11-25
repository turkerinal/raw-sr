/**
 * Presentational pieces of the ROI editor: the translucent overlay rectangle
 * (with its 8 resize handles) drawn on top of the preview, and the numeric
 * readout row for one ROI.
 *
 * Owned by AGENT-D.
 */
import type { CSSProperties } from 'react'
import type { Rect } from '../../api/types'
import { Button, NumberField } from '../../ui'
import { ROI_HANDLES, round, toPercent, withRectField } from './geometry'
import type { RoiRectId } from './useDragRect'

const LABEL: Record<RoiRectId, string> = {
  target: 'Target ROI',
  search: 'Search Area'
}

export interface RoiBoxProps {
  rect: Rect
  kind: RoiRectId
  sensorWidth: number
  sensorHeight: number
  /** The rect currently being edited (the panel's active rect). */
  active?: boolean
}

/** A rectangle positioned as a percentage of the rendered preview box. */
export function RoiBox({ rect, kind, sensorWidth, sensorHeight, active = false }: RoiBoxProps) {
  const [x, y, w, h] = rect
  const style: CSSProperties = {
    left: toPercent(x, sensorWidth),
    top: toPercent(y, sensorHeight),
    width: toPercent(w, sensorWidth),
    height: toPercent(h, sensorHeight)
  }
  return (
    <div className={`roi-box roi-box--${kind}${active ? ' is-active' : ''}`} style={style}>
      {ROI_HANDLES.map((handle) => (
        <span key={handle} className={`roi-handle roi-handle--${handle}`} aria-hidden="true" />
      ))}
      <span className="roi-box__label">
        {LABEL[kind]} · {round(w)}×{round(h)}
      </span>
    </div>
  )
}

export interface RoiReadoutProps {
  kind: RoiRectId
  rect: Rect | null
  sensorWidth: number
  sensorHeight: number
  onChange: (rect: Rect) => void
  onClear: () => void
}

/** Editable x/y/w/h readout for a single ROI, with a clear button. */
export function RoiReadout({ kind, rect, sensorWidth, sensorHeight, onChange, onClear }: RoiReadoutProps) {
  const maxAt = (index: number) => (index === 0 || index === 2 ? sensorWidth : sensorHeight)
  return (
    <div className={`roi-readout roi-readout--${kind}`}>
      <div className="roi-readout__head">
        <span className={`roi-swatch roi-swatch--${kind}`} />
        <span className="roi-readout__title">{LABEL[kind]}</span>
        <span className="grow" />
        <Button size="sm" variant="ghost" onClick={onClear} disabled={!rect} title={`Clear ${LABEL[kind]}`}>
          Clear
        </Button>
      </div>
      {rect ? (
        <div className="roi-fields">
          {(['X', 'Y', 'W', 'H'] as const).map((label, index) => (
            <NumberField
              key={label}
              label={label}
              value={round(rect[index])}
              min={0}
              max={maxAt(index)}
              onChange={(value) => onChange(withRectField(rect, index, value, sensorWidth, sensorHeight))}
            />
          ))}
        </div>
      ) : (
        <div className="roi-readout__empty faint">Unspecified — the full frame is used.</div>
      )}
    </div>
  )
}
