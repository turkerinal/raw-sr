/**
 * Free-form super-resolution ratio control.
 *
 * The user can type ANY value (e.g. 1.7, 3.3) — the field is not restricted to
 * slider steps — while a slider stays in sync for coarse adjustment. Values are
 * clamped to a sane minimum of 1×. SPEC §10.
 *
 * Owned by AGENT-F.
 */
import { useEffect, useState } from 'react'
import { Slider } from '../../ui'

export interface SrRatioInputProps {
  value: number
  onChange: (v: number) => void
  disabled?: boolean
}

const MIN_RATIO = 1
const SLIDER_STEP = 0.1
/** Slider grows past this when the user types a larger free-form value. */
const SLIDER_BASE_MAX = 5

export function clampRatio(v: number): number {
  if (!Number.isFinite(v)) return MIN_RATIO
  return Math.max(MIN_RATIO, v)
}

function round2(v: number): number {
  return Math.round(v * 100) / 100
}

/** "N×" display — trims trailing zeros (2 -> "2×", 1.7 -> "1.7×"). */
export function formatRatio(v: number): string {
  if (!Number.isFinite(v)) return '—'
  return `${round2(v)}×`
}

export function SrRatioInput({ value, onChange, disabled }: SrRatioInputProps) {
  const [draft, setDraft] = useState<string>(() => String(round2(clampRatio(value))))
  const [focused, setFocused] = useState(false)

  // Keep the text field in sync with external changes (slider, reset, …) while
  // the user is not actively editing it.
  useEffect(() => {
    if (!focused) setDraft(String(round2(clampRatio(value))))
  }, [value, focused])

  const handleText = (raw: string) => {
    setDraft(raw)
    const parsed = Number(raw)
    // Only commit parseable, non-empty input; partial states like "1." are kept
    // in the draft so typing stays smooth.
    if (raw.trim() !== '' && Number.isFinite(parsed)) {
      onChange(clampRatio(parsed))
    }
  }

  const handleBlur = () => {
    setFocused(false)
    const parsed = Number(draft)
    const next =
      draft.trim() !== '' && Number.isFinite(parsed) ? clampRatio(parsed) : clampRatio(value)
    setDraft(String(round2(next)))
    if (next !== value) onChange(next)
  }

  const safeValue = clampRatio(value)
  const sliderMax = Math.max(SLIDER_BASE_MAX, Math.ceil(safeValue))

  return (
    <div className="sr-ratio">
      <div className="sr-ratio__head">
        <span className="field__label">Super-resolution ratio</span>
        <span className="sr-ratio__value mono">
          {formatRatio(safeValue)}
          {safeValue >= 3 ? ' (experimental)' : ''}
        </span>
      </div>

      <div className="sr-ratio__input">
        <input
          className="input sr-ratio__field"
          type="text"
          inputMode="decimal"
          value={draft}
          disabled={disabled}
          aria-label="Super-resolution ratio"
          onFocus={() => setFocused(true)}
          onChange={(e) => handleText(e.target.value)}
          onBlur={handleBlur}
        />
        <span className="sr-ratio__unit">×</span>
      </div>

      <Slider
        label="Fine-tune"
        min={MIN_RATIO}
        max={sliderMax}
        step={SLIDER_STEP}
        value={safeValue}
        onChange={(v) => onChange(clampRatio(v))}
        format={formatRatio}
        disabled={disabled}
      />

      <span className="field__hint">
        Recommended: 2× (default). 3×/4× are experimental and benefit from 20–40 frames. Type any value ≥ 1 or drag the slider.
      </span>
    </div>
  )
}
