export interface SliderProps {
  label: string
  min: number
  max: number
  step: number
  value: number
  onChange: (value: number) => void
  format?: (value: number) => string
  disabled?: boolean
}

export function Slider({ label, min, max, step, value, onChange, format, disabled }: SliderProps) {
  return (
    <div className="slider">
      <div className="slider__top">
        <span className="slider__label">{label}</span>
        <span className="slider__value">{format ? format(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}
