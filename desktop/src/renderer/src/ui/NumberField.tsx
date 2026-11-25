export interface NumberFieldProps {
  label: string
  value: number
  min?: number
  max?: number
  step?: number
  onChange: (value: number) => void
  disabled?: boolean
}

export function NumberField({ label, value, min, max, step = 1, onChange, disabled }: NumberFieldProps) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      <span className="numberfield">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </span>
    </label>
  )
}
