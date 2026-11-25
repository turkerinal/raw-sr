import type { ReactNode } from 'react'

export interface CardProps {
  title?: string
  hint?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}

export function Card({ title, hint, actions, children, className = '' }: CardProps) {
  return (
    <section className={`card ${className}`.trim()}>
      {(title || actions) && (
        <header className="card__head">
          {title && (
            <h2 className="card__title">
              {title}
              {hint && <span className="hint">{hint}</span>}
            </h2>
          )}
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}
