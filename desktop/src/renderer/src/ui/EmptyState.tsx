import type { ReactNode } from 'react'

export interface EmptyStateProps {
  title: string
  hint?: string
  icon?: ReactNode
}

export function EmptyState({ title, hint, icon }: EmptyStateProps) {
  return (
    <div className="empty">
      {icon && <div className="empty__icon">{icon}</div>}
      <div className="empty__title">{title}</div>
      {hint && <div className="empty__hint">{hint}</div>}
    </div>
  )
}
