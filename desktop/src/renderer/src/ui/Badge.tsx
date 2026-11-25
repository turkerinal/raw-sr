import type { ReactNode } from 'react'

export interface BadgeProps {
  tone?: 'neutral' | 'warn' | 'ok' | 'bad'
  children: ReactNode
}

export function Badge({ tone = 'neutral', children }: BadgeProps) {
  return <span className={`badge badge--${tone}`}>{children}</span>
}
