import type { ButtonHTMLAttributes, ReactNode } from 'react'

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  title: string
  children: ReactNode
}

export function IconButton({ title, children, className = '', ...rest }: IconButtonProps) {
  return (
    <button className={`icon-btn ${className}`.trim()} title={title} aria-label={title} {...rest}>
      {children}
    </button>
  )
}
