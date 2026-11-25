import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

type Tone = 'info' | 'error' | 'ok'
interface ToastItem {
  id: number
  message: string
  tone: Tone
}

interface ToastApi {
  toast: (message: string, tone?: Tone) => void
}

const ToastContext = createContext<ToastApi>({ toast: () => {} })

export function useToast(): ToastApi {
  return useContext(ToastContext)
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])

  const toast = useCallback((message: string, tone: Tone = 'info') => {
    const id = Date.now() + Math.random()
    setItems((prev) => [...prev, { id, message, tone }])
    window.setTimeout(() => {
      setItems((prev) => prev.filter((t) => t.id !== id))
    }, tone === 'error' ? 6000 : 3500)
  }, [])

  const api = useMemo(() => ({ toast }), [toast])

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className="toast-host">
        {items.map((t) => (
          <div key={t.id} className={`toast toast--${t.tone}`}>
            {t.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
