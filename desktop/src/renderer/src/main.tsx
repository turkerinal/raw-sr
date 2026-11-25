import React from 'react'
import { createRoot } from 'react-dom/client'
import './styles/global.css'
import { App } from './App'
import { ToastProvider } from './ui'

const root = document.getElementById('root')
if (!root) throw new Error('#root not found')

createRoot(root).render(
  <React.StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </React.StrictMode>
)
