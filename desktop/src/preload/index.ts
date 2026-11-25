/**
 * Preload: exposes a minimal, typed bridge to the renderer over contextBridge.
 * Owned by AGENT-B. Baseline implementation by the orchestrator.
 */
import { contextBridge, ipcRenderer } from 'electron'
import { IPC, type DesktopBridge } from '../shared/ipc'

const bridge: DesktopBridge = {
  backendPort: () => ipcRenderer.invoke(IPC.backendPort),
  openRawFiles: () => ipcRenderer.invoke(IPC.openRawFiles),
  saveFile: (defaultName: string) => ipcRenderer.invoke(IPC.saveFile, defaultName),
  showItemInFolder: (path: string) => ipcRenderer.invoke(IPC.showItemInFolder, path),
  isElectron: true
}

contextBridge.exposeInMainWorld('desktop', bridge)
