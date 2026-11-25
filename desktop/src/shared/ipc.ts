/**
 * Shared IPC contract between the Electron main process, the preload bridge
 * and the renderer. Frozen — see desktop/ARCHITECTURE.md.
 */

export interface DesktopBridge {
  /** TCP port the Python FastAPI backend is listening on. */
  backendPort(): Promise<number>
  /** Native open dialog filtered to RAW files; resolves to chosen paths. */
  openRawFiles(): Promise<string[]>
  /** Native save dialog; resolves to a path or null when cancelled. */
  saveFile(defaultName: string): Promise<string | null>
  /** Reveal a file in the OS file manager. */
  showItemInFolder(path: string): Promise<void>
  isElectron: true
}

declare global {
  interface Window {
    desktop?: DesktopBridge
  }
}

export const IPC = {
  backendPort: 'desktop:backend-port',
  openRawFiles: 'desktop:open-raw-files',
  saveFile: 'desktop:save-file',
  showItemInFolder: 'desktop:show-item-in-folder'
} as const

export const RAW_EXTENSIONS = [
  'dng', 'cr2', 'cr3', 'nef', 'nrw', 'arw', 'srf', 'sr2',
  'raf', 'rw2', 'orf', 'pef', 'raw', 'tif', 'tiff'
] as const
