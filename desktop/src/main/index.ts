/**
 * Electron main process: boots the Python FastAPI backend on a free port,
 * waits until it is healthy, opens the window and exposes native dialogs over
 * IPC.
 *
 * Owned by AGENT-B.
 */
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  Menu,
  shell,
  type MenuItemConstructorOptions
} from 'electron'
import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import { createServer } from 'node:net'
import { accessSync, existsSync, mkdirSync, readdirSync, rmSync, constants as fsConstants } from 'node:fs'
import { homedir } from 'node:os'
import { get as httpGet } from 'node:http'
import { join, resolve, extname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { IPC, RAW_EXTENSIONS } from '../shared/ipc'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

// Repo root while developing: out/main/index.js -> desktop/out/main -> repo.
const DEV_REPO_ROOT = resolve(__dirname, '../../..')

// Where `desktop.backend.server` and `raw_sr` live. In a packaged build
// `extraResources` copies `raw_sr`, `desktop/backend` and `requirements.txt`
// next to the app; electron exposes that directory as `process.resourcesPath`.
const BACKEND_BASE = app.isPackaged ? process.resourcesPath : DEV_REPO_ROOT

// Self-contained builds (AppImage) ship their own Python under
// resources/python; it is preferred over anything on the system.
const BUNDLED_PYTHON = join(process.resourcesPath ?? '', 'python', 'bin', 'python3')
const HAS_BUNDLED_PYTHON = app.isPackaged && existsSync(BUNDLED_PYTHON)

// Optional GPU add-on (CuPy + CUDA libraries, ~2 GB) installed per user by
// `--install-gpu`, so the AppImage itself stays small. Added to PYTHONPATH
// only when present; without it the core falls back to demosaic_drizzle.
const GPU_SITE = join(
  process.env.XDG_DATA_HOME?.trim() || join(homedir(), '.local', 'share'),
  'raw-sr',
  'gpu-site'
)

// How long to wait for `GET /api/health` before giving up (imports of cv2 et al.
// can take a while on a cold start).
const configuredTimeout = Number(process.env.RAW_SR_BACKEND_TIMEOUT_MS ?? 90_000)
const HEALTH_TIMEOUT_MS =
  Number.isFinite(configuredTimeout) && configuredTimeout > 0 ? configuredTimeout : 90_000
const HEALTH_POLL_MS = 300

let backend: ChildProcess | null = null
let backendPort = 0
let backendReady = false
let backendExitedEarly = false
let backendLog = ''

// --------------------------------------------------------------------------- //
// Small helpers
// --------------------------------------------------------------------------- //

function delay(ms: number): Promise<void> {
  return new Promise((res) => setTimeout(res, ms))
}

function appendBackendLog(chunk: Buffer): void {
  backendLog = (backendLog + chunk.toString()).slice(-4000)
}

function looksLikePath(value: string): boolean {
  return value.includes('/') || value.includes('\\')
}

function isExecutable(path: string): boolean {
  try {
    accessSync(path, fsConstants.X_OK)
    return true
  } catch {
    return false
  }
}

function commandExists(command: string): boolean {
  const probe = process.platform === 'win32' ? 'where' : 'which'
  const result = spawnSync(probe, [command], { stdio: 'ignore' })
  return result.status === 0
}

// --------------------------------------------------------------------------- //
// Python + backend lifecycle
// --------------------------------------------------------------------------- //

/**
 * Resolve a usable Python interpreter:
 *   1. RAW_SR_PYTHON (authoritative when set),
 *   2. the repo-root virtualenv,
 *   3. `python3` / `python` on PATH.
 * Throws with an actionable message when nothing usable is found.
 */
function resolvePython(): string {
  const problems: string[] = []

  const explicit = process.env.RAW_SR_PYTHON?.trim()
  if (explicit) {
    const usable = looksLikePath(explicit) ? isExecutable(explicit) : commandExists(explicit)
    if (usable) return explicit
    problems.push(
      `RAW_SR_PYTHON is set to "${explicit}" but it is not an executable/usable Python interpreter.`
    )
  }

  if (HAS_BUNDLED_PYTHON) return BUNDLED_PYTHON

  const venvCandidates = [
    join(DEV_REPO_ROOT, 'venv', 'bin', 'python'),
    join(DEV_REPO_ROOT, '.venv', 'bin', 'python'),
    join(DEV_REPO_ROOT, 'venv', 'Scripts', 'python.exe')
  ]
  for (const candidate of venvCandidates) {
    if (isExecutable(candidate)) return candidate
  }

  for (const candidate of ['python3', 'python']) {
    if (commandExists(candidate)) return candidate
  }

  const extra = problems.length ? `\n${problems.join('\n')}` : ''
  throw new Error(
    'Could not find a usable Python interpreter.\n' +
      'Set RAW_SR_PYTHON=/path/to/python, create a virtualenv at ' +
      `${join(DEV_REPO_ROOT, 'venv')}, or put "python3" on PATH.${extra}`
  )
}

function findFreePort(): Promise<number> {
  return new Promise((res, rej) => {
    const srv = createServer()
    srv.once('error', rej)
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : 0
      srv.close(() => res(port))
    })
  })
}

async function startBackend(): Promise<number> {
  const python = resolvePython()
  const port = await findFreePort()
  backendPort = port

  const child = spawn(python, ['-m', 'desktop.backend.server', '--port', String(port)], {
    cwd: BACKEND_BASE,
    env: {
      ...process.env,
      PYTHONPATH: existsSync(GPU_SITE) ? `${BACKEND_BASE}:${GPU_SITE}` : BACKEND_BASE,
      // The bundled interpreter must not pick up the user's ~/.local packages.
      ...(HAS_BUNDLED_PYTHON ? { PYTHONNOUSERSITE: '1' } : {}),
      PYTHONUNBUFFERED: '1',
      PYTHONDONTWRITEBYTECODE: '1'
    },
    stdio: ['ignore', 'pipe', 'pipe']
  })

  backend = child
  child.stdout?.on('data', (d: Buffer) => {
    appendBackendLog(d)
    process.stdout.write(`[backend] ${d}`)
  })
  child.stderr?.on('data', (d: Buffer) => {
    appendBackendLog(d)
    process.stderr.write(`[backend] ${d}`)
  })
  child.on('error', (err) => {
    appendBackendLog(Buffer.from(String(err)))
    backendExitedEarly = true
  })
  child.on('exit', (code, signal) => {
    console.log(`[backend] exited code=${code} signal=${signal}`)
    backend = null
    if (!backendReady) backendExitedEarly = true
  })

  return port
}

function probeHealth(port: number): Promise<boolean> {
  return new Promise((resolveProbe) => {
    const req = httpGet(
      { host: '127.0.0.1', port, path: '/api/health', timeout: 1500 },
      (res) => {
        res.resume()
        resolveProbe(res.statusCode === 200)
      }
    )
    req.on('timeout', () => {
      req.destroy()
      resolveProbe(false)
    })
    req.on('error', () => resolveProbe(false))
  })
}

async function waitForBackend(port: number): Promise<boolean> {
  const deadline = Date.now() + HEALTH_TIMEOUT_MS
  while (Date.now() < deadline) {
    if (backendExitedEarly) return false
    if (await probeHealth(port)) return true
    await delay(HEALTH_POLL_MS)
  }
  return false
}

function stopBackend(): void {
  const child = backend
  backend = null
  if (!child || child.killed) return
  try {
    if (process.platform === 'win32' && child.pid) {
      spawnSync('taskkill', ['/pid', String(child.pid), '/f', '/t'], { stdio: 'ignore' })
    } else {
      child.kill('SIGTERM')
      // Escalate if it ignores the polite request.
      setTimeout(() => {
        if (!child.killed) child.kill('SIGKILL')
      }, 3000).unref()
    }
  } catch (err) {
    console.error('[backend] failed to stop cleanly:', err)
  }
}

// --------------------------------------------------------------------------- //
// Window / UI
// --------------------------------------------------------------------------- //

function escapeHtml(value: string): string {
  return value.replace(/[&<>"]/g, (c) => {
    switch (c) {
      case '&':
        return '&amp;'
      case '<':
        return '&lt;'
      case '>':
        return '&gt;'
      default:
        return '&quot;'
    }
  })
}

function errorPage(message: string): string {
  const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>raw_sr — backend failed to start</title>
    <style>
      :root { color-scheme: dark; }
      body {
        margin: 0; min-height: 100vh; display: grid; place-items: center;
        font: 14px/1.6 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        background: #0e1116; color: #e6edf3;
      }
      main { max-width: 720px; padding: 32px; }
      h1 { margin: 0 0 4px; font-size: 20px; }
      p { color: #9aa4b2; margin: 0 0 20px; }
      pre {
        white-space: pre-wrap; word-break: break-word;
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; color: #ffb4a2; font-size: 12.5px;
      }
      code { color: #79c0ff; }
    </style>
  </head>
  <body>
    <main>
      <h1>raw_sr could not start its Python backend</h1>
      <p>The super-resolution engine did not become ready. Fix the issue below and restart the app.</p>
      <pre>${escapeHtml(message)}</pre>
      <p>Tips: run <code>desktop/run.sh</code>, or set <code>RAW_SR_PYTHON</code> to a Python that has the backend requirements installed.</p>
    </main>
  </body>
</html>`
  return 'data:text/html;charset=utf-8,' + encodeURIComponent(html)
}

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    backgroundColor: '#0e1116',
    title: 'raw_sr — RAW Multi-Frame Super-Resolution',
    show: false,
    webPreferences: {
      preload: join(__dirname, '../preload/index.mjs'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  win.once('ready-to-show', () => win.show())
  // Safety net: never leave an invisible, seemingly-blank window behind.
  setTimeout(() => {
    if (!win.isDestroyed() && !win.isVisible()) win.show()
  }, 5000).unref()

  // Open external links in the OS browser, never inside the app window.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (event, url) => {
    const devUrl = process.env['ELECTRON_RENDERER_URL']
    const isDevServer = Boolean(devUrl) && url.startsWith(devUrl as string)
    if (!isDevServer && /^https?:/i.test(url)) {
      event.preventDefault()
      void shell.openExternal(url)
    }
  })

  return win
}

function loadRenderer(win: BrowserWindow): void {
  const devUrl = process.env['ELECTRON_RENDERER_URL']
  const loading = devUrl
    ? win.loadURL(devUrl)
    : win.loadFile(join(__dirname, '../renderer/index.html'))
  loading.catch((err) => console.error('[main] failed to load renderer:', err))
}

function buildMenu(): void {
  const template: MenuItemConstructorOptions[] = []
  if (process.platform === 'darwin') template.push({ role: 'appMenu' })
  template.push(
    { role: 'fileMenu' },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About raw_sr',
          click: () => {
            const win = BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows()[0]
            const detail =
              `Electron ${process.versions.electron}\n` +
              `Chromium ${process.versions.chrome}\n` +
              `Node ${process.versions.node}\n` +
              `Backend port: ${backendPort || 'not started'}`
            void dialog
              .showMessageBox(win, {
                type: 'info',
                title: 'About raw_sr',
                message: 'raw_sr — RAW Multi-Frame Super-Resolution',
                detail
              })
              .catch(() => undefined)
          }
        }
      ]
    }
  )
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

// --------------------------------------------------------------------------- //
// IPC
// --------------------------------------------------------------------------- //

function registerIpc(): void {
  ipcMain.handle(IPC.backendPort, () => backendPort)

  ipcMain.handle(IPC.openRawFiles, async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select RAW burst',
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'RAW images', extensions: [...RAW_EXTENSIONS] }]
    })
    return result.canceled ? [] : result.filePaths
  })

  ipcMain.handle(IPC.saveFile, async (_e, defaultName: string) => {
    const result = await dialog.showSaveDialog({
      title: 'Export',
      defaultPath: defaultName,
      filters: [{ name: 'TIFF image', extensions: ['tiff', 'tif'] }]
    })
    if (result.canceled || !result.filePath) return null
    const ext = extname(result.filePath).toLowerCase()
    return ext ? result.filePath : `${result.filePath}.tiff`
  })

  ipcMain.handle(IPC.showItemInFolder, (_e, path: string) => {
    shell.showItemInFolder(path)
  })
}

// --------------------------------------------------------------------------- //
// Boot
// --------------------------------------------------------------------------- //

async function boot(): Promise<void> {
  if (process.platform === 'win32') app.setAppUserModelId('org.rawsr.desktop')

  registerIpc()
  buildMenu()

  let startupError: string | null = null
  try {
    const port = await startBackend()
    const healthy = await waitForBackend(port)
    if (healthy) {
      backendReady = true
    } else {
      startupError = backendExitedEarly
        ? 'The Python backend process stopped before it became ready.'
        : `The Python backend did not respond on 127.0.0.1:${port} within ${Math.round(
            HEALTH_TIMEOUT_MS / 1000
          )}s.`
    }
  } catch (err) {
    startupError = (err as Error).message
  }

  const win = createWindow()

  if (startupError) {
    const details = [startupError, backendLog.trim()].filter(Boolean).join('\n\n')
    console.error('[main] backend startup failed:\n' + details)
    dialog.showErrorBox('raw_sr — backend failed to start', details || 'Unknown error')
    void win.loadURL(errorPage(details || 'Unknown error'))
  } else {
    loadRenderer(win)
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      const next = createWindow()
      loadRenderer(next)
    }
  })
}

/**
 * `raw-sr.AppImage --install-gpu`: install CuPy for the local CUDA driver
 * into GPU_SITE with the bundled Python, print progress to the terminal and
 * exit. Returns the process exit code.
 */
function installGpuAddon(): number {
  if (!HAS_BUNDLED_PYTHON) {
    console.error('--install-gpu is only available in the self-contained build (AppImage).')
    console.error('From a source checkout run: venv/bin/pip install "cupy-cuda13x[ctk]" (or cupy-cuda12x[ctk])')
    return 2
  }
  const smi = spawnSync('nvidia-smi', [], { encoding: 'utf8' })
  // Older drivers print "CUDA Version: 12.4", newer ones "CUDA UMD Version: 13.5".
  const m = /CUDA[^:\n]*Version:\s*(\d+)/.exec(smi.stdout ?? '')
  if (smi.status !== 0 || !m) {
    console.error('No NVIDIA driver found (nvidia-smi failed). The GPU add-on needs an NVIDIA GPU;')
    console.error('without it raw-sr still works with the classical CPU path.')
    return 3
  }
  const cudaMajor = Number(m[1])
  const pkg = cudaMajor >= 13 ? 'cupy-cuda13x[ctk]' : 'cupy-cuda12x[ctk]'
  console.log(`NVIDIA driver supports CUDA ${cudaMajor}; installing ${pkg} into ${GPU_SITE}`)
  console.log('This downloads about 1-2 GB once.')
  mkdirSync(GPU_SITE, { recursive: true })
  // pip unpacks ~2 GB of wheels; /tmp is often a small tmpfs, so stage them
  // next to the add-on and remove the staging directory afterwards.
  const stage = join(GPU_SITE, '..', 'gpu-install-tmp')
  mkdirSync(stage, { recursive: true })
  const pip = spawnSync(
    BUNDLED_PYTHON,
    ['-m', 'pip', 'install', '--upgrade', '--target', GPU_SITE, pkg],
    { stdio: 'inherit', env: { ...process.env, PYTHONNOUSERSITE: '1', TMPDIR: stage } }
  )
  rmSync(stage, { recursive: true, force: true })
  if (pip.status !== 0) {
    console.error('Installing the GPU add-on failed (see the pip output above).')
    return pip.status ?? 1
  }
  // pip --target pulls its own numpy as a CuPy dependency; it would shadow the
  // bundled numpy that OpenCV/SciPy were built against, so drop it.
  for (const name of readdirSync(GPU_SITE)) {
    if (name === 'numpy' || name === 'numpy.libs' || /^numpy-.*\.dist-info$/.test(name)) {
      rmSync(join(GPU_SITE, name), { recursive: true, force: true })
    }
  }
  const check = spawnSync(
    BUNDLED_PYTHON,
    ['-c', 'import cupy; cupy.zeros(1).sum(); print("CuPy", cupy.__version__, "OK on", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())'],
    { stdio: 'inherit', env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: GPU_SITE } }
  )
  if (check.status !== 0) {
    console.error('CuPy was installed but could not use the GPU (see above).')
    return check.status ?? 1
  }
  console.log('GPU add-on ready. Start raw-sr normally; 2x runs will use the MAP solver.')
  return 0
}

const INSTALL_GPU_MODE = process.argv.includes('--install-gpu')
if (INSTALL_GPU_MODE) {
  app.exit(installGpuAddon())
}

// Single-instance lock: focus the existing window instead of starting a
// second backend.
const gotLock = !INSTALL_GPU_MODE && app.requestSingleInstanceLock()
if (!gotLock) {
  if (!INSTALL_GPU_MODE) app.quit()
} else {
  app.on('second-instance', () => {
    const win = BrowserWindow.getAllWindows()[0]
    if (win) {
      if (win.isMinimized()) win.restore()
      win.focus()
    }
  })

  app
    .whenReady()
    .then(boot)
    .catch((err) => {
      console.error('[main] fatal startup error:', err)
      dialog.showErrorBox('raw_sr — fatal error', String(err))
      app.quit()
    })
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  stopBackend()
})

app.on('will-quit', () => {
  stopBackend()
})

process.on('unhandledRejection', (reason) => {
  console.error('[main] unhandled promise rejection:', reason)
})
process.on('uncaughtException', (err) => {
  console.error('[main] uncaught exception:', err)
})
