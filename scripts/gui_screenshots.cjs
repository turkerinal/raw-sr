/**
 * Real full-screen (1920x1080) screenshots of the Electron desktop app.
 *
 * Nothing is mocked: this file is used as the Electron *entry point* instead of
 * desktop/out/main/index.js. It (1) answers the native "open RAW files" dialog
 * with the burst folder given in RAW_SR_SHOT_DIR, (2) imports the real main
 * process, (3) puts the window in full screen, drives the real UI (clicks the
 * real buttons, waits for the real backend) and (4) saves
 * webContents.capturePage() images.
 *
 * Usage (from the repository root, after `npx electron-vite build` in desktop/):
 *   RAW_SR_SHOT_DIR=/tmp/raw-sr-demo/hdrplus_0919 RAW_SR_SHOT_OUT=/tmp/shots \
 *     desktop/node_modules/.bin/electron scripts/gui_screenshots.cjs
 *
 * Optional: RAW_SR_PYTHON (backend interpreter), RAW_SR_SHOT_ROI="x,y,w,h".
 */
const path = require('node:path')
const fs = require('node:fs')
const { app, BrowserWindow, dialog } = require('electron')

const SHOT_DIR = process.env.RAW_SR_SHOT_DIR
const OUT_DIR = process.env.RAW_SR_SHOT_OUT || path.join(process.cwd(), 'gui_shots')
if (!SHOT_DIR) throw new Error('RAW_SR_SHOT_DIR is required')
fs.mkdirSync(OUT_DIR, { recursive: true })

const FRAMES = fs
  .readdirSync(SHOT_DIR)
  .filter((n) => /\.(dng|cr2|nef|arw)$/i.test(n))
  .sort()
  .map((n) => path.join(SHOT_DIR, n))

// Replace only the native file dialog (it cannot be clicked programmatically).
dialog.showOpenDialog = async () => ({ canceled: false, filePaths: FRAMES })
// Never leave a native "save" dialog open.
dialog.showSaveDialog = async () => ({ canceled: true, filePath: undefined })

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function main() {
  await app.whenReady()
  // Import the real built main process (ESM) from the CommonJS entry.
  await import(path.resolve(__dirname, '../desktop/out/main/index.js'))

  let win
  for (let i = 0; i < 200 && !win; i++) {
    win = BrowserWindow.getAllWindows()[0]
    await sleep(100)
  }
  const wc = win.webContents
  const js = (code) => wc.executeJavaScript(code, true)
  const waitFor = async (code, label, timeoutMs = 120000) => {
    const t0 = Date.now()
    while (Date.now() - t0 < timeoutMs) {
      try {
        if (await js(code)) return
      } catch {
        /* page still loading */
      }
      await sleep(400)
    }
    throw new Error('timeout waiting for ' + label)
  }
  const clickText = (text, exact = true) =>
    js(`(() => {
      const b = [...document.querySelectorAll('button')].find(
        (x) => ${exact ? 'x.textContent.trim() === ' + JSON.stringify(text) : 'x.textContent.includes(' + JSON.stringify(text) + ')'} && !x.disabled);
      if (!b) return false; b.click(); return true })()`)
  const shot = async (name) => {
    await sleep(900)
    const img = await wc.capturePage()
    const size = img.getSize()
    fs.writeFileSync(path.join(OUT_DIR, name + '.png'), img.toPNG())
    console.log('[shot]', name, size.width + 'x' + size.height)
  }

  win.setFullScreen(true)
  await sleep(1500)
  console.log('[win] content size', JSON.stringify(win.getContentSize()))

  await waitFor(`document.body.innerText.includes('Engine ready')`, 'engine ready')
  await shot('00_empty')

  await clickText('Add RAW Files…')
  await waitFor(`document.querySelectorAll('[role="list"] [role="button"]').length >= ${FRAMES.length}`, 'frames listed')
  await sleep(2500)

  if (process.env.RAW_SR_SHOT_DUMP) {
    console.log(
      await js(`JSON.stringify({
        buttons: [...document.querySelectorAll('button')].map(b => b.textContent.trim()),
        inputs: [...document.querySelectorAll('input')].map(i => [i.type, i.getAttribute('aria-label'), i.value])
      })`)
    )
  }
  await shot('01_loaded')

  if (process.env.RAW_SR_SHOT_STOP === 'loaded') return
  // Settings / ROI: switch to the real "Target" tool and draw a region with
  // real (synthetic-input) mouse events on the frame preview. This also keeps
  // the processing region small.
  await clickText('Target')
  await sleep(500)
  const [x0, y0, x1, y1] = (process.env.RAW_SR_SHOT_DRAG || '960,560,1320,800').split(',').map(Number)
  const mouse = (type, x, y, extra = {}) =>
    wc.sendInputEvent({ type, x, y, button: 'left', clickCount: 1, ...extra })
  mouse('mouseMove', x0, y0)
  mouse('mouseDown', x0, y0)
  for (let i = 1; i <= 12; i++) {
    mouse('mouseMove', x0 + ((x1 - x0) * i) / 12, y0 + ((y1 - y0) * i) / 12, { modifiers: ['leftButtonDown'] })
    await sleep(30)
  }
  mouse('mouseUp', x1, y1)
  await sleep(800)
  // Scroll the left column so the ratio / motion / Process controls are visible.
  await js(`(() => { const p = [...document.querySelectorAll('*')].filter(e => e.scrollHeight > e.clientHeight + 50 && getComputedStyle(e).overflowY !== 'visible' && e.getBoundingClientRect().left < 100); p.forEach(e => e.scrollTop = 400); })()`)
  await shot('02_settings')

  await clickText('Process')
  await waitFor(`document.body.innerText.includes('Before (reference)')`, 'result', 2400000)
  await sleep(4000)
  // The result panel sits below the ROI view: scroll it to the top of the screen.
  await js(`(() => { const h = [...document.querySelectorAll('h4')].find(e => e.textContent.includes('Before (reference)'));
    let el = h; while (el && !el.classList.contains('result')) el = el.parentElement;
    (el || h).scrollIntoView({ block: 'start' }); })()`)
  await sleep(1500)
  await shot('03_result')
  // Metrics / export buttons live just above the comparison panes.
  await js(`(() => { const b = [...document.querySelectorAll('button')].find(x => x.textContent.includes('Export')); if (b) b.scrollIntoView({ block: 'center' }) })()`)
  await shot('04_export')
}

main()
  .then(() => app.exit(0))
  .catch((e) => {
    console.error('[error]', e)
    app.exit(1)
  })
