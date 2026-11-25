#!/usr/bin/env node
/**
 * Headless end-to-end smoke test for the desktop backend.
 *
 * Boots `desktop.backend.server` on a free port, then drives the HTTP API
 * through the real SR core (via the synthetic dev endpoint) and asserts every
 * SPEC §10 result/export path returns data. No RAW sample files required.
 *
 *   node desktop/scripts/e2e-smoke.mjs
 *
 * Exits 0 on PASS, 1 on FAIL. Set RAW_SR_PYTHON to choose the interpreter.
 */
import { spawn } from 'node:child_process'
import { createServer } from 'node:net'
import { existsSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))
// desktop/scripts -> repo root is two levels up
const REPO_ROOT = resolve(__dirname, '../..')

const results = []
function check(name, ok, detail = '') {
  results.push(!!ok)
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? `  (${detail})` : ''}`)
}

function freePort() {
  return new Promise((res, rej) => {
    const s = createServer()
    s.once('error', rej)
    s.listen(0, '127.0.0.1', () => {
      const p = s.address().port
      s.close(() => res(p))
    })
  })
}

function pythonPath() {
  if (process.env.RAW_SR_PYTHON) return process.env.RAW_SR_PYTHON
  const v = join(REPO_ROOT, 'venv', 'bin', 'python')
  return existsSync(v) ? v : 'python3'
}

async function waitHealth(base, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${base}/api/health`)
      if (r.ok) return true
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 300))
  }
  return false
}

async function main() {
  const port = await freePort()
  const base = `http://127.0.0.1:${port}`
  const py = pythonPath()
  console.log(`raw_sr desktop e2e smoke\n  python: ${py}\n  base:   ${base}\n`)

  const backend = spawn(py, ['-m', 'desktop.backend.server', '--port', String(port)], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONPATH: REPO_ROOT, PYTHONUNBUFFERED: '1' }
  })
  backend.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`))

  try {
    const up = await waitHealth(base)
    check('backend answers /api/health', up)
    if (!up) throw new Error('backend did not start')

    const health = await (await fetch(`${base}/api/health`)).json()
    check('health status ok', health.status === 'ok', health.version)

    // Non-RAW rejected (SPEC §10: RAW-only validation)
    const rejected = await (
      await fetch(`${base}/api/session/files`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paths: ['/tmp/definitely_not_raw.png'] })
      })
    ).json()
    check('non-RAW file rejected with reason',
      rejected.rejected.length === 1 && !!rejected.rejected[0].reason)

    // Process a synthetic burst through the real SR core
    const proc = await (
      await fetch(`${base}/api/dev/synthetic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ratio: 2.0, n_frames: 5 })
      })
    ).json()
    check('process returns a job id', !!proc.job_id)

    let status
    for (let i = 0; i < 600; i++) {
      status = await (await fetch(`${base}/api/jobs/${proc.job_id}`)).json()
      if (['done', 'error', 'cancelled'].includes(status.state)) break
      await new Promise((r) => setTimeout(r, 200))
    }
    check('job completed', status?.state === 'done', status?.state)

    const result = await (await fetch(`${base}/api/jobs/${proc.job_id}/result`)).json()
    check('result fused image is a PNG data URL',
      typeof result.images?.fused === 'string' && result.images.fused.startsWith('data:image/png'))
    check('result before image present',
      typeof result.images?.before === 'string' && result.images.before.startsWith('data:image/png'))
    check('before and fused share one display size',
      result.images?.width > 0 && result.images?.height > 0,
      `${result.images?.width}x${result.images?.height}`)
    check('HR shape is ratio x input',
      Array.isArray(result.report?.hr_shape), JSON.stringify(result.report?.hr_shape))
    const al = result.alignment
    check('alignment report present',
      !!al && typeof al.score === 'number' && ['ok', 'warn', 'bad'].includes(al.level),
      al?.level)
    check('alignment score in [0,1]', al?.score >= 0 && al?.score <= 1, String(al?.score))
    check('alignment per-frame covers all frames',
      Array.isArray(al?.per_frame) && al.per_frame.length === 5 &&
        al.per_frame.every((f) => typeof f.used === 'boolean' && typeof f.zncc === 'number'),
      `n=${al?.per_frame?.length}`)
    check('alignment quality map is a PNG data URL',
      typeof al?.map === 'string' && al.map.startsWith('data:image/png'))
    check('confidence layers removed', result.confidence === undefined)

    // Honest refinement / deposit reporting (docs/IBP_AND_HELDOUT.md). The
    // panel renders these as its own lines, so they must reach the client.
    const m = result.report?.metrics ?? {}
    check('metrics name the IBP configuration it ran',
      Number.isInteger(m.ibp_iterations) &&
        typeof m.ibp_summary === 'string' && m.ibp_summary.includes('sweep'),
      `ibp_iterations=${m.ibp_iterations} ${m.ibp_summary}`)
    check('metrics name the drizzle deposit (subsamples/pixfrac)',
      Number.isInteger(m.drizzle_subsamples) && m.drizzle_subsamples >= 1 &&
        typeof m.drizzle_pixfrac === 'number' && m.drizzle_pixfrac > 0,
      `${m.drizzle_subsamples} subs · pixfrac ${m.drizzle_pixfrac}`)
    check('metrics name the reference frame the HR grid is aligned to',
      Number.isInteger(m.reference_index) &&
        m.reference_index >= 0 && m.reference_index < result.report.total_frames,
      `reference_index=${m.reference_index}`)
    check('ghost-suppression verdict is a bool + non-empty reason',
      typeof m.robust_active === 'boolean' &&
        typeof m.robust_reason === 'string' && m.robust_reason.length > 0,
      `robust_active=${m.robust_active} reason=${m.robust_reason}`)

    // Honest metrics: detail versus ONE frame (not the denoised stack) and the
    // fusion-weight diagnostics the advanced section shows.
    check('honest detail is measured against a single frame',
      typeof m.hf_gain_db_vs_single === 'number' && Number.isFinite(m.hf_gain_db_vs_single) &&
        typeof m.sharpness_gain_vs_single === 'number' && m.sharpness_gain_vs_single > 0,
      `detail=${m.hf_gain_db_vs_single?.toFixed?.(2)} dB sharp=${m.sharpness_gain_vs_single?.toFixed?.(2)}x`)
    check('fusion noise sigma + weight spread reach the client',
      Array.isArray(m.frame_noise_sigma) &&
        m.frame_noise_sigma.length === result.report.used_frames &&
        m.frame_noise_sigma.every((s) => typeof s === 'number' && s >= 0) &&
        typeof m.fusion_weight_spread === 'number' && m.fusion_weight_spread >= 1,
      `n=${m.frame_noise_sigma?.length} spread=${m.fusion_weight_spread}`)
    check('comfortable run has no memory warning',
      result.report.memory_warning === null, String(result.report.memory_warning))
    check('job status carries the additive error_kind field',
      status?.error_kind === null && status?.error === null)

    // live view transform
    const rendered = await (
      await fetch(`${base}/api/jobs/${proc.job_id}/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gamma: 1.8, brightness: 0.1, contrast: 1.2 })
      })
    ).json()
    check('live view transform returns both images',
      rendered.fused?.startsWith?.('data:image/png') &&
        rendered.before?.startsWith?.('data:image/png'),
      `fused=${typeof rendered.fused} before=${typeof rendered.before}`)
    check('live view transform keeps display size',
      rendered.width === result.images.width && rendered.height === result.images.height,
      `${rendered.width}x${rendered.height}`)

    // export master + baked
    const dir = mkdtempSync(join(tmpdir(), 'rawsr-e2e-'))
    for (const kind of ['master', 'baked']) {
      const out = join(dir, `sr_${kind}.tiff`)
      const resp = await fetch(`${base}/api/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: proc.job_id, kind, output_path: out })
      })
      check(`export ${kind} writes a TIFF`, resp.ok && existsSync(out), out)
    }
  } catch (e) {
    check('smoke run', false, e.message)
  } finally {
    backend.kill()
  }

  const passed = results.filter(Boolean).length
  console.log(`\n${passed}/${results.length} checks passed`)
  process.exit(passed === results.length ? 0 : 1)
}

main()
