/**
 * Pure geometry helpers for the ROI editor.
 *
 * Everything here works in SOURCE pixel coordinates (the full sensor,
 * `width` x `height`). Display coordinates are only produced at render time by
 * mapping a source value onto a percentage of the rendered preview box; the
 * zoom/pan transform that lives in ROIPanel is inverted before calling in here.
 *
 * Owned by AGENT-D.
 */
import type { MotionInfo, Rect } from '../../api/types'

/** Root-2 two-handle axis id. `'nw'` etc. are the 8 resize handles. */
export type RoiHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'

/** All handles in a stable order (corners then edges). */
export const ROI_HANDLES: RoiHandle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']

/** Minimum source-pixel size a rect may be resized/drawn to. */
export const MIN_ROI_SIZE = 8

/** Twitch threshold for a completed draw (shorter than this reverts). */
export const MIN_DRAW_SIZE = 4

/** Search-area margin floor, in source pixels (per side). */
export const MIN_MARGIN_PX = 16

/** Search-area margin used when no motion data is available (per side). */
export const DEFAULT_MARGIN_PX = 32

/** Clamp `value` into [lo, hi]; tolerates an inverted/hi<lo range. */
export function clamp(value: number, lo: number, hi: number): number {
  if (!Number.isFinite(value)) return lo
  if (hi < lo) return lo
  return Math.min(hi, Math.max(lo, value))
}

export function round(value: number): number {
  return Math.round(value)
}

/** Round every component of a rect to whole source pixels. */
export function roundRect(rect: Rect): Rect {
  return [Math.round(rect[0]), Math.round(rect[1]), Math.round(rect[2]), Math.round(rect[3])]
}

/**
 * Clamp a rect to the sensor bounds: the origin is pushed inside, then the
 * size is trimmed so the rect never extends past width/height. Never produces
 * negative width/height (which would break the backend contract).
 */
export function clampRect(rect: Rect, width: number, height: number): Rect {
  const x = clamp(rect[0], 0, width)
  const y = clamp(rect[1], 0, height)
  const w = clamp(rect[2], 0, width - x)
  const h = clamp(rect[3], 0, height - y)
  return [x, y, w, h]
}

/** Build a source-space rect from two arbitrary corner points, then clamp. */
export function rectFromPoints(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  width: number,
  height: number
): Rect {
  return roundRect(
    clampRect([Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)], width, height)
  )
}

/**
 * Translate a rect by a pointer delta, keeping its size and clamping the
 * origin so the whole rect stays inside the sensor.
 */
export function moveRect(
  rect: Rect,
  dx: number,
  dy: number,
  width: number,
  height: number
): Rect {
  return roundRect(clampRect([rect[0] + dx, rect[1] + dy, rect[2], rect[3]], width, height))
}

/**
 * Resize a rect from one of its 8 handles. The opposite edge(s) stay fixed;
 * the moving edge is clamped to the sensor and to a minimum size so the rect
 * can never invert or collapse.
 */
export function resizeRect(
  start: Rect,
  handle: RoiHandle,
  x: number,
  y: number,
  width: number,
  height: number,
  min = MIN_ROI_SIZE
): Rect {
  let l = start[0]
  let t = start[1]
  let r = start[0] + start[2]
  let b = start[1] + start[3]
  if (handle.includes('w')) l = clamp(x, 0, r - min)
  if (handle.includes('e')) r = clamp(x, l + min, width)
  if (handle.includes('n')) t = clamp(y, 0, b - min)
  if (handle.includes('s')) b = clamp(y, t + min, height)
  return roundRect([l, t, Math.max(0, r - l), Math.max(0, b - t)])
}

/**
 * Hit-test a source-space point against a rect.
 *
 * Priority: corner handle > edge handle > `'inside'` > `null` (outside).
 * `tol` is the handle tolerance in SOURCE pixels (the caller converts a fixed
 * screen tolerance through the current zoom so handles stay a constant size).
 */
export function hitRect(
  rect: Rect,
  x: number,
  y: number,
  tol: number
): RoiHandle | 'inside' | null {
  const [rx, ry, rw, rh] = rect
  const l = rx
  const r = rx + rw
  const t = ry
  const b = ry + rh
  if (x < l - tol || x > r + tol || y < t - tol || y > b + tol) return null
  const nearL = Math.abs(x - l) <= tol
  const nearR = Math.abs(x - r) <= tol
  const nearT = Math.abs(y - t) <= tol
  const nearB = Math.abs(y - b) <= tol
  if (nearL && nearT) return 'nw'
  if (nearR && nearT) return 'ne'
  if (nearL && nearB) return 'sw'
  if (nearR && nearB) return 'se'
  if (nearT && x >= l && x <= r) return 'n'
  if (nearB && x >= l && x <= r) return 's'
  if (nearL && y >= t && y <= b) return 'w'
  if (nearR && y >= t && y <= b) return 'e'
  if (x >= l && x <= r && y >= t && y <= b) return 'inside'
  return null
}

/**
 * The centre point (source px) of one of a rect's 8 resize handles: corners
 * sit on the corner, edge handles on the midpoint of that edge. Used to rank
 * competing handle hits by pointer proximity in screen space (see ROIPanel).
 */
export function handleCenter(rect: Rect, handle: RoiHandle): [number, number] {
  const [rx, ry, rw, rh] = rect
  const x = handle.includes('w') ? rx : handle.includes('e') ? rx + rw : rx + rw / 2
  const y = handle.includes('n') ? ry : handle.includes('s') ? ry + rh : ry + rh / 2
  return [x, y]
}

/**
 * Return a copy of `rect` with one component (`index` 0..3 = x,y,w,h) replaced,
 * clamped to the sensor AND to `MIN_ROI_SIZE` on each axis so a numeric edit can
 * never produce a degenerate (zero/negative) rect that the backend would reject.
 *
 * The origin is first bounded so the minimum size always fits; the size is then
 * trimmed to the sensor, but never below the minimum (or the sensor's own
 * dimension when that is smaller than `MIN_ROI_SIZE`).
 */
export function withRectField(
  rect: Rect,
  index: number,
  value: number,
  width: number,
  height: number
): Rect {
  const minW = Math.min(MIN_ROI_SIZE, width)
  const minH = Math.min(MIN_ROI_SIZE, height)
  const next: Rect = [rect[0], rect[1], rect[2], rect[3]]
  next[index] = Number.isFinite(value) ? value : 0
  // Keep the origin inside the sensor so a full minimum-size rect still fits.
  next[0] = clamp(next[0], 0, Math.max(0, width - minW))
  next[1] = clamp(next[1], 0, Math.max(0, height - minH))
  next[2] = clamp(next[2], minW, width - next[0])
  next[3] = clamp(next[3], minH, height - next[1])
  return roundRect(next)
}

/** A source value as a CSS percentage of the rendered preview (0 when unknown). */
export function toPercent(value: number, total: number): string {
  if (!(total > 0)) return '0%'
  return `${(value / total) * 100}%`
}

/* ------------------------------------------------------------------ */
/* Motion-aware search area + best (fully-covered) crop                */
/* ------------------------------------------------------------------ */

/**
 * Search-area margin (source px, per side) derived from inter-frame motion.
 *
 *   margin = max(16, ceil(max_shift) + 16)   when motion data is present
 *   margin = 32                              when it is absent
 *
 * The `+16` keeps a fixed search slack on top of the observed motion, and the
 * floor of 16 guarantees a usable search window even for a near-static burst.
 */
export function motionMargin(motion: MotionInfo | null): number {
  if (!motion || !Array.isArray(motion.shifts) || motion.shifts.length === 0) {
    return DEFAULT_MARGIN_PX
  }
  const shift = Number.isFinite(motion.max_shift) ? Math.abs(motion.max_shift) : 0
  if (shift <= 0) return MIN_MARGIN_PX
  return Math.max(MIN_MARGIN_PX, Math.ceil(shift) + MIN_MARGIN_PX)
}

/**
 * Grow `rect` by `margin` source pixels on every side, clamped to the sensor.
 * Unlike `clampRect`, an expansion that runs off one edge is trimmed there and
 * does NOT shift the opposite edge.
 */
export function expandRect(rect: Rect, margin: number, width: number, height: number): Rect {
  const x0 = clamp(rect[0] - margin, 0, width)
  const y0 = clamp(rect[1] - margin, 0, height)
  const x1 = clamp(rect[0] + rect[2] + margin, 0, width)
  const y1 = clamp(rect[1] + rect[3] + margin, 0, height)
  return [
    Math.round(x0),
    Math.round(y0),
    Math.max(0, Math.round(x1 - x0)),
    Math.max(0, Math.round(y1 - y0))
  ]
}

/**
 * THE SEARCH-CONTAINS-TARGET INVARIANT (single source of truth).
 *
 * The search area must always strictly contain the target plus at least
 * `margin` source px of slack per side (within the sensor), because
 * registration seeks the target's inter-frame motion *inside* the search
 * window: a search area that is smaller than / does not enclose the target
 * starves the search and silently degrades the result.
 *
 * Every mutation of either rect funnels through this helper. Edges are pushed
 * OUT only — a deliberately larger search is preserved, and the target is never
 * moved — while each offending side is grown to `target edge ± margin`. A side
 * that would leave the sensor is pinned to the sensor bound, which still fully
 * contains a sensor-clamped target (the only case where the margin can be 0 on
 * that side). Result is rounded and never negative.
 */
export function clampSearchContainsTarget(
  search: Rect,
  target: Rect,
  width: number,
  height: number,
  margin = MIN_MARGIN_PX
): Rect {
  const left = Math.min(search[0], target[0] - margin)
  const top = Math.min(search[1], target[1] - margin)
  const right = Math.max(search[0] + search[2], target[0] + target[2] + margin)
  const bottom = Math.max(search[1] + search[3], target[1] + target[3] + margin)
  const x0 = clamp(left, 0, width)
  const y0 = clamp(top, 0, height)
  const x1 = clamp(right, 0, width)
  const y1 = clamp(bottom, 0, height)
  return [Math.round(x0), Math.round(y0), Math.max(0, Math.round(x1 - x0)), Math.max(0, Math.round(y1 - y0))]
}

/**
 * The canonical search area for a `target`: the target grown by the motion
 * `margin` on every side, clamped to the sensor, then guaranteed to contain the
 * target (see `clampSearchContainsTarget`). This is the ONE derivation used by
 * draw, numeric edit, handle resize and "Reset to full frame".
 */
export function deriveSearchAroundTarget(
  target: Rect,
  margin: number,
  width: number,
  height: number
): Rect {
  return clampSearchContainsTarget(expandRect(target, margin, width, height), target, width, height, margin)
}

/**
 * The largest axis-aligned rectangle **fully covered by every frame**, derived
 * from the per-frame shifts (relative to the reference/first frame).
 *
 * Frame `i` with shift `(dx, dy)` sees a `width x height` window whose position
 * in reference coordinates is `[-dx, width - dx] x [-dy, height - dy]`. The
 * intersection of every frame is therefore:
 *
 *   left   = max_i(-dx_i) = -min(dx)
 *   right  = min_i(width - dx_i) = width - max(dx)
 *   ...
 *   width  = width  - (max(dx) - min(dx))
 *   height = height - (max(dy) - min(dy))
 *
 * i.e. the reference frame trimmed by the total motion RANGE on each axis.
 * Since more overlapping frames contribute to a pixel, this is the maximum
 * coverage / sharpness region.
 *
 * Returns `null` when the size is unknown, no usable shifts exist, or the
 * result is degenerate (range >= sensor on an axis) so the caller can fall back
 * to the full frame.
 */
export function bestCoverageRect(
  width: number,
  height: number,
  motion: MotionInfo | null
): Rect | null {
  if (!(width > 0) || !(height > 0)) return null
  const shifts = motion && Array.isArray(motion.shifts) ? motion.shifts : []
  const valid = shifts.filter(
    (s): s is [number, number] =>
      Array.isArray(s) && s.length >= 2 && Number.isFinite(s[0]) && Number.isFinite(s[1])
  )
  if (valid.length === 0) return null

  let minDx = Infinity
  let maxDx = -Infinity
  let minDy = Infinity
  let maxDy = -Infinity
  for (const [dx, dy] of valid) {
    if (dx < minDx) minDx = dx
    if (dx > maxDx) maxDx = dx
    if (dy < minDy) minDy = dy
    if (dy > maxDy) maxDy = dy
  }

  const w = width - (maxDx - minDx)
  const h = height - (maxDy - minDy)
  if (!(w >= 1) || !(h >= 1)) return null
  const x = -minDx
  const y = -minDy
  return roundRect(clampRect([x, y, w, h], width, height))
}
