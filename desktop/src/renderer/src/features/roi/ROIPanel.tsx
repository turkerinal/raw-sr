/**
 * ROI panel: zoomable/pannable preview with a drag-drawn, resizable Target ROI
 * and an auto-derived, resizable Search Area, numeric readouts, a motion-aware
 * "reset" (largest fully-covered crop), and live "N/total frames usable"
 * feedback. SPEC §10.
 *
 * Top-level mode is a two-way segmented control — **Target** and **Pan**:
 *   • Pan (default) — drag pans the view; drawing is OFF until the user picks
 *     Target, so a press can never accidentally start a ROI.
 *   • Target — a press that is not on a resize handle starts a FRESH target
 *     draw. The interior of a rect is never a move target, so a full-frame ROI
 *     (e.g. right after "Reset to full frame") can still be replaced by
 *     dragging anywhere.
 * Resize handles are live in BOTH modes, for BOTH the target and the search
 * rect (8 each, corner > edge), so an edge can always be grabbed to correct
 * either region. When two handles compete at the pointer (target edge inside
 * the grab tolerance of the search edge), the one CLOSEST to the pointer in
 * screen space wins — see `handleAt` — and hover/press share that one rule, so
 * the cursor always previews what a press will grab. Middle-mouse / Space pan
 * in every mode.
 *
 * Zoom/pan lives here (not in ui/ZoomPanView) because the editable overlays
 * must transform together with the preview image so the pointer→source mapping
 * stays exact. The transform is `translate(offset) scale(zoom)` with
 * `transform-origin: 0 0`; the image and the overlay share one transformed
 * container.
 *
 * Owned by AGENT-D.
 */
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent
} from 'react'
import type { MotionInfo, Rect } from '../../api/types'
import { Badge, Button, Card, EmptyState, Tabs } from '../../ui'
import { RoiBox, RoiReadout } from './ROIRect'
import { useDragRect, type RoiRectId } from './useDragRect'
import {
  bestCoverageRect,
  clamp,
  handleCenter,
  hitRect,
  motionMargin,
  round,
  type RoiHandle
} from './geometry'
import './roi.css'

export interface ROIPanelProps {
  preview: string | null
  width: number
  height: number
  targetRoi: Rect | null
  searchArea: Rect | null
  motion: MotionInfo | null
  onChange: (target: Rect | null, search: Rect | null) => void
  usableFrames: number
  totalFrames: number
}

/** Top-level tool: draw/resize the target, or just pan the view. */
type PanelMode = 'target' | 'pan'

const MODE_TABS: Array<{ id: PanelMode; label: string }> = [
  { id: 'target', label: 'Target' },
  { id: 'pan', label: 'Pan' }
]

const MODE_HINT: Record<PanelMode, string> = {
  target: 'Drag to draw a new target; drag the handles to adjust ROI / search.',
  pan: 'Drag to pan; Ctrl+wheel to zoom.'
}

const MIN_ZOOM = 1
const MAX_ZOOM = 12
const ZOOM_STEP = 1.25
/** Handle hit radius + visual size in screen pixels (constant across zoom). */
const HANDLE_PX = 9

const HANDLE_CURSOR: Record<RoiHandle, string> = {
  nw: 'nwse-resize',
  n: 'ns-resize',
  ne: 'nesw-resize',
  e: 'ew-resize',
  se: 'nwse-resize',
  s: 'ns-resize',
  sw: 'nesw-resize',
  w: 'ew-resize'
}

interface Point {
  x: number
  y: number
}

/**
 * The stage's CONTENT box in client coordinates.
 *
 * The image and the overlay rects are laid out inside the stage's border
 * (`clientLeft`/`clientTop` wide) via `.roi-content { inset: 0 }`, so the
 * pointer→source mapping and the pan/zoom clamping must be expressed relative
 * to the content box, not the border box returned by `getBoundingClientRect()`.
 * Mixing the two lags drawn rects from the cursor by the border width (which
 * the zoom then amplifies).
 */
interface StageBox {
  left: number
  top: number
  width: number
  height: number
}

function contentBox(el: HTMLElement): StageBox {
  const r = el.getBoundingClientRect()
  return {
    left: r.left + el.clientLeft,
    top: r.top + el.clientTop,
    width: el.clientWidth,
    height: el.clientHeight
  }
}

export function ROIPanel({
  preview,
  width,
  height,
  targetRoi,
  searchArea,
  motion,
  onChange,
  usableFrames,
  totalFrames
}: ROIPanelProps) {
  const stageRef = useRef<HTMLDivElement>(null)

  const editor = useDragRect({
    target: targetRoi,
    search: searchArea,
    sensorWidth: width,
    sensorHeight: height,
    motion,
    onChange
  })

  const hasSensor = width > 0 && height > 0
  const badgeTone = usableFrames === 0 ? 'bad' : usableFrames < totalFrames ? 'warn' : 'ok'

  /* ---------------- mode / zoom / pan state ---------------- */
  /** Drawing is OFF by default: the user must pick Target explicitly. */
  const [mode, setMode] = useState<PanelMode>('pan')
  /** The rect that reads as "being edited" for the active highlight. */
  const [activeRect, setActiveRect] = useState<RoiRectId>('target')
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const [spaceHeld, setSpaceHeld] = useState(false)
  const [hoverCursor, setHoverCursor] = useState('crosshair')
  const [cropNote, setCropNote] = useState<string | null>(null)

  const zoomRef = useRef(zoom)
  const offsetRef = useRef(offset)
  zoomRef.current = zoom
  offsetRef.current = offset

  // Mirrored so handleAt (a stable callback) can break handle-proximity ties
  // with the *current* active rect without re-subscribing on every change.
  const activeRectRef = useRef(activeRect)
  activeRectRef.current = activeRect

  const panRef = useRef<{ px: number; py: number; ox: number; oy: number } | null>(null)
  const spaceRef = useRef(false)
  spaceRef.current = spaceHeld

  /** Pick the top-level mode; entering Target makes the target the active rect. */
  const chooseMode = useCallback((next: PanelMode) => {
    setMode(next)
    if (next === 'target') setActiveRect('target')
  }, [])

  /** Commit zoom + offset together and keep the refs immediately in sync. */
  const setView = useCallback((z: number, o: Point) => {
    zoomRef.current = z
    offsetRef.current = o
    setZoom(z)
    setOffset(o)
  }, [])

  /** Keep the scaled content covering the viewport; 1× pins offset to 0,0. */
  const clampOffset = useCallback((o: Point, z: number): Point => {
    const el = stageRef.current
    if (!el) return o
    const b = contentBox(el)
    if (b.width <= 0 || b.height <= 0) return o
    const minX = b.width * (1 - z)
    const minY = b.height * (1 - z)
    return { x: clamp(o.x, minX, 0), y: clamp(o.y, minY, 0) }
  }, [])

  /**
   * Zoom so the point `(sx, sy)` (stage-relative CSS px) stays under the
   * cursor. With `transform-origin: 0 0` and `scale(z)` the content point under
   * a stage point `s` is `c = (s - offset) / z`; keeping `c` fixed gives
   * `offset' = s - (z'/z) · (s - offset)`.
   */
  const zoomAt = useCallback(
    (targetZoom: number, sx: number, sy: number) => {
      const z = zoomRef.current
      const o = offsetRef.current
      const z2 = clamp(targetZoom, MIN_ZOOM, MAX_ZOOM)
      if (z2 === z) return
      const ratio = z2 / z
      const next = { x: sx - ratio * (sx - o.x), y: sy - ratio * (sy - o.y) }
      setView(z2, clampOffset(next, z2))
    },
    [clampOffset, setView]
  )

  const resetView = useCallback(() => setView(MIN_ZOOM, { x: 0, y: 0 }), [setView])

  /** Ctrl+wheel (or ⌘+wheel) zooms toward the cursor; plain wheel scrolls. */
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      const b = el.getBoundingClientRect()
      const factor = Math.exp(-e.deltaY * 0.0015)
      zoomAt(zoomRef.current * factor, e.clientX - b.left, e.clientY - b.top)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoomAt, preview, hasSensor])

  // A new frame resets the view to fit.
  useEffect(() => {
    setView(MIN_ZOOM, { x: 0, y: 0 })
  }, [preview, width, height, setView])

  // Space temporarily enables panning (ignored while typing in a field).
  useEffect(() => {
    const isField = (t: EventTarget | null) => {
      const el = t as HTMLElement | null
      const tag = el?.tagName
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
    }
    const down = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || isField(e.target)) return
      e.preventDefault()
      setSpaceHeld(true)
    }
    const up = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return
      setSpaceHeld(false)
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  /** Map a client point to source pixels, inverting the zoom/pan transform. */
  const toSource = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      const el = stageRef.current
      if (!el || !hasSensor) return [0, 0]
      const box = contentBox(el)
      if (box.width <= 0 || box.height <= 0) return [0, 0]
      const sx = clientX - box.left
      const sy = clientY - box.top
      const z = zoomRef.current
      const o = offsetRef.current
      // Stage point -> content point -> source pixel.
      const cx = (sx - o.x) / z
      const cy = (sy - o.y) / z
      return [(cx / box.width) * width, (cy / box.height) * height]
    },
    [hasSensor, width, height]
  )

  /**
   * Source px -> screen px scale of the current view (0 when not measurable).
   * The stage keeps the sensor aspect ratio and the zoom is uniform, so one
   * factor covers both axes.
   */
  const screenScale = useCallback((): number => {
    const el = stageRef.current
    if (!el || !hasSensor) return 0
    const box = contentBox(el)
    return (box.width / width) * zoomRef.current
  }, [hasSensor, width])

  /** Handle grab tolerance converted from screen px to source px at this zoom. */
  const handleTol = useCallback((): number => {
    const scale = screenScale()
    if (!(scale > 0)) return 0
    return HANDLE_PX / scale
  }, [screenScale])

  /**
   * Hit-test the resize handles of both rects. `hitRect` already prioritises
   * corners over edges *within* one rect. Between the two rects a fixed
   * target-first order is wrong: right after a draw the auto search area wraps
   * the target with a margin that can be a few screen px at 1x zoom — well
   * inside the 9 px grab tolerance — so dragging the visible search edge would
   * silently grab the target handle underneath. When BOTH rects have a handle
   * hit at the pointer, the handle whose CENTER is closest to the pointer in
   * SCREEN space wins; an exact tie goes to `activeRect` (the rect most
   * recently interacted with). A hit in only one rect behaves as before.
   */
  const handleAt = useCallback(
    (x: number, y: number, tol: number): { rect: RoiRectId; handle: RoiHandle } | null => {
      const scale = screenScale()
      // Compare distances in screen px (content-relative): "closest" must mean
      // closest on the display, matching where the user aims.
      const px = x * scale
      const py = y * scale
      const hits: Array<{ rect: RoiRectId; handle: RoiHandle; dist: number; active: boolean }> = []
      const consider = (rect: RoiRectId, r: Rect | null): void => {
        if (!r) return
        const hit = hitRect(r, x, y, tol)
        if (!hit || hit === 'inside') return
        const [hx, hy] = handleCenter(r, hit)
        hits.push({
          rect,
          handle: hit,
          dist: Math.hypot(hx * scale - px, hy * scale - py),
          active: rect === activeRectRef.current
        })
      }
      consider('target', editor.liveTarget)
      consider('search', editor.liveSearch)
      if (hits.length === 0) return null
      let best = hits[0]
      for (const h of hits) {
        if (h.dist < best.dist || (h.dist === best.dist && h.active && !best.active)) best = h
      }
      return { rect: best.rect, handle: best.handle }
    },
    [editor.liveTarget, editor.liveSearch, screenScale]
  )

  const beginPan = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const o = offsetRef.current
    panRef.current = { px: e.clientX, py: e.clientY, ox: o.x, oy: o.y }
    setPanning(true)
  }, [])

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    const el = stageRef.current
    if (!el) return
    const middle = e.button === 1
    if (!middle && e.button !== 0) return

    el.setPointerCapture(e.pointerId)

    // Middle-drag or Space always pans, in every mode.
    if (middle || spaceRef.current) {
      e.preventDefault()
      beginPan(e)
      return
    }
    if (!hasSensor) return

    const [x, y] = toSource(e.clientX, e.clientY)
    const tol = handleTol()

    // Handles are live in BOTH modes, for BOTH rects (corner > edge).
    const hit = handleAt(x, y, tol)
    if (hit) {
      e.preventDefault()
      setActiveRect(hit.rect)
      editor.begin('resize', hit.rect, x, y, hit.handle)
      return
    }

    // No handle. Pan mode pans; Target mode starts a FRESH target draw. The
    // plain interior is never a move target, so a full-frame ROI cannot block
    // drawing a new, smaller target.
    e.preventDefault()
    if (mode === 'pan') {
      beginPan(e)
      return
    }
    setActiveRect('target')
    editor.begin('draw', 'target', x, y)
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (panRef.current) {
      const p = panRef.current
      setOffset(clampOffset({ x: p.ox + (e.clientX - p.px), y: p.oy + (e.clientY - p.py) }, zoomRef.current))
      return
    }
    if (editor.dragging) {
      const [x, y] = toSource(e.clientX, e.clientY)
      editor.move(x, y)
      return
    }
    if (!hasSensor) return
    const [x, y] = toSource(e.clientX, e.clientY)
    const tol = handleTol()
    // Handles win the cursor in both modes; otherwise the mode's own cursor.
    // Same handleAt (incl. the both-rects proximity rule) that onPointerDown
    // uses, so the cursor previews exactly the handle a press would grab.
    const hit = handleAt(x, y, tol)
    if (hit) setHoverCursor(HANDLE_CURSOR[hit.handle])
    else setHoverCursor(mode === 'pan' ? 'grab' : 'crosshair')
  }

  const endPointer = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (panRef.current) {
      panRef.current = null
      setPanning(false)
    } else {
      editor.end()
    }
    const el = stageRef.current
    if (el && el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId)
  }

  const zoomByStep = (factor: number) => {
    const box = stageRef.current?.getBoundingClientRect()
    const cx = box ? box.width / 2 : 0
    const cy = box ? box.height / 2 : 0
    zoomAt(zoomRef.current * factor, cx, cy)
  }

  const resetToBest = () => {
    editor.resetToBest()
    // After a reset the ROI covers (at least most of) the frame; drop into
    // Target so the user can immediately drag anywhere to draw a new target.
    chooseMode('target')
    const crop = bestCoverageRect(width, height, motion)
    if (!crop) {
      setCropNote(`No usable motion data — using the full ${round(width)}×${round(height)} frame.`)
      return
    }
    const margin = motionMargin(motion)
    setCropNote(
      `Best crop ${round(crop[0])},${round(crop[1])} · ${round(crop[2])}×${round(crop[3])} px` +
        ` (max shift ±${round(Math.abs(motion?.max_shift ?? 0))}px, +${margin}px search margin).`
    )
  }

  const stageCursor = panning ? 'grabbing' : spaceHeld ? 'grab' : hoverCursor

  return (
    <Card
      title="Region of Interest"
      hint="target + search area"
      actions={
        <div className="row" style={{ gap: 8 }}>
          <Badge tone={badgeTone}>
            {usableFrames}/{totalFrames} frames usable
          </Badge>
          <Button
            size="sm"
            variant="ghost"
            onClick={resetToBest}
            disabled={!hasSensor}
            title="Largest fully-covered crop from inter-frame motion; falls back to the full frame"
          >
            Reset to full frame
          </Button>
        </div>
      }
    >
      {!preview ? (
        <EmptyState icon="🔍" title="No preview" hint="Add RAW frames to select a region." />
      ) : !hasSensor ? (
        <EmptyState icon="🔍" title="No frame dimensions" hint="The selected frame reports no size." />
      ) : (
        <div className="stack">
          <div className="roi-toolbar">
            <div className="roi-toolbar__group">
              <Tabs
                tabs={MODE_TABS}
                active={mode}
                onChange={(id) => chooseMode(id as PanelMode)}
              />
              <span className="tiny faint">{MODE_HINT[mode]}</span>
            </div>
            <div className="roi-toolbar__group">
              <Button size="sm" variant="ghost" onClick={() => zoomByStep(1 / ZOOM_STEP)} disabled={zoom <= MIN_ZOOM} title="Zoom out" aria-label="Zoom out">
                −
              </Button>
              <span className="roi-zoom__value mono tiny">{zoom.toFixed(2)}×</span>
              <Button size="sm" variant="ghost" onClick={() => zoomByStep(ZOOM_STEP)} disabled={zoom >= MAX_ZOOM} title="Zoom in" aria-label="Zoom in">
                +
              </Button>
              <Button size="sm" variant="ghost" onClick={resetView} disabled={zoom === MIN_ZOOM && offset.x === 0 && offset.y === 0} title="Fit to view">
                Fit
              </Button>
            </div>
          </div>

          {cropNote && <div className="roi-cropnote tiny faint">{cropNote}</div>}

          <div
            ref={stageRef}
            className={`roi-stage${editor.dragging ? ' roi-stage--dragging' : ''}${panning ? ' roi-stage--panning' : ''}`}
            style={{ aspectRatio: `${width} / ${height}`, cursor: stageCursor }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endPointer}
            onPointerCancel={endPointer}
            onDoubleClick={mode === 'pan' ? resetView : undefined}
          >
            <div
              className="roi-content"
              style={
                {
                  transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
                  // Keep handles a constant on-screen size while the content zooms.
                  '--roi-handle-scale': String(1 / zoom)
                } as CSSProperties
              }
            >
              <img src={preview} alt="reference frame preview" draggable={false} />
              <div className="roi-overlay">
                {editor.liveSearch && (
                  <RoiBox
                    rect={editor.liveSearch}
                    kind="search"
                    sensorWidth={width}
                    sensorHeight={height}
                    active={activeRect === 'search'}
                  />
                )}
                {editor.liveTarget && (
                  <RoiBox
                    rect={editor.liveTarget}
                    kind="target"
                    sensorWidth={width}
                    sensorHeight={height}
                    active={activeRect === 'target'}
                  />
                )}
              </div>
            </div>
          </div>

          <div className="roi-readouts">
            <RoiReadout
              kind="target"
              rect={editor.liveTarget}
              sensorWidth={width}
              sensorHeight={height}
              onChange={(rect) => editor.setRect('target', rect)}
              onClear={() => editor.clear('target')}
            />
            <RoiReadout
              kind="search"
              rect={editor.liveSearch}
              sensorWidth={width}
              sensorHeight={height}
              onChange={(rect) => editor.setRect('search', rect)}
              onClear={() => editor.clear('search')}
            />
          </div>
        </div>
      )}
    </Card>
  )
}
