/**
 * ZoomPanView — shared zoom + pan viewer for a single image (SPEC §10).
 *
 * Interaction:
 *   • mouse wheel zooms toward the cursor (optionally only while Ctrl is held)
 *   • drag pans (pointer capture)
 *   • double-click resets to fit
 *   • optional toolbar: Fit / 1:1 / zoom in / zoom out / reset, plus a readout
 *
 * Zoom is expressed as a factor over "fit" (1× = image fitted inside the
 * viewport) and is clamped to [minZoom, maxZoom]. The viewport has
 * `overflow: hidden` and the transform is applied to the inner image, so the
 * component stays usable at any container size. No external libraries.
 */
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent
} from 'react'
import './ZoomPanView.css'

export interface ZoomPanViewProps {
  /** data URL or URL of the image to display */
  src: string
  /** accessible description */
  alt?: string
  /** minimum zoom factor over fit (default 1) */
  minZoom?: number
  /** maximum zoom factor over fit (default 12) */
  maxZoom?: number
  /** when true, the wheel zooms only while Ctrl is held */
  ctrlToZoom?: boolean
  /** show the Fit / 1:1 toolbar (default true) */
  controls?: boolean
}

interface Size {
  w: number
  h: number
}

const STEP = 1.25

export function ZoomPanView({
  src,
  alt = 'image',
  minZoom = 1,
  maxZoom = 12,
  ctrlToZoom = false,
  controls = true
}: ZoomPanViewProps) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const [natural, setNatural] = useState<Size | null>(null)
  const [box, setBox] = useState<Size>({ w: 0, h: 0 })
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)
  const dragRef = useRef<{ px: number; py: number; ox: number; oy: number } | null>(null)

  const hasImage = src.trim().length > 0

  const fitScale =
    natural && box.w > 0 && box.h > 0 ? Math.min(box.w / natural.w, box.h / natural.h) : 1
  const scale = fitScale * zoom

  // Latest values for the stable (native, non-passive) wheel listener.
  const liveRef = useRef({ zoom, offset })
  liveRef.current = { zoom, offset }

  const clampZoom = useCallback(
    (z: number) => Math.min(maxZoom, Math.max(minZoom, z)),
    [minZoom, maxZoom]
  )

  const reset = useCallback(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [])

  // A new image invalidates the measured natural size and the current view.
  useEffect(() => {
    setNatural(null)
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [src])

  // Measure the viewport and keep it in sync on resize.
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const update = () => setBox({ w: el.clientWidth, h: el.clientHeight })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const zoomAt = useCallback(
    (targetZoom: number, rx: number, ry: number) => {
      const { zoom: z, offset: o } = liveRef.current
      const z2 = clampZoom(targetZoom)
      if (z2 === z) return
      const ratio = z2 / z
      setOffset({ x: rx - ratio * (rx - o.x), y: ry - ratio * (ry - o.y) })
      setZoom(z2)
    },
    [clampZoom]
  )

  const zoomBy = useCallback(
    (factor: number) => zoomAt(liveRef.current.zoom * factor, 0, 0),
    [zoomAt]
  )

  const actualPixels = useCallback(() => {
    setOffset({ x: 0, y: 0 })
    setZoom(clampZoom(fitScale > 0 ? 1 / fitScale : 1))
  }, [clampZoom, fitScale])

  // Native wheel listener so preventDefault actually stops page scrolling.
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (ctrlToZoom && !e.ctrlKey) return
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const rx = e.clientX - rect.left - rect.width / 2
      const ry = e.clientY - rect.top - rect.height / 2
      const factor = Math.exp(-e.deltaY * 0.0015)
      zoomAt(liveRef.current.zoom * factor, rx, ry)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoomAt, ctrlToZoom])

  const onPointerDown = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const o = liveRef.current.offset
    dragRef.current = { px: e.clientX, py: e.clientY, ox: o.x, oy: o.y }
    setDragging(true)
    e.currentTarget.setPointerCapture(e.pointerId)
  }, [])

  const onPointerMove = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const d = dragRef.current
    if (!d) return
    setOffset({ x: d.ox + (e.clientX - d.px), y: d.oy + (e.clientY - d.py) })
  }, [])

  const endDrag = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    dragRef.current = null
    setDragging(false)
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }
  }, [])

  return (
    <div className="zp">
      {controls && (
        <div className="zp__toolbar" role="toolbar" aria-label="Zoom and pan controls">
          <button type="button" className="zp__btn" title="Fit to view" onClick={reset}>
            Fit
          </button>
          <button type="button" className="zp__btn" title="Actual pixels (1:1)" onClick={actualPixels}>
            1:1
          </button>
          <span className="zp__sep" aria-hidden="true" />
          <button
            type="button"
            className="zp__btn"
            title="Zoom out"
            aria-label="Zoom out"
            disabled={zoom <= minZoom}
            onClick={() => zoomBy(1 / STEP)}
          >
            −
          </button>
          <button
            type="button"
            className="zp__btn"
            title="Zoom in"
            aria-label="Zoom in"
            disabled={zoom >= maxZoom}
            onClick={() => zoomBy(STEP)}
          >
            +
          </button>
          <button
            type="button"
            className="zp__btn"
            title="Reset view"
            aria-label="Reset view"
            onClick={reset}
          >
            ⟲
          </button>
          <span className="zp__zoom" aria-live="polite">
            {zoom.toFixed(2)}×
          </span>
        </div>
      )}

      <div
        ref={viewportRef}
        className={`zp__viewport${dragging ? ' is-dragging' : ''}${hasImage ? '' : ' is-empty'}`}
        onPointerDown={hasImage ? onPointerDown : undefined}
        onPointerMove={hasImage ? onPointerMove : undefined}
        onPointerUp={hasImage ? endDrag : undefined}
        onPointerCancel={hasImage ? endDrag : undefined}
        onDoubleClick={hasImage ? reset : undefined}
      >
        {hasImage ? (
          <img
            src={src}
            alt={alt}
            draggable={false}
            onLoad={(e) =>
              setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
            }
            className="zp__img"
            style={{
              width: natural ? natural.w : 'auto',
              height: natural ? natural.h : 'auto',
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`
            }}
          />
        ) : (
          <span className="zp__placeholder">No image</span>
        )}
      </div>
    </div>
  )
}
