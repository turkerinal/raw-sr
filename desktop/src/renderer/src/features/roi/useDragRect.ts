/**
 * Drag-draw / resize state machine for the ROI editor.
 *
 * Owns the *live* target/search rects (so overlays track the pointer at full
 * frame rate) while debouncing the `onChange` callback that reaches the parent
 * (~120 ms), so dragging does not spam live analysis.
 *
 * Interactions (all coordinates in/out are SOURCE pixels, clamped to sensor):
 *   • draw    — press on empty space, drag out a NEW target. Only the target is
 *               drawn; the search area is re-derived from it (see `end`).
 *   • resize  — press one of the 8 handles of the target or the search rect.
 *
 * The rectangle an interaction applies to is named directly (`RoiRectId`:
 * 'target' | 'search'). The panel owns the *top-level* mode (Target vs Pan) and
 * decides when draw/resize are allowed; this hook only carries it out. Note that
 * the interior of a rect is never a move target: a full-frame ROI can therefore
 * never absorb a press, so drawing a new, smaller target always works.
 *
 * INVARIANT: the search area ALWAYS strictly contains the target + motion
 * margin. It is enforced at every mutation point through two pure helpers in
 * ./geometry:
 *   • whenever the TARGET changes (draw, numeric edit, handle resize, reset),
 *     the search is re-derived as `deriveSearchAroundTarget(target, margin)`;
 *   • whenever the SEARCH changes directly (numeric edit, handle resize), it is
 *     clamped with `clampSearchContainsTarget(search, target)` so it can never
 *     end up smaller than / outside the target.
 *
 * Owned by AGENT-D.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { MotionInfo, Rect } from '../../api/types'
import {
  MIN_DRAW_SIZE,
  bestCoverageRect,
  clamp,
  clampRect,
  clampSearchContainsTarget,
  deriveSearchAroundTarget,
  motionMargin,
  rectFromPoints,
  resizeRect,
  roundRect,
  type RoiHandle
} from './geometry'

/** Which editable rectangle an interaction applies to. */
export type RoiRectId = 'target' | 'search'
export type DragKind = 'draw' | 'resize'

export interface UseDragRectOptions {
  target: Rect | null
  search: Rect | null
  sensorWidth: number
  sensorHeight: number
  motion: MotionInfo | null
  onChange: (target: Rect | null, search: Rect | null) => void
  debounceMs?: number
}

export interface DragRectController {
  /** Live rects for rendering; may be ahead of the (debounced) parent props. */
  liveTarget: Rect | null
  liveSearch: Rect | null
  dragging: boolean
  /** Begin a pointer interaction in source coordinates. */
  begin: (kind: DragKind, rect: RoiRectId, x: number, y: number, handle?: RoiHandle) => void
  move: (x: number, y: number) => void
  end: () => void
  /** Replace a rect (numeric fields). Clamped + debounced. */
  setRect: (rect: RoiRectId, value: Rect) => void
  /** Set a rect to "unspecified". */
  clear: (rect: RoiRectId) => void
  /**
   * Motion-aware reset: target = largest fully-covered crop (falls back to the
   * full frame), search = that region expanded by the motion margin.
   */
  resetToBest: () => void
}

interface DragState {
  kind: DragKind
  rect: RoiRectId
  handle?: RoiHandle
  /** Press point, source px. */
  x0: number
  y0: number
  /** Rect before the interaction (resize), or the previous target (draw). */
  before: Rect | null
}

export function useDragRect({
  target,
  search,
  sensorWidth,
  sensorHeight,
  motion,
  onChange,
  debounceMs = 120
}: UseDragRectOptions): DragRectController {
  const [liveTarget, setLiveTarget] = useState<Rect | null>(target)
  const [liveSearch, setLiveSearch] = useState<Rect | null>(search)
  const [dragging, setDragging] = useState(false)

  // Refs mirror the latest values so callbacks and timers never read stale state.
  const liveTargetRef = useRef(liveTarget)
  liveTargetRef.current = liveTarget
  const liveSearchRef = useRef(liveSearch)
  liveSearchRef.current = liveSearch
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange
  const motionRef = useRef(motion)
  motionRef.current = motion
  const sensorRef = useRef({ w: sensorWidth, h: sensorHeight })
  sensorRef.current = { w: sensorWidth, h: sensorHeight }

  const dragRef = useRef<DragState | null>(null)
  const timerRef = useRef<number | null>(null)
  const lastCommittedRef = useRef<string>(serialize(target, search))

  const hasSensor = sensorWidth > 0 && sensorHeight > 0

  // Adopt external prop changes unless the user is mid-drag.
  useEffect(() => {
    if (!dragRef.current) setLiveTarget(target)
  }, [target])
  useEffect(() => {
    if (!dragRef.current) setLiveSearch(search)
  }, [search])

  const commit = useCallback(() => {
    if (timerRef.current != null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
    const t = liveTargetRef.current
    const s = liveSearchRef.current
    const payload = serialize(t, s)
    if (payload === lastCommittedRef.current) return
    lastCommittedRef.current = payload
    onChangeRef.current(t, s)
  }, [])

  const schedule = useCallback(() => {
    if (timerRef.current != null) window.clearTimeout(timerRef.current)
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null
      commit()
    }, debounceMs)
  }, [commit, debounceMs])

  useEffect(() => {
    return () => {
      if (timerRef.current != null) window.clearTimeout(timerRef.current)
    }
  }, [])

  const applyLive = useCallback((next: Rect | null, rect: RoiRectId) => {
    if (rect === 'target') {
      liveTargetRef.current = next
      setLiveTarget(next)
    } else {
      liveSearchRef.current = next
      setLiveSearch(next)
    }
  }, [])

  /** Canonical search area around `target` for the current motion. */
  const searchFor = useCallback((rect: Rect): Rect => {
    const { w, h } = sensorRef.current
    return deriveSearchAroundTarget(rect, motionMargin(motionRef.current), w, h)
  }, [])

  const begin = useCallback(
    (kind: DragKind, rect: RoiRectId, x: number, y: number, handle?: RoiHandle) => {
      if (!hasSensor) return
      const { w, h } = sensorRef.current
      const before = rect === 'target' ? liveTargetRef.current : liveSearchRef.current
      const sx = clamp(x, 0, w)
      const sy = clamp(y, 0, h)
      dragRef.current = { kind, rect, handle, x0: sx, y0: sy, before }
      setDragging(true)
      // A draw starts collapsed at the press point; it grows as the pointer
      // moves. A resize keeps the rect in place on press (nothing to preview).
      if (kind === 'draw') applyLive([sx, sy, 0, 0], rect)
    },
    [applyLive, hasSensor]
  )

  const move = useCallback(
    (x: number, y: number) => {
      const drag = dragRef.current
      if (!drag) return
      const { w, h } = sensorRef.current
      if (drag.kind === 'draw') {
        applyLive(rectFromPoints(drag.x0, drag.y0, x, y, w, h), drag.rect)
      } else if (drag.before && drag.handle) {
        applyLive(resizeRect(drag.before, drag.handle, x, y, w, h), drag.rect)
      }
      schedule()
    },
    [applyLive, schedule]
  )

  const end = useCallback(() => {
    const drag = dragRef.current
    if (!drag) return
    dragRef.current = null
    setDragging(false)
    const current = drag.rect === 'target' ? liveTargetRef.current : liveSearchRef.current

    if (drag.kind === 'draw') {
      // A click / accidental twitch must not wipe an existing rect.
      if (!current || current[2] < MIN_DRAW_SIZE || current[3] < MIN_DRAW_SIZE) {
        applyLive(drag.before, drag.rect)
        commit()
        return
      }
      // A brand-new target recomputes the search area from the motion margin.
      if (drag.rect === 'target') {
        applyLive(searchFor(current), 'search')
      }
      commit()
      return
    }

    // resize: clamp once more, then enforce the search-contains-target
    // invariant. A resized TARGET re-derives the search from the motion margin;
    // a resized SEARCH is clamped so it still encloses the target.
    if (current) {
      const { w, h } = sensorRef.current
      const clamped = roundRect(clampRect(current, w, h))
      if (drag.rect === 'target') {
        applyLive(clamped, 'target')
        applyLive(searchFor(clamped), 'search')
      } else {
        const t = liveTargetRef.current
        applyLive(t ? clampSearchContainsTarget(clamped, t, w, h) : clamped, 'search')
      }
    }
    commit()
  }, [applyLive, commit, searchFor])

  const setRect = useCallback(
    (rect: RoiRectId, value: Rect) => {
      const { w, h } = sensorRef.current
      const clamped = roundRect(clampRect(value, w, h))
      // Enforce the search-contains-target invariant: a target edit re-derives
      // the search; a direct search edit is clamped to enclose the target.
      if (rect === 'target') {
        applyLive(clamped, 'target')
        applyLive(searchFor(clamped), 'search')
      } else {
        const t = liveTargetRef.current
        applyLive(t ? clampSearchContainsTarget(clamped, t, w, h) : clamped, 'search')
      }
      schedule()
    },
    [applyLive, schedule, searchFor]
  )

  const clear = useCallback(
    (rect: RoiRectId) => {
      applyLive(null, rect)
      commit()
    },
    [applyLive, commit]
  )

  const resetToBest = useCallback(() => {
    if (!hasSensor) return
    const { w, h } = sensorRef.current
    const crop = bestCoverageRect(w, h, motionRef.current)
    if (!crop) {
      const full: Rect = [0, 0, w, h]
      applyLive(full, 'target')
      applyLive(full, 'search')
    } else {
      applyLive(crop, 'target')
      applyLive(deriveSearchAroundTarget(crop, motionMargin(motionRef.current), w, h), 'search')
    }
    commit()
  }, [applyLive, commit, hasSensor])

  return {
    liveTarget,
    liveSearch,
    dragging,
    begin,
    move,
    end,
    setRect,
    clear,
    resetToBest
  }
}

function serialize(t: Rect | null, s: Rect | null): string {
  return `${JSON.stringify(t)}|${JSON.stringify(s)}`
}
