/**
 * File selection panel: a compact vertical list of frame names with
 * auto-flagged blurry frames and a per-frame user override, a live active/total
 * count, and a rejected-files section. Names only — no thumbnails or previews.
 * SPEC §10: RAW-only validation with a clear rejection message.
 *
 * Owned by AGENT-C.
 */
import type { FrameInfo, RejectedFile } from '../../api/types'
import { Badge, Button, Card, EmptyState, Spinner } from '../../ui'
import { FrameRow } from './FrameRow'
import { RejectedList } from './RejectedList'
import './files.css'

export interface FilePanelProps {
  frames: FrameInfo[]
  rejected: RejectedFile[]
  loading: boolean
  excluded: Set<string>
  selectedId?: string | null
  perFrame?: Record<string, { used: boolean; zncc: number }> | null
  onAddFiles: () => void
  onRemoveFile: (id: string) => void
  onToggleBlur: (id: string) => void
  onSelectFrame?: (id: string) => void
}

export function FilePanel({
  frames,
  rejected,
  loading,
  excluded,
  selectedId,
  perFrame,
  onAddFiles,
  onRemoveFile,
  onToggleBlur,
  onSelectFrame
}: FilePanelProps) {
  const total = frames.length
  const active = total - excluded.size
  const blurry = frames.filter((f) => f.blurry).length
  const skipped = frames.filter((f) => excluded.has(f.id)).length

  const addButton = (
    <Button variant="primary" onClick={onAddFiles} disabled={loading}>
      {loading ? 'Loading…' : '+ Add RAW files'}
    </Button>
  )

  return (
    <Card
      title="Frames"
      hint={total > 0 ? `${active}/${total} active` : undefined}
      actions={
        <Button variant="secondary" size="sm" onClick={onAddFiles} disabled={loading}>
          + Add
        </Button>
      }
    >
      {loading && (
        <div className="files-loading">
          <Spinner />
          <span>Reading RAW files…</span>
        </div>
      )}

      {total === 0 ? (
        <div>
          <EmptyState
            icon="🗂"
            title="No RAW frames yet"
            hint="Add a handheld burst of the same static scene (2–20 frames)."
          />
          <div className="files-cta">{addButton}</div>
        </div>
      ) : (
        <>
          <div className="files-stats">
            <Badge tone="neutral">{total} frame{total === 1 ? '' : 's'}</Badge>
            <Badge tone={active > 0 ? 'ok' : 'bad'}>{active} active</Badge>
            {blurry > 0 && <Badge tone="warn">{blurry} auto-blurry</Badge>}
            {skipped > 0 && <Badge tone="neutral">{skipped} skipped</Badge>}
          </div>

          <div className="frame-list" role="list">
            {frames.map((f) => (
              <FrameRow
                key={f.id}
                frame={f}
                excluded={excluded.has(f.id)}
                selected={f.id === selectedId}
                result={perFrame?.[f.id] ?? null}
                onRemove={onRemoveFile}
                onToggleBlur={onToggleBlur}
                onSelect={onSelectFrame}
              />
            ))}
          </div>

          {blurry > 0 && (
            <p className="tiny faint" style={{ marginTop: 10 }}>
              Auto-flagged blurry frames are still used. Use “Skip” to override and exclude one;
              “Include” restores it.
            </p>
          )}
        </>
      )}

      <RejectedList rejected={rejected} />
    </Card>
  )
}
