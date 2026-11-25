/**
 * Rejected files section: shows files the backend refused, with the reason and
 * the SPEC §10 RAW-only policy note. Renders nothing when there are no rejects.
 *
 * Owned by AGENT-C.
 */
import type { RejectedFile } from '../../api/types'
import { Badge } from '../../ui'

export interface RejectedListProps {
  rejected: RejectedFile[]
}

/** SPEC §10: accepted RAW extensions (mirrors ARCHITECTURE.md). */
export const RAW_ONLY_MESSAGE =
  'Only RAW files are supported. Rejected files are not DNG, CR2/CR3, NEF/NRW, ' +
  'ARW/SRF/SR2, RAF, RW2, ORF, PEF, RAW, TIF or TIFF — or they could not be decoded.'

export function RejectedList({ rejected }: RejectedListProps) {
  if (rejected.length === 0) return null

  return (
    <div className="rejected">
      <div className="rejected__head">
        <span className="rejected__title">Rejected files</span>
        <Badge tone="bad">{rejected.length}</Badge>
      </div>

      <div className="rejected__note">
        <span aria-hidden>⚠</span>
        <span>{RAW_ONLY_MESSAGE}</span>
      </div>

      <div className="rejected__list">
        {rejected.map((r) => (
          <div key={r.path} className="rejected__item" title={r.path}>
            <span className="rejected__path">{r.path}</span>
            <span className="rejected__reason">{r.reason || 'Rejected by RAW validation.'}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
