"""Background job manager (threads) for SR processing.

Owned by AGENT-A. Baseline implementation by the orchestrator.

Cancellation model
------------------
A job carries a :class:`threading.Event`. Worker code polls
``job.cancel_requested`` between coarse stages (and per frame while loading).
Cancellation is therefore *cooperative* and *coarse*: a call already inside
``rawpy`` (decoding one RAW) or inside ``SuperResolution.run`` (registration /
drizzle) cannot be interrupted mid-call — it is only observed once that call
returns. Jobs whose target returns early while ``cancel_requested`` is set are
marked ``cancelled``.

Memory model
------------
A finished 'process' job owns the heavy arrays in :data:`HEAVY_RESULT_KEYS`
(full-res HR float32 + coverage map + full-res reference RGB — hundreds of MB
per job). Retaining them for every job leaks for the backend's whole lifetime,
so the manager keeps them for only the ``keep_heavy_results`` most recent
COMPLETED jobs (LRU, default 1). Evicted jobs keep everything else (report,
display arrays, alignment) so ``GET /api/jobs/{id}/result`` still renders;
only the export path — which needs ``hr`` — must notice the eviction.
Cancelled/errored jobs never serve a result, so their heavy arrays are
dropped immediately.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

TERMINAL_STATES = ('done', 'error', 'cancelled')

#: Per-job ``data`` keys that dominate backend memory. See the class docstring.
HEAVY_RESULT_KEYS = ('hr', 'coverage', 'before_rgb')


def _memory_error_message(exc: MemoryError) -> str:
    """Turn the core's HR-size ``MemoryError`` into an actionable sentence.

    The core message already names the output size, the ratio, the estimate and
    the limit (``SuperResolution._check_hr_size``); keep it verbatim and only
    add the "this is fixable" lead-in when the hint is missing, so a client
    never has to parse a traceback to learn what to change.
    """
    detail = str(exc).strip() or 'the HR output does not fit in memory'
    low = detail.lower()
    prefix = '' if low.startswith('not enough memory') else 'Not enough memory: '
    hint = ('' if 'reduce' in low
            else ' Reduce the SR ratio or the ROI, then try again.')
    return prefix + detail + hint


class Job:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.state: str = 'queued'
        self.progress: int = 0
        self.message: str = 'Queued'
        self.error: Optional[str] = None
        #: Additive classification of ``error``: 'memory' for the HR-size
        #: guard, None otherwise. Lets a client word the failure precisely.
        self.error_kind: Optional[str] = None
        self.data: Dict[str, Any] = {}
        self.created: float = time.time()
        self.updated: float = time.time()
        self._cancel = threading.Event()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def request_cancel(self) -> None:
        self._cancel.set()


class JobManager:
    """Runs blocking SR work on daemon threads and tracks status.

    ``keep_heavy_results`` bounds how many completed jobs retain their heavy
    result arrays (see :data:`HEAVY_RESULT_KEYS`); older ones are evicted down
    to a renderable lightweight record. ``create(on_finish=...)`` additionally
    lets the caller hang a once-only cleanup hook (used by the process
    endpoint's single-flight lock) that runs on *every* terminal path —
    including cancel-before-start, where the target never executes.
    """

    def __init__(self, keep_heavy_results: int = 1) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep_heavy = max(0, int(keep_heavy_results))
        # Completed jobs still holding heavy arrays, oldest first (LRU).
        self._heavy: 'OrderedDict[str, Job]' = OrderedDict()

    def create(self, target: Callable[[Job], None],
               on_finish: Optional[Callable[[], None]] = None) -> Job:
        job = Job(uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job, target, on_finish), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, target: Callable[[Job], None],
             on_finish: Optional[Callable[[], None]] = None) -> None:
        if job.cancel_requested:
            self._finish(job, on_finish, 'cancelled', message='Cancelled.',
                         progress=job.progress)
            return
        job.state = 'running'
        job.updated = time.time()
        try:
            target(job)
            if job.cancel_requested:
                self._finish(job, on_finish, 'cancelled', message='Cancelled.',
                             progress=job.progress)
            else:
                self._finish(job, on_finish, 'done',
                             message=job.message or 'Done.', progress=100)
        except MemoryError as exc:
            # The core's HR-size guard refused this ROI/ratio. That is a user
            # mistake with a fix, not a crash: say what to change.
            self._finish(job, on_finish, 'error',
                         error=_memory_error_message(exc), error_kind='memory',
                         message='Not enough memory.', progress=job.progress)
        except Exception as exc:  # noqa: BLE001
            self._finish(job, on_finish, 'error',
                         error=f'{type(exc).__name__}: {exc}',
                         message='Error.', progress=job.progress)

    def _finish(self, job: Job, on_finish: Optional[Callable[[], None]],
                state: str, *, message: str, progress: int,
                error: Optional[str] = None,
                error_kind: Optional[str] = None) -> None:
        """Publish a terminal state exactly once.

        Housekeeping (heavy-array retention) and the ``on_finish`` hook run
        BEFORE the terminal state becomes visible, so a poller that observes
        completion can immediately start the next job — a caller using
        ``on_finish`` to release a single-flight guard (the process endpoint)
        never races a client that reacts to ``done`` with a new request.
        """
        if state == 'done':
            self._retain_heavy(job)
        else:
            # Cancelled/errored jobs never serve a result: release any heavy
            # arrays the target may have stored before bailing out.
            self._drop_heavy(job)
        if on_finish is not None:
            try:
                on_finish()
            except Exception:  # noqa: BLE001 - the hook must never mask the job outcome
                pass
        job.state = state
        job.progress = progress
        job.message = message
        job.error = error
        job.error_kind = error_kind
        job.updated = time.time()

    # ------------------------------------------------------------------ #
    # Heavy-result retention (LRU over completed process jobs)
    # ------------------------------------------------------------------ #

    def _retain_heavy(self, job: Job) -> None:
        """Register a completed job's heavy arrays, evicting all but the newest K.

        Jobs without result data (e.g. plain worker targets) occupy no slot, so
        they can never push a real processed result out of the window.
        """
        if not any(k in job.data for k in HEAVY_RESULT_KEYS):
            return
        with self._lock:
            self._heavy[job.id] = job
            self._heavy.move_to_end(job.id)
            evicted = []
            while len(self._heavy) > self._keep_heavy:
                evicted.append(self._heavy.popitem(last=False)[1])
        for old in evicted:
            self._drop_heavy(old)

    def _drop_heavy(self, job: Job) -> None:
        """Release the heavy arrays of one job (report/preview data is kept)."""
        with self._lock:
            self._heavy.pop(job.id, None)
        for key in HEAVY_RESULT_KEYS:
            job.data.pop(key, None)

    def held_heavy_ids(self) -> List[str]:
        """Ids of completed jobs still holding heavy arrays, oldest first."""
        with self._lock:
            return list(self._heavy)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        if not job.is_terminal:
            job.request_cancel()
            job.message = 'Cancelling…'
            job.updated = time.time()
        return True
