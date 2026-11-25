"""Locate the optional real RAW burst used by the integration tests.

The burst is large binary data and is intentionally not committed. Tests must
find it through ``RAW_SR_TEST_DATA`` or a repo-relative ``test_data/``
directory and skip cleanly when neither exists, so a fresh clone still has a
green suite.
"""
from __future__ import annotations

import glob
import os


def burst_dir() -> str | None:
    """Return the first directory that holds a real RAW burst, or None."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.environ.get("RAW_SR_TEST_DATA"),
        os.path.join(here, "test_data"),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def dng_files(min_frames: int = 1) -> list[str]:
    """Sorted DNG paths from the burst, or [] when it is absent/too small."""
    directory = burst_dir()
    if not directory:
        return []
    files = sorted(glob.glob(os.path.join(directory, "*.dng")))
    return files if len(files) >= min_frames else []


REAL_DATA_HELP = "no real DNG burst: set RAW_SR_TEST_DATA or add test_data/*.dng"


def missing_reason(minimum: int = 1) -> str:
    """Skip reason naming what was looked for (kept for the ghost-gate tests)."""
    return (f"needs at least {minimum} DNG frames: set RAW_SR_TEST_DATA or add "
            f"test_data/*.dng")
