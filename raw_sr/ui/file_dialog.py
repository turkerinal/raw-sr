"""
File dialog for RAW-only selection (v0.1.0).

Filters file dialogs to only accept RAW formats.
"""

from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QDir


class RawFileDialog:
    """File dialog wrapper that only accepts RAW image formats."""

    RAW_FILTERS = [
        "RAW Images",
        "*.cr2;*.nef;*.arw;*.dng;*.orf;*.rw2;*.raf;*.k25;*.g2;*.g3;*.g6;*.erf;*.mrw;*.ob2;*.pef;*.sr2;*.srf;*.x3f",
    ]

    @staticmethod
    def select_raw_files(parent=None):
        """
        Open a file dialog to select multiple RAW files.

        Returns:
            List of file paths, or empty list if cancelled.
        """
        filter_str = (
            "RAW Images (*.cr2 *.nef *.arw *.dng *.orf *.rw2 *.raf *.k25 *.g2 *.g3 *.g6 *.erf *.mrw *.ob2 *.pef *.sr2 *.srf *.x3f);;"
            "All Files (*)"
        )

        files, _ = QFileDialog.getOpenFileNames(
            parent,
            "Select RAW Files",
            QDir.homePath(),
            filter_str,
        )

        return list(files)

    @staticmethod
    def select_output_path(parent=None, default_name="output.tiff"):
        """
        Open a save dialog for the output TIFF.

        Returns:
            File path string, or None if cancelled.
        """
        path, _ = QFileDialog.getSaveFileName(
            parent,
            "Save Output TIFF",
            default_name,
            "TIFF Files (*.tiff *.tif);;All Files (*)",
        )

        return path if path else None
