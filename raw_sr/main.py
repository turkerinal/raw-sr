"""Application entry point for raw_sr."""

import sys
from PySide6.QtWidgets import QApplication
from raw_sr.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())