"""AlgoPilot entry point."""
import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def main():
    from PySide6.QtWidgets import QApplication
    from algopilot.ui.theme import STYLE
    from algopilot.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AlgoPilot")
    app.setQuitOnLastWindowClosed(False)   # robot can keep running in the tray
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
