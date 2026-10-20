UP = "#26a69a"
DN = "#ef5350"
BG = "#131722"
PANEL = "#1e222d"
TEXT = "#d1d4dc"
ACCENT = "#2962ff"

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 12px; }}
QMainWindow::separator {{ background: #2a2e39; width: 4px; height: 4px; }}
QDockWidget::title {{ background: {PANEL}; padding: 6px; }}
QGroupBox {{ border: 1px solid #2a2e39; border-radius: 6px; margin-top: 14px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8f96a3; }}
QPushButton {{ background: {PANEL}; border: 1px solid #363a45; border-radius: 4px; padding: 5px 10px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton#start {{ background: {UP}; color: white; font-weight: bold; font-size: 14px; padding: 10px; }}
QPushButton#stop {{ background: {DN}; color: white; font-weight: bold; font-size: 14px; padding: 10px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {PANEL}; border: 1px solid #363a45;
    border-radius: 4px; padding: 3px 6px; }}
QListWidget, QTableWidget, QPlainTextEdit {{ background: {PANEL}; border: 1px solid #2a2e39;
    gridline-color: #2a2e39; }}
QHeaderView::section {{ background: {BG}; color: #8f96a3; border: none; padding: 4px; }}
QTabBar::tab {{ background: {BG}; padding: 6px 14px; color: #8f96a3; }}
QTabBar::tab:selected {{ color: white; border-bottom: 2px solid {ACCENT}; }}
QSplitter::handle {{ background: #2a2e39; }}
QToolBar {{ background: {PANEL}; border: none; spacing: 6px; }}
"""
