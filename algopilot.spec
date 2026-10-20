# PyInstaller spec: one build file for Windows (.exe) and macOS (.app)
import sys

hidden = []  # ccxt and sklearn ship their own PyInstaller hooks; collecting every submodule made the build very slow
a = Analysis(["main.py"], pathex=["."],
             datas=[("assets", "assets")],
             hiddenimports=hidden + ["PySide6.QtMultimedia", "keyring.backends.Windows", "keyring.backends.macOS"],
             excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AlgoPilot", console=False,
          icon="assets/icon.ico" if sys.platform == "win32" else None)
coll = COLLECT(exe, a.binaries, a.datas, name="AlgoPilot")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="AlgoPilot.app", icon="assets/icon.icns",
                 bundle_identifier="com.algopilot.app",
                 info_plist={"NSHighResolutionCapable": True, "LSUIElement": False})
