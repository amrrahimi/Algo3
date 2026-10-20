from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QLabel, QMainWindow, QMenu,
                               QMessageBox, QScrollArea, QSystemTrayIcon, QTabWidget, QToolBar)

from .. import __version__
from ..config import Settings, resource_path
from ..engine.robot import Robot
from ..storage.db import Database
from .chart import ChartGrid
from .dialogs import ExchangeDialog, RankDialog
from .panels import HistoryTable, LogView, PositionsTable, SetupPanel, SignalsTable
from .theme import DN, UP


class Bridge(QObject):
    sig_log = Signal(str)
    sig_chart = Signal(str, object)
    sig_signal = Signal(object)
    sig_positions = Signal(object)
    sig_trade = Signal(object)
    sig_account = Signal(object)
    sig_status = Signal(str)
    sig_error = Signal(str)
    sig_rank = Signal(object)
    sig_reconf_failed = Signal(str)
    sig_reconf_done = Signal()
    sig_tuning = Signal(object)


class Events:
    """Robot -> UI adapter (thread-safe through Qt queued signals)."""

    def __init__(self, b: Bridge):
        self.b = b

    def log(self, msg): self.b.sig_log.emit(str(msg))
    def chart(self, symbol, payload): self.b.sig_chart.emit(symbol, payload)
    def signal(self, sig): self.b.sig_signal.emit(sig)
    def positions(self, positions): self.b.sig_positions.emit(list(positions))
    def trade_event(self, ev): self.b.sig_trade.emit(ev)
    def account(self, info): self.b.sig_account.emit(info)
    def status(self, text): self.b.sig_status.emit(text)
    def tuning(self, info): self.b.sig_tuning.emit(info)


class Sounds:
    def __init__(self):
        self.fx = {}
        try:
            from PySide6.QtMultimedia import QSoundEffect
            for name in ("signal", "open", "close"):
                e = QSoundEffect()
                e.setSource(QUrl.fromLocalFile(resource_path(f"assets/{name}.wav")))
                e.setVolume(0.9)
                self.fx[name] = e
        except Exception:
            self.fx = {}

    def play(self, name):
        e = self.fx.get(name)
        if e is not None:
            e.play()
        else:
            QApplication.beep()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AlgoPilot {__version__}  ·  Crypto auto-trading robot")
        self.resize(1500, 920)
        self.icon = QIcon(resource_path("assets/icon.png"))
        self.setWindowIcon(self.icon)
        self.settings = Settings.load()
        self.db = Database()
        self.bridge = Bridge()
        self.events = Events(self.bridge)
        self.sounds = Sounds()
        self._quitting = False

        try:
            self.robot = Robot(self.settings, self.db, self.events)
        except Exception as e:
            QMessageBox.warning(self, "Live account", f"Could not connect live account:\n{e}\n\n"
                                                      f"Switching to the Demo account.")
            self.settings.account = "paper"
            self.robot = Robot(self.settings, self.db, self.events)

        # ---- central charts
        self.grid = ChartGrid()
        self.grid.configure(self.settings.chart_layout, self.settings.symbols)
        self.grid.symbolNeeded.connect(lambda s: self.robot._wake.set())
        self.setCentralWidget(self.grid)

        # ---- left dock: setup
        self.panel = SetupPanel(self.settings)
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        dock = QDockWidget("Robot setup", self)
        dock.setWidget(scroll)
        dock.setObjectName("setup")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # ---- bottom dock: tabs
        self.tabs = QTabWidget()
        self.pos_table = PositionsTable()
        self.sig_table = SignalsTable()
        self.hist = HistoryTable(self.db)
        self.logv = LogView()
        self.tabs.addTab(self.pos_table, "Open positions")
        self.tabs.addTab(self.sig_table, "Signals")
        self.tabs.addTab(self.hist, "History")
        self.tabs.addTab(self.logv, "Log")
        bdock = QDockWidget("Trading", self)
        bdock.setWidget(self.tabs)
        bdock.setObjectName("trading")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bdock)
        self.resizeDocks([bdock], [260], Qt.Orientation.Vertical)

        # ---- toolbar
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addWidget(QLabel(" Charts: "))
        self.layout_combo = QComboBox()
        for n in (1, 2, 3, 4, 6):
            self.layout_combo.addItem(f"{n} chart{'s' if n > 1 else ''}", n)
        self.layout_combo.setCurrentIndex(max(0, self.layout_combo.findData(self.settings.chart_layout)))
        self.layout_combo.currentIndexChanged.connect(self._layout_changed)
        tb.addWidget(self.layout_combo)
        tb.addSeparator()
        act_api = QAction("🔑 Exchange API", self)
        act_api.triggered.connect(self._exchange_dialog)
        tb.addAction(act_api)
        act_rank = QAction("🏆 Rank strategies", self)
        act_rank.triggered.connect(self._rank)
        tb.addAction(act_rank)
        tb.addSeparator()
        self.acct_lbl = QLabel("")
        self.acct_lbl.setStyleSheet("font-weight:bold; padding:0 10px;")
        tb.addWidget(self.acct_lbl)
        self.status_lbl = QLabel("Robot stopped (monitoring positions)")
        self.statusBar().addPermanentWidget(self.status_lbl)

        # ---- tray
        self.tray = QSystemTrayIcon(self.icon, self)
        menu = QMenu()
        menu.addAction("Show AlgoPilot", self._show)
        menu.addAction("Quit", self._quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r: self._show())
        self.tray.show()

        # ---- wiring
        b = self.bridge
        b.sig_log.connect(self.logv.add)
        b.sig_chart.connect(self.grid.update_symbol)
        b.sig_signal.connect(self._on_signal)
        b.sig_positions.connect(self.pos_table.update_positions)
        b.sig_trade.connect(self._on_trade)
        b.sig_account.connect(self._on_account)
        b.sig_status.connect(self.status_lbl.setText)
        b.sig_error.connect(lambda m: QMessageBox.warning(self, "AlgoPilot", m))
        b.sig_rank.connect(self._on_rank)
        b.sig_reconf_failed.connect(self._on_reconf_failed)
        b.sig_reconf_done.connect(lambda: self.status_lbl.setText(
            "Robot running" if self.robot.scanning else "Robot stopped (monitoring positions)"))
        self.panel.changed.connect(self._settings_changed)
        self.panel.startStop.connect(self._start_stop)
        self.panel.rankRequested.connect(self._rank)
        self.panel.retuneRequested.connect(lambda: self.robot.retune())
        b.sig_tuning.connect(lambda info: self.grid.update_symbol_tuning(info))
        self.pos_table.closeRequested.connect(self._close_position)
        self.sig_table.openRequested.connect(self._open_signal)
        self.hist.resetDemo.connect(self._reset_demo)

        self.logv.add("AlgoPilot ready. Pick style, coins and strategy, then press START ROBOT.")
        acct = "Demo" if self.settings.account == "paper" else \
            ("TESTNET " if self.settings.creds.testnet else "LIVE ") + self.settings.creds.exchange
        self.logv.add(f"Account: {acct}"
                      f" · data: {self.settings.data_exchange} · {self.settings.market_type}")
        self.robot.start_engine()

    # ------------------------------------------------------------ settings
    def _settings_changed(self):
        s = self.panel.collect()
        if s.account == "live" and not (s.creds.api_key and s.creds.secret):
            QMessageBox.information(self, "Live account", "Add your exchange API keys first (🔑 Exchange API).")
            self.panel._set_combo(self.panel.account, "paper")
            s.account = "paper"
        snapshot = s.copy()
        if self.robot.needs_rebuild(snapshot):
            # connecting to an exchange does network calls: never block the UI thread
            self.status_lbl.setText("Connecting…")

            def run():
                try:
                    self.robot.reconfigure(snapshot)
                    self.bridge.sig_reconf_done.emit()
                except Exception as e:  # noqa: BLE001
                    self.bridge.sig_reconf_failed.emit(str(e))
            threading.Thread(target=run, daemon=True).start()
        else:
            self.robot.reconfigure(snapshot)
        s.save()
        if getattr(self, "_last_tf", s.timeframe) != s.timeframe or \
                getattr(self, "_last_market", s.market_type) != s.market_type:
            self.grid.cache.clear()
            self.grid.charts = []
            self.grid.configure(s.chart_layout, s.symbols)
        elif set(self.grid.symbols) != set(s.symbols):
            self.grid.configure(s.chart_layout, s.symbols)
        self._last_tf, self._last_market = s.timeframe, s.market_type

    def _on_reconf_failed(self, msg):
        QMessageBox.warning(self, "Settings", f"Could not apply settings:\n{msg}\n\n"
                                              f"Kept the previous account ({'Demo' if self.robot.settings.account == 'paper' else 'LIVE'}).")
        self.panel._loading = True
        self.panel._set_combo(self.panel.account, self.robot.settings.account)
        self.panel._loading = False
        self.settings.account = self.robot.settings.account
        self.settings.save()
        self.status_lbl.setText("Robot running" if self.robot.scanning else "Robot stopped (monitoring positions)")

    def _layout_changed(self):
        self.settings.chart_layout = self.layout_combo.currentData()
        self.settings.save()
        self.grid.configure(self.settings.chart_layout, self.settings.symbols)

    def _exchange_dialog(self):
        dlg = ExchangeDialog(self.settings, self)
        if dlg.exec():
            self.settings.save()
            self._settings_changed()

    # ------------------------------------------------------------ robot
    def _start_stop(self):
        on = not self.robot.scanning
        rs = self.robot.settings
        if on and rs.account == "live" and rs.trade_mode == "auto":
            where = "TESTNET orders (fake money)" if rs.creds.testnet else "REAL orders"
            r = QMessageBox.question(self, "Start live auto-trading",
                                     f"The robot will place {where} on {rs.creds.exchange} "
                                     f"risking {rs.risk_pct}% per trade. Continue?")
            if r != QMessageBox.StandardButton.Yes:
                return
        if on and not self.settings.symbols:
            QMessageBox.information(self, "AlgoPilot", "Select at least one coin.")
            return
        self.robot.set_scanning(on)
        self.panel.set_running(on)
        self.logv.add("Robot started" if on else "Robot stopped")

    def _bg(self, fn, *args):
        def run():
            try:
                fn(*args)
            except Exception as e:
                self.bridge.sig_error.emit(str(e))
                self.bridge.sig_log.emit(f"Error: {e}")
        threading.Thread(target=run, daemon=True).start()

    def _open_signal(self, sig):
        r = QMessageBox.question(self, "Open trade",
                                 f"Open {sig.side.upper()} {sig.symbol} on the "
                                 f"{'Demo' if self.settings.account == 'paper' else 'LIVE'} account?\n\n"
                                 f"SL {sig.sl:.6g}   TP {' / '.join(f'{t:.6g}' for t in sig.tps)}")
        if r == QMessageBox.StandardButton.Yes:
            self._bg(self.robot.execute_signal, sig)

    def _close_position(self, pid):
        if QMessageBox.question(self, "Close position", "Close this position at market?") == \
                QMessageBox.StandardButton.Yes:
            self._bg(self.robot.close_position, pid)

    def _reset_demo(self):
        if QMessageBox.question(self, "Reset demo", "Reset the demo account balance and delete demo history?") \
                == QMessageBox.StandardButton.Yes:
            from ..brokers.paper import PaperBroker
            if isinstance(self.robot.broker, PaperBroker):
                self.robot.broker.reset(self.settings.paper_start_balance)
            else:
                self.db.clear_account("paper")
                self.db.set("paper_cash", self.settings.paper_start_balance)
            self.hist.refresh()
            self.pos_table.update_positions(self.robot.broker.open_positions())

    # ------------------------------------------------------------ events
    def _notify(self, title, msg, sound):
        if self.settings.sound:
            self.sounds.play(sound)
        if self.tray.isVisible():
            self.tray.showMessage(title, msg, QSystemTrayIcon.MessageIcon.Information, 6000)

    def _on_signal(self, sig):
        self.sig_table.add(sig)
        self.logv.add(f"SIGNAL {sig.side.upper()} {sig.symbol} {sig.timeframe} @ {sig.entry:.6g} "
                      f"SL {sig.sl:.6g} TP {sig.tps[0]:.6g}/{sig.tps[1]:.6g}/{sig.tps[2]:.6g} "
                      f"conf {sig.confidence:.0%}")
        self._notify(f"{sig.side.upper()} signal: {sig.symbol}",
                     f"Entry {sig.entry:.6g} · SL {sig.sl:.6g} · TP1 {sig.tps[0]:.6g} · {sig.confidence:.0%}",
                     "signal")
        if self.settings.trade_mode == "notify":
            self.tabs.setCurrentWidget(self.sig_table)

    def _on_trade(self, ev):
        p = ev["pos"]
        t = ev["type"]
        if t == "info":
            self.logv.add(f"{p.symbol}: {ev['reason']}")
            return
        if t == "opened":
            msg = f"OPENED {p.side.upper()} {p.symbol} {p.qty:.6g} @ {p.entry:.6g} ({ev['reason']})"
            self._notify("Position opened", msg, "open")
        else:
            msg = f"{ev['reason'].upper()} {p.symbol}: {ev['qty']:.6g} @ {ev['price']:.6g}  PnL {ev['pnl']:+.2f}"
            self._notify("Position closed" if t == "closed" else "Take profit hit", msg, "close")
            if t == "closed":
                self.hist.refresh()
        self.logv.add(msg)

    def _on_account(self, info):
        name = "DEMO" if info["account"] == "paper" else ("TESTNET" if info.get("testnet") else "LIVE")
        color = "#f6c343" if name != "LIVE" else UP
        self.acct_lbl.setText(f"{name}   Balance {info['balance']:,.2f}   Equity {info['equity']:,.2f} USDT")
        self.acct_lbl.setStyleSheet(f"font-weight:bold; padding:0 10px; color:{color};")

    # ------------------------------------------------------------ ranking
    def _rank(self):
        s = self.settings
        if not s.symbols:
            QMessageBox.information(self, "AlgoPilot", "Select at least one coin.")
            return
        more = f" (first 6 of {len(s.symbols)} coins)" if len(s.symbols) > 6 else ""
        self.logv.add(f"Ranking strategies for {s.mode} on {', '.join(s.symbols[:6])}{more}… (takes a moment)")
        self.status_lbl.setText("Backtesting strategies…")
        feed = self.robot.feed
        s = s.copy()

        def run():
            from ..core.backtest import rank_strategies
            dfs = {}
            for sym in s.symbols[:6]:
                try:
                    dfs[sym] = feed.fetch_ohlcv(sym, s.timeframe, 1000)
                except Exception as e:
                    self.bridge.sig_log.emit(f"{sym}: {e}")
            if not dfs:
                self.bridge.sig_error.emit("No data to backtest.")
                return
            self.bridge.sig_rank.emit(rank_strategies(dfs, s.mode, s.market_type, s.risk_pct,
                                                           s.leverage, s.min_confidence))
        self._bg(run)

    def _on_rank(self, results):
        self.status_lbl.setText("Backtest done")
        dlg = RankDialog(results, self)
        if dlg.exec() and dlg.chosen:
            self.panel.select_strategy(dlg.chosen)

    # ------------------------------------------------------------ window
    def _show(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _quit(self):
        self._quitting = True
        self.settings.save()
        self.robot.shutdown()
        self.tray.hide()
        QApplication.quit()

    def closeEvent(self, e):
        if not self._quitting and self.robot.scanning and self.tray.isVisible():
            r = QMessageBox.question(self, "AlgoPilot",
                                     "Keep the robot running in the background (system tray)?\n"
                                     "Choose No to quit completely.")
            if r == QMessageBox.StandardButton.Yes:
                self.hide()
                self.tray.showMessage("AlgoPilot", "Robot is still running in the tray.",
                                      QSystemTrayIcon.MessageIcon.Information, 4000)
                e.ignore()
                return
        self.settings.save()
        self.robot.shutdown()
        self.tray.hide()
        e.accept()
        QApplication.quit()
