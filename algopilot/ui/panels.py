"""Left setup panel and bottom tables (positions, signals, history, log)."""
from __future__ import annotations

import csv
import time
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..config import DATA_SOURCES, DEFAULT_COINS, MODES
from ..core.strategies import STRATEGIES, strategies_for_mode
from .theme import DN, UP


def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""


class SetupPanel(QWidget):
    changed = Signal()
    startStop = Signal()
    rankRequested = Signal()
    retuneRequested = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self._loading = True
        root = QVBoxLayout(self)

        # ---- 1. trading style
        g1 = QGroupBox("1. Trading style")
        f1 = QFormLayout(g1)
        self.mode = QComboBox()
        for k, v in MODES.items():
            self.mode.addItem(v["label"], k)
        self.tf = QComboBox()
        self.market = QComboBox()
        self.market.addItem("Futures (long + short, leverage)", "futures")
        self.market.addItem("Spot (long only)", "spot")
        self.lev = QSpinBox()
        self.lev.setRange(1, 125)
        self.lev.setSuffix("x")
        f1.addRow("Mode", self.mode)
        f1.addRow("Timeframe", self.tf)
        f1.addRow("Market", self.market)
        f1.addRow("Leverage", self.lev)
        root.addWidget(g1)

        # ---- 2. coins
        g2 = QGroupBox("2. Coins")
        v2 = QVBoxLayout(g2)
        self.coins = QListWidget()
        self.coins.setMinimumHeight(150)
        add = QHBoxLayout()
        self.coin_edit = QLineEdit()
        self.coin_edit.setPlaceholderText("e.g. PEPE/USDT")
        btn = QPushButton("Add")
        btn.clicked.connect(self._add_coin)
        add.addWidget(self.coin_edit)
        add.addWidget(btn)
        v2.addWidget(self.coins)
        v2.addLayout(add)
        root.addWidget(g2)

        # ---- 3. strategy
        g3 = QGroupBox("3. Strategy")
        v3 = QVBoxLayout(g3)
        self.strat = QListWidget()
        self.strat.setMinimumHeight(140)
        self.strat_desc = QLabel()
        self.strat_desc.setWordWrap(True)
        self.strat_desc.setStyleSheet("color:#8f96a3;")
        rank = QPushButton("🏆 Rank strategies on my coins (backtest)")
        rank.clicked.connect(self.rankRequested.emit)
        v3.addWidget(self.strat)
        v3.addWidget(self.strat_desc)
        v3.addWidget(rank)
        root.addWidget(g3)

        # ---- 4. robot + risk
        g4 = QGroupBox("4. Robot & risk")
        f4 = QFormLayout(g4)
        self.trade_mode = QComboBox()
        self.trade_mode.addItem("Semi-auto: notify me, I open trades", "notify")
        self.trade_mode.addItem("Full auto: robot trades 24/7", "auto")
        self.account = QComboBox()
        self.account.addItem("Demo account (virtual money)", "paper")
        self.account.addItem("Live exchange (API keys)", "live")
        self.data_src = QComboBox()
        for d in DATA_SOURCES:
            self.data_src.addItem("Demo data (offline)" if d == "demo" else d.capitalize(), d)
        self.risk = QDoubleSpinBox()
        self.risk.setRange(0.1, 10)
        self.risk.setSingleStep(0.1)
        self.risk.setSuffix(" % per trade")
        self.maxpos = QSpinBox()
        self.maxpos.setRange(1, 50)
        self.dloss = QDoubleSpinBox()
        self.dloss.setRange(0.5, 50)
        self.dloss.setSuffix(" %")
        self.minconf = QSpinBox()
        self.minconf.setRange(0, 100)
        self.minconf.setSuffix(" %")
        self.use_ml = QCheckBox("Machine-learning filter")
        self.auto_tune = QCheckBox("Auto-tune settings on each coin's chart")
        self.freq = QComboBox()
        self.freq.addItem("Quality: fewer, stronger signals", "quality")
        self.freq.addItem("Balanced", "balanced")
        self.freq.addItem("More trades: looser filter", "high")
        self.sound = QCheckBox("Sound notifications")
        self.start_bal = QDoubleSpinBox()
        self.start_bal.setRange(100, 10_000_000)
        self.start_bal.setDecimals(0)
        self.start_bal.setPrefix("$")
        self.poll = QSpinBox()
        self.poll.setRange(3, 300)
        self.poll.setSuffix(" s")
        f4.addRow("Robot", self.trade_mode)
        f4.addRow("Account", self.account)
        f4.addRow("Chart data", self.data_src)
        f4.addRow("Risk", self.risk)
        f4.addRow("Max open trades", self.maxpos)
        f4.addRow("Max daily loss", self.dloss)
        f4.addRow("Min confidence", self.minconf)
        f4.addRow("Demo balance", self.start_bal)
        f4.addRow("Refresh every", self.poll)
        f4.addRow("Signal frequency", self.freq)
        f4.addRow(self.use_ml)
        f4.addRow(self.auto_tune)
        f4.addRow(self.sound)
        root.addWidget(g4)

        # ---- 5. chart overlays
        g5 = QGroupBox("5. Chart overlays")
        f5 = QFormLayout(g5)
        self.show_smc = QCheckBox("Smart Money Concepts (BOS/CHoCH, OB, FVG)")
        self.show_lux = QCheckBox("Signals & Overlays (arrows, Smart Trail, zones)")
        self.show_sp2l = QCheckBox("SP2L spikes and entry levels")
        self.show_emas = QCheckBox("EMA 20 / 50 / 200")
        self.osc = QComboBox()
        self.osc.addItem("Squeeze Momentum [LazyBear]", "squeeze")
        self.osc.addItem("RSI", "rsi")
        self.retune = QPushButton("Re-tune indicators now")
        self.retune.clicked.connect(self.retuneRequested.emit)
        f5.addRow(self.show_smc)
        f5.addRow(self.show_lux)
        f5.addRow(self.show_sp2l)
        f5.addRow(self.show_emas)
        f5.addRow("Bottom pane", self.osc)
        f5.addRow(self.retune)
        root.addWidget(g5)

        self.start_btn = QPushButton("▶  START ROBOT")
        self.start_btn.setObjectName("start")
        self.start_btn.clicked.connect(self.startStop.emit)
        root.addWidget(self.start_btn)
        root.addStretch(1)

        self._populate()
        for w in (self.tf, self.market, self.trade_mode, self.account, self.data_src):
            w.currentIndexChanged.connect(self._emit)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        for w in (self.lev, self.maxpos, self.minconf, self.poll):
            w.valueChanged.connect(self._emit)
        for w in (self.risk, self.dloss, self.start_bal):
            w.valueChanged.connect(self._emit)
        self.use_ml.toggled.connect(self._emit)
        self.auto_tune.toggled.connect(self._emit)
        self.sound.toggled.connect(self._emit)
        self.freq.currentIndexChanged.connect(self._emit)
        for w in (self.show_smc, self.show_lux, self.show_sp2l, self.show_emas):
            w.toggled.connect(self._emit)
        self.osc.currentIndexChanged.connect(self._emit)
        self.coins.itemChanged.connect(self._emit)
        self.strat.currentRowChanged.connect(self._strat_changed)
        self._loading = False

    # ------------------------------------------------------------------
    def _set_combo(self, combo, data):
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _populate(self):
        s = self.s
        self._set_combo(self.mode, s.mode)
        self._fill_tf()
        self._set_combo(self.market, s.market_type)
        self.lev.setValue(int(s.leverage))
        coins = list(dict.fromkeys(DEFAULT_COINS + list(s.symbols)))
        for c in coins:
            it = QListWidgetItem(c)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if c in s.symbols else Qt.CheckState.Unchecked)
            self.coins.addItem(it)
        self._fill_strategies()
        self._set_combo(self.trade_mode, s.trade_mode)
        self._set_combo(self.account, s.account)
        self._set_combo(self.data_src, s.data_exchange)
        self.risk.setValue(s.risk_pct)
        self.maxpos.setValue(s.max_positions)
        self.dloss.setValue(s.max_daily_loss_pct)
        self.minconf.setValue(int(s.min_confidence * 100))
        self.use_ml.setChecked(s.use_ml)
        self.auto_tune.setChecked(s.auto_tune)
        self._set_combo(self.freq, s.signal_frequency)
        self.show_smc.setChecked(s.show_smc)
        self.show_lux.setChecked(s.show_lux)
        self.show_sp2l.setChecked(s.show_sp2l)
        self.show_emas.setChecked(s.show_emas)
        self._set_combo(self.osc, s.osc_pane)
        sound_default = s.sound
        self.sound.setChecked(sound_default)
        self.start_bal.setValue(s.paper_start_balance)
        self.poll.setValue(s.poll_seconds)

    def _fill_tf(self):
        self.tf.blockSignals(True)
        self.tf.clear()
        mode = self.mode.currentData()
        for t in MODES[mode]["timeframes"]:
            self.tf.addItem(t, t)
        cur = self.s.timeframe if self.s.timeframe in MODES[mode]["timeframes"] else MODES[mode]["default"]
        self._set_combo(self.tf, cur)
        self.tf.blockSignals(False)

    def _fill_strategies(self):
        self.strat.blockSignals(True)
        self.strat.clear()
        avail = strategies_for_mode(self.mode.currentData())
        for st in avail:
            it = QListWidgetItem(("⭐ " if st.key == "ai_confluence" else "") + st.name)
            it.setData(Qt.ItemDataRole.UserRole, st.key)
            it.setToolTip(st.description)
            self.strat.addItem(it)
        keys = [st.key for st in avail]
        row = keys.index(self.s.strategy) if self.s.strategy in keys else 0
        self.strat.setCurrentRow(row)
        self.strat_desc.setText(avail[row].description)
        self.strat.blockSignals(False)

    def _mode_changed(self):
        self._fill_tf()
        self._fill_strategies()
        self._emit()

    def _strat_changed(self, row):
        it = self.strat.item(row)
        if it:
            self.strat_desc.setText(STRATEGIES[it.data(Qt.ItemDataRole.UserRole)].description)
        self._emit()

    def _add_coin(self):
        sym = self.coin_edit.text().strip().upper()
        if not sym:
            return
        if "/" not in sym:
            sym += "/USDT"
        if not self.coins.findItems(sym, Qt.MatchFlag.MatchExactly):
            it = QListWidgetItem(sym)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.coins.insertItem(0, it)
        self.coin_edit.clear()
        self._emit()

    def select_strategy(self, key):
        for i in range(self.strat.count()):
            if self.strat.item(i).data(Qt.ItemDataRole.UserRole) == key:
                self.strat.setCurrentRow(i)
                return

    def _emit(self, *args):
        if self._loading:
            return
        self.collect()
        self.changed.emit()

    def collect(self):
        s = self.s
        s.mode = self.mode.currentData()
        s.timeframe = self.tf.currentData() or MODES[s.mode]["default"]
        s.market_type = self.market.currentData()
        s.leverage = self.lev.value()
        s.symbols = [self.coins.item(i).text() for i in range(self.coins.count())
                     if self.coins.item(i).checkState() == Qt.CheckState.Checked]
        it = self.strat.currentItem()
        if it:
            s.strategy = it.data(Qt.ItemDataRole.UserRole)
        s.trade_mode = self.trade_mode.currentData()
        s.account = self.account.currentData()
        s.data_exchange = self.data_src.currentData()
        s.risk_pct = self.risk.value()
        s.max_positions = self.maxpos.value()
        s.max_daily_loss_pct = self.dloss.value()
        s.min_confidence = self.minconf.value() / 100
        s.use_ml = self.use_ml.isChecked()
        s.auto_tune = self.auto_tune.isChecked()
        s.signal_frequency = self.freq.currentData() or "balanced"
        s.show_smc = self.show_smc.isChecked()
        s.show_lux = self.show_lux.isChecked()
        s.show_sp2l = self.show_sp2l.isChecked()
        s.show_emas = self.show_emas.isChecked()
        s.osc_pane = self.osc.currentData() or "squeeze"
        s.sound = self.sound.isChecked()
        s.paper_start_balance = self.start_bal.value()
        s.poll_seconds = self.poll.value()
        self.lev.setEnabled(s.market_type == "futures")
        return s

    def set_running(self, on: bool):
        self.start_btn.setText("■  STOP ROBOT" if on else "▶  START ROBOT")
        self.start_btn.setObjectName("stop" if on else "start")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)


def _table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _item(text, color=None):
    it = QTableWidgetItem(str(text))
    if color:
        it.setForeground(QColor(color))
    return it


class PositionsTable(QWidget):
    closeRequested = Signal(str)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.t = _table(["Symbol", "Side", "Account", "Qty", "Entry", "Price", "SL", "TP1", "TP2", "TP3",
                         "TPs hit", "Lev", "uPnL", ""])
        lay.addWidget(self.t)

    def update_positions(self, positions):
        self.t.setRowCount(len(positions))
        for r, p in enumerate(positions):
            pnl = p.upnl()
            vals = [p.symbol, p.side.upper(), "Demo" if p.account == "paper" else p.exchange, f"{p.remaining:.6g}",
                    f"{p.entry:.6g}", f"{(p.last_price or p.entry):.6g}", f"{p.sl:.6g}"] + \
                   [f"{tp:.6g}" for tp in (p.tps + [0, 0, 0])[:3]] + [str(p.tp_hit), f"{p.leverage}x", f"{pnl:+.2f}"]
            for c, v in enumerate(vals):
                color = (UP if p.side == "long" else DN) if c == 1 else (UP if pnl >= 0 else DN) if c == 12 else None
                self.t.setItem(r, c, _item(v, color))
            b = QPushButton("Close")
            b.clicked.connect(lambda _=False, pid=p.id: self.closeRequested.emit(pid))
            self.t.setCellWidget(r, 13, b)


class SignalsTable(QWidget):
    openRequested = Signal(object)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.t = _table(["Time", "Symbol", "TF", "Side", "Entry", "SL", "TP1 / TP2 / TP3", "R:R",
                         "Confidence", "Strategy", "Why", ""])
        lay.addWidget(self.t)
        self.signals = []

    def add(self, sig):
        self.signals.insert(0, sig)
        self.signals = self.signals[:200]
        self.t.insertRow(0)
        vals = [fmt_time(sig.created), sig.symbol, sig.timeframe, sig.side.upper(), f"{sig.entry:.6g}",
                f"{sig.sl:.6g}", " / ".join(f"{x:.6g}" for x in sig.tps), f"{sig.risk_reward:.1f}",
                f"{sig.confidence:.0%}", sig.strategy, "; ".join(sig.reasons)]
        for c, v in enumerate(vals):
            self.t.setItem(0, c, _item(v, (UP if sig.side == "long" else DN) if c == 3 else None))
        b = QPushButton("Open trade")
        b.clicked.connect(lambda _=False, s=sig: self.openRequested.emit(s))
        self.t.setCellWidget(0, 11, b)
        if self.t.rowCount() > 200:
            self.t.removeRow(200)


class HistoryTable(QWidget):
    resetDemo = Signal()

    def __init__(self, db):
        super().__init__()
        self.db = db
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItem("All accounts", None)
        self.filter.addItem("Demo", "paper")
        self.filter.addItem("Live", "live")
        self.filter.currentIndexChanged.connect(self.refresh)
        self.stats = QLabel()
        exp = QPushButton("Export CSV")
        exp.clicked.connect(self.export)
        rst = QPushButton("Reset demo account")
        rst.clicked.connect(self.resetDemo.emit)
        top.addWidget(self.filter)
        top.addWidget(self.stats, 1)
        top.addWidget(exp)
        top.addWidget(rst)
        lay.addLayout(top)
        self.t = _table(["Opened", "Closed", "Account", "Symbol", "Side", "Entry", "Exit", "Qty", "PnL",
                         "Fees", "TPs hit", "Exit reason", "Strategy", "TF"])
        lay.addWidget(self.t)
        self.rows = []
        self.refresh()

    def refresh(self, *args):
        self.rows = self.db.history(self.filter.currentData())
        self.t.setRowCount(len(self.rows))
        for r, h in enumerate(self.rows):
            pnl = h["pnl"] or 0
            vals = [fmt_time(h["opened_at"]), fmt_time(h["closed_at"]),
                    "Demo" if h["account"] == "paper" else h["exchange"], h["symbol"], h["side"].upper(),
                    f"{h['entry']:.6g}", f"{(h['exit_price'] or 0):.6g}", f"{h['qty']:.6g}", f"{pnl:+.2f}",
                    f"{(h['fees'] or 0):.2f}", h["tp_hit"], h["close_reason"], h["strategy"], h["timeframe"]]
            for c, v in enumerate(vals):
                self.t.setItem(r, c, _item(v, (UP if pnl >= 0 else DN) if c == 8 else None))
        pnls = [h["pnl"] or 0 for h in self.rows]
        if pnls:
            wins = [p for p in pnls if p > 0]
            losses = [-p for p in pnls if p < 0]
            pf = sum(wins) / sum(losses) if losses else float("inf")
            self.stats.setText(f"  Trades: {len(pnls)}   Win rate: {len(wins) / len(pnls):.0%}   "
                               f"Net PnL: {sum(pnls):+.2f}   Profit factor: {pf:.2f}")
        else:
            self.stats.setText("  No closed trades yet")

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export trade history",
                                              f"algopilot_history_{int(time.time())}.csv", "CSV (*.csv)")
        if not path or not self.rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


class LogView(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(3000)

    def add(self, msg):
        self.appendPlainText(f"[{datetime.now():%H:%M:%S}] {msg}")
