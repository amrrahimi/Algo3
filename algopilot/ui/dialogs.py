from __future__ import annotations

import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QAbstractItemView, QHeaderView)

from ..config import EXCHANGES


class ExchangeDialog(QDialog):
    _tested = Signal(bool, str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exchange API connection")
        self.setMinimumWidth(460)
        self.s = settings
        c = settings.creds
        lay = QVBoxLayout(self)
        warn = QLabel("⚠️ Create an API key with <b>trading permission only</b>. "
                      "<b>Never enable withdrawals.</b> Restrict the key to your IP if the exchange allows it. "
                      "Keys are stored only on this computer.")
        warn.setWordWrap(True)
        lay.addWidget(warn)
        f = QFormLayout()
        self.ex = QComboBox()
        for e in EXCHANGES:
            self.ex.addItem(e.capitalize(), e)
        self.ex.setCurrentIndex(max(0, self.ex.findData(c.exchange)))
        self.key = QLineEdit(c.api_key)
        self.secret = QLineEdit(c.secret)
        self.secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.pw = QLineEdit(c.password)
        self.pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.pw.setPlaceholderText("KuCoin only")
        self.testnet = QCheckBox("Test account: Binance Demo Trading / Bybit testnet (fake money)")
        self.testnet.setToolTip("Binance: create keys at demo.binance.com (API Management).\n"
                                "Bybit: create keys at testnet.bybit.com.\n"
                                "Test keys only work with this box ticked; real keys only without it.")
        self.testnet.setChecked(c.testnet)
        self.ex_sl = QCheckBox("Place stop-loss on the exchange too (futures, recommended)")
        self.ex_sl.setChecked(settings.exchange_side_sl)
        f.addRow("Exchange", self.ex)
        f.addRow("API key", self.key)
        f.addRow("Secret", self.secret)
        f.addRow("Passphrase", self.pw)
        f.addRow(self.testnet)
        f.addRow(self.ex_sl)
        lay.addLayout(f)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        test = QPushButton("Test connection")
        test.clicked.connect(self._test)
        lay.addWidget(test)
        lay.addWidget(self.result)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._tested.connect(self._on_tested)

    def _creds(self):
        from ..config import ExchangeCreds
        return ExchangeCreds(self.ex.currentData(), self.key.text().strip(), self.secret.text().strip(),
                             self.pw.text().strip(), self.testnet.isChecked())

    def _test(self):
        creds = self._creds()
        market = self.s.market_type
        self.result.setText("Testing…")

        def run():
            try:
                from ..data.feed import make_exchange
                ex = make_exchange(creds.exchange, market, creds.api_key, creds.secret, creds.password,
                                   creds.testnet)
                ex.load_markets()
                b = ex.fetch_balance()
                usdt = (b.get("USDT") or {})
                env = "TESTNET / demo" if creds.testnet else "MAINNET (real money)"
                self._tested.emit(True, f"✅ Connected to {creds.exchange} {env} ({market}). "
                                        f"USDT free: {usdt.get('free')}  total: {usdt.get('total')}")
            except Exception as e:
                self._tested.emit(False, f"❌ {e}")
        threading.Thread(target=run, daemon=True).start()

    def _on_tested(self, ok, msg):
        self.result.setText(msg)

    def _save(self):
        self.s.creds = self._creds()
        self.s.exchange_side_sl = self.ex_sl.isChecked()
        self.accept()


class RankDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Strategy ranking (backtest on your coins)")
        self.resize(820, 380)
        self.chosen = None
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Backtest on the latest ~1000 candles of each selected coin, with fees, "
                             "slippage, partial TPs and breakeven stops. Past results don't guarantee "
                             "future profit; confirm on the Demo account first."))
        t = QTableWidget(len(results), 7)
        t.setHorizontalHeaderLabels(["Strategy", "Trades", "Win rate", "Profit factor", "Avg net %",
                                     "Max DD %", "Score"])
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for r, d in enumerate(results):
            vals = [d["name"], d["trades"], f"{d['win_rate']:.0%}", f"{d['pf']:.2f}", f"{d['net_pct']:+.2f}",
                    f"{d['max_dd']:.1f}", f"{d['score']:.2f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setData(Qt.ItemDataRole.UserRole, d["key"])
                t.setItem(r, c, it)
        if results:
            t.selectRow(0)
        self.t = t
        lay.addWidget(t)
        bb = QDialogButtonBox()
        use = bb.addButton("Use selected strategy", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QDialogButtonBox.StandardButton.Close)
        use.clicked.connect(self._use)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _use(self):
        it = self.t.item(self.t.currentRow(), 0)
        if it:
            self.chosen = it.data(Qt.ItemDataRole.UserRole)
        self.accept()
