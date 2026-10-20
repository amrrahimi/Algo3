"""SQLite storage for positions, trade history, signals and app state."""
from __future__ import annotations

import json
import os
import sqlite3
import threading

from ..config import app_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY, account TEXT, exchange TEXT, symbol TEXT, side TEXT, market TEXT,
    qty REAL, remaining REAL, entry REAL, sl REAL, tps TEXT, tp_hit INTEGER, leverage INTEGER,
    opened_at REAL, closed_at REAL, exit_price REAL, pnl REAL, fees REAL, status TEXT,
    strategy TEXT, timeframe TEXT, close_reason TEXT, liq REAL, extra TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY, created REAL, symbol TEXT, timeframe TEXT, strategy TEXT, side TEXT,
    entry REAL, sl REAL, tps TEXT, confidence REAL, reasons TEXT, acted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

POS_COLS = ["id", "account", "exchange", "symbol", "side", "market", "qty", "remaining", "entry",
            "sl", "tps", "tp_hit", "leverage", "opened_at", "closed_at", "exit_price", "pnl", "fees",
            "status", "strategy", "timeframe", "close_reason", "liq", "extra"]


class Database:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(app_dir(), "algopilot.db")
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------------- kv
    def get(self, key, default=None):
        with self._lock:
            r = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set(self, key, value):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))
            self._conn.commit()

    # ---------------- positions
    def save_position(self, pos):
        d = pos.to_row()
        with self._lock:
            self._conn.execute(f"INSERT OR REPLACE INTO positions ({','.join(POS_COLS)}) "
                               f"VALUES ({','.join('?' * len(POS_COLS))})", [d[c] for c in POS_COLS])
            self._conn.commit()

    def open_positions(self, account: str, exchange: str | None = None):
        q, args = "SELECT * FROM positions WHERE account=? AND status='open'", [account]
        if exchange and account != "paper":
            q += " AND exchange=?"
            args.append(exchange)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def history(self, account: str | None = None, limit: int = 1000):
        q = "SELECT * FROM positions WHERE status='closed'"
        args = []
        if account:
            q += " AND account=?"
            args.append(account)
        q += " ORDER BY closed_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, args).fetchall()]

    def realized_since(self, account: str, since_ts: float) -> float:
        """Net PnL of trades closed since `since_ts` plus realized partials (and fees) of
        positions still open that were opened since then."""
        with self._lock:
            r = self._conn.execute("SELECT COALESCE(SUM(pnl),0) s FROM positions WHERE account=? "
                                   "AND (closed_at>=? OR (status='open' AND opened_at>=?))",
                                   (account, since_ts, since_ts)).fetchone()
        return float(r["s"])

    def clear_account(self, account: str):
        with self._lock:
            self._conn.execute("DELETE FROM positions WHERE account=?", (account,))
            self._conn.commit()

    # ---------------- signals
    def add_signal(self, s):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                               (s.id, s.created, s.symbol, s.timeframe, s.strategy, s.side, s.entry,
                                s.sl, json.dumps(s.tps), s.confidence, json.dumps(s.reasons), 0))
            self._conn.commit()

    def mark_signal(self, sig_id: str):
        with self._lock:
            self._conn.execute("UPDATE signals SET acted=1 WHERE id=?", (sig_id,))
            self._conn.commit()

    def recent_signals(self, limit: int = 200):
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM signals ORDER BY created DESC LIMIT ?", (limit,)).fetchall()]
