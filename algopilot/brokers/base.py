"""Broker base: position model + shared trade management (partial TPs,
breakeven, trailing, liquidation) used by both Demo and Live brokers."""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..core.risk import liquidation_price

TP_FRACTIONS = (0.5, 0.3, 0.2)


@dataclass
class Position:
    symbol: str
    side: str
    market: str
    qty: float
    entry: float
    sl: float
    tps: list
    leverage: int = 1
    account: str = "paper"
    exchange: str = "demo"
    strategy: str = ""
    timeframe: str = ""
    remaining: float = -1.0
    tp_hit: int = 0
    opened_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    fees: float = 0.0
    status: str = "open"
    close_reason: str = ""
    liq: float | None = None
    extra: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    last_price: float = 0.0

    def __post_init__(self):
        if self.remaining < 0:
            self.remaining = self.qty
        if self.market == "futures" and self.liq is None:
            self.liq = liquidation_price(self.side, self.entry, self.leverage)

    @property
    def sgn(self):
        return 1 if self.side == "long" else -1

    def upnl(self, price: float | None = None) -> float:
        p = price or self.last_price or self.entry
        return self.sgn * (p - self.entry) * self.remaining

    def to_row(self) -> dict:
        return {"id": self.id, "account": self.account, "exchange": self.exchange, "symbol": self.symbol,
                "side": self.side, "market": self.market, "qty": self.qty, "remaining": self.remaining,
                "entry": self.entry, "sl": self.sl, "tps": json.dumps(self.tps), "tp_hit": self.tp_hit,
                "leverage": self.leverage, "opened_at": self.opened_at, "closed_at": self.closed_at,
                "exit_price": self.exit_price, "pnl": self.pnl, "fees": self.fees, "status": self.status,
                "strategy": self.strategy, "timeframe": self.timeframe, "close_reason": self.close_reason,
                "liq": self.liq, "extra": json.dumps(self.extra)}

    @classmethod
    def from_row(cls, r: dict) -> "Position":
        d = dict(r)
        d["tps"] = json.loads(d["tps"] or "[]")
        d["extra"] = json.loads(d.get("extra") or "{}")
        return cls(**d)


class ExecError(RuntimeError):
    """An exchange order failed; the position state was NOT changed."""


class BaseBroker:
    account = "paper"
    exchange = "demo"
    simulated = True    # paper: fills happen at the touched level (wicks between polls count)

    def __init__(self, db, market_type: str, log=print):
        self.db = db
        self.market_type = market_type
        self.log = log
        self.lock = threading.RLock()
        self.positions: dict[str, Position] = {}
        self._err_at: dict = {}
        for r in db.open_positions(self.account, self.exchange):
            p = Position.from_row(r)
            self.positions[p.id] = p

    # ---- to implement
    def balance(self) -> float: ...
    def equity(self) -> float: ...
    def _exec_open(self, pos: Position, price: float) -> Position: ...
    def _exec_close(self, pos: Position, qty: float, price: float, reason: str) -> tuple:
        """Execute close of qty at ~price. Return (fill_price, fee_in_quote).
        Raise ExecError if the order did not go through (position stays open)."""
    def _on_sl_moved(self, pos: Position): ...
    def sync(self) -> list:
        """Reconcile with the exchange. Returns trade events."""
        return []

    # ---- public API
    def open_positions(self):
        with self.lock:
            return [p for p in self.positions.values() if p.status == "open"]

    def open_from_signal(self, signal, qty: float, leverage: int, price: float | None = None) -> Position:
        with self.lock:
            pos = Position(symbol=signal.symbol, side=signal.side, market=self.market_type, qty=qty,
                           entry=price or signal.entry, sl=signal.sl, tps=list(signal.tps),
                           leverage=leverage if self.market_type == "futures" else 1,
                           account=self.account, exchange=self.exchange, strategy=signal.strategy,
                           timeframe=signal.timeframe)
            pos = self._exec_open(pos, price or signal.entry)
            self.positions[pos.id] = pos
            self.db.save_position(pos)
            return pos

    def close(self, pos_id: str, price: float, reason: str = "manual") -> list:
        with self.lock:
            pos = self.positions.get(pos_id)
            if not pos or pos.status != "open":
                return []
            return [self._partial(pos, pos.remaining, price, reason)]

    def _record(self, pos: Position, qty: float, fill: float, fee: float, reason: str) -> dict:
        """Book a (partial) exit that already happened. PnL is net of fees."""
        qty = min(qty, pos.remaining)
        pnl = pos.sgn * (fill - pos.entry) * qty - fee
        pos.pnl += pnl
        pos.fees += fee
        pos.remaining -= qty
        ev = {"type": "partial", "pos": pos, "reason": reason, "price": fill, "qty": qty, "pnl": pnl}
        if pos.remaining <= pos.qty * 1e-6:
            pos.remaining = 0.0
            pos.status = "closed"
            pos.closed_at = time.time()
            pos.exit_price = fill
            pos.close_reason = reason
            ev["type"] = "closed"
            self.positions.pop(pos.id, None)
        self.db.save_position(pos)
        return ev

    def _partial(self, pos: Position, qty: float, price: float, reason: str) -> dict:
        qty = min(qty, pos.remaining)
        fill, fee = self._exec_close(pos, qty, price, reason)   # raises ExecError -> nothing booked
        return self._record(pos, qty, fill, fee, reason)

    def _throttled_log(self, key, msg, every=60):
        now = time.time()
        if now - self._err_at.get(key, 0) >= every:
            self._err_at[key] = now
            self.log(msg)

    def on_price(self, symbol: str, price: float, high: float | None = None,
                 low: float | None = None, window_start: float | None = None) -> list:
        """Manage SL / TP / liquidation for every open position on `symbol`.
        `high`/`low`: extremes traded since the previous check (catches wicks between polls).
        Stops are checked before targets (conservative)."""
        events = []
        use_range = self.simulated and high is not None and low is not None
        with self.lock:
            for pos in [p for p in self.positions.values() if p.symbol == symbol and p.status == "open"]:
                pos.last_price = price
                hi = lo = price
                # extremes only count for positions that were already open when the window began
                if use_range and (window_start is None or pos.opened_at <= window_start):
                    hi, lo = max(price, high), min(price, low)
                try:
                    events.extend(self._manage(pos, price, hi, lo))
                except ExecError as e:
                    self._throttled_log(("exec", pos.id), f"{pos.symbol}: order failed, position kept "
                                                         f"open and will retry: {e}")
        return events

    def _manage(self, pos: Position, price: float, hi: float, lo: float) -> list:
        s = pos.sgn
        adverse = lo if s == 1 else hi
        favour = hi if s == 1 else lo
        if pos.liq and ((s == 1 and adverse <= pos.liq) or (s == -1 and adverse >= pos.liq)) and \
                not ((s == 1 and pos.sl > pos.liq) or (s == -1 and pos.sl < pos.liq)):
            return [self._partial(pos, pos.remaining, pos.liq, "liquidated")]
        if (s == 1 and adverse <= pos.sl) or (s == -1 and adverse >= pos.sl):
            reason = "stop loss" if pos.tp_hit == 0 else ("breakeven" if pos.tp_hit == 1 else "trailing stop")
            # filled at the stop, or at the current price if it already gapped through
            fill = min(pos.sl, price) if s == 1 else max(pos.sl, price)
            return [self._partial(pos, pos.remaining, fill, reason)]
        events = []
        while pos.status == "open" and pos.tp_hit < len(pos.tps) and \
                ((s == 1 and favour >= pos.tps[pos.tp_hit]) or (s == -1 and favour <= pos.tps[pos.tp_hit])):
            k = pos.tp_hit
            last = k == len(pos.tps) - 1
            qty = pos.remaining if last else min(pos.remaining, pos.qty * TP_FRACTIONS[min(k, 2)])
            tp_px = pos.tps[k]
            fill = tp_px if (s == 1 and price < tp_px) or (s == -1 and price > tp_px) else price
            events.append(self._partial(pos, qty, fill, f"TP{k + 1}"))
            pos.tp_hit += 1
            if pos.status == "open":
                pos.sl = pos.entry if pos.tp_hit == 1 else pos.tps[pos.tp_hit - 2]
                self._on_sl_moved(pos)
            self.db.save_position(pos)
        return events
