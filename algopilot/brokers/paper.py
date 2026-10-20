"""Demo account: simulated trading with virtual money on live prices."""
from __future__ import annotations

from .base import BaseBroker, Position


class PaperBroker(BaseBroker):
    account = "paper"
    exchange = "demo"

    def __init__(self, db, market_type: str, start_balance: float = 10000.0,
                 fee: float = 0.0006, slippage: float = 0.0002, log=print):
        super().__init__(db, market_type, log)
        self.fee, self.slippage = fee, slippage
        self.start_balance = start_balance
        self.cash = float(db.get("paper_cash", start_balance))

    def _save(self):
        self.db.set("paper_cash", self.cash)

    def used_margin(self) -> float:
        return sum(p.entry * p.remaining / max(1, p.leverage) for p in self.open_positions()
                   if p.market == "futures")

    def balance(self) -> float:
        """Free balance available for new trades."""
        return self.cash - self.used_margin()

    def equity(self) -> float:
        eq = self.cash
        for p in self.open_positions():
            if p.market == "spot":
                eq += (p.last_price or p.entry) * p.remaining
            else:
                eq += p.upnl()
        return eq

    def reset(self, start_balance: float | None = None):
        with self.lock:
            self.cash = float(start_balance or self.start_balance)
            self.positions.clear()
            self.db.clear_account("paper")
            self._save()

    def _exec_open(self, pos: Position, price: float) -> Position:
        fill = price * (1 + pos.sgn * self.slippage)
        pos.entry = fill
        notional = fill * pos.qty
        fee = notional * self.fee
        if pos.market == "spot":
            if notional + fee > self.cash:
                raise RuntimeError("Not enough demo balance")
            self.cash -= notional + fee
        else:
            if notional / max(1, pos.leverage) + fee > self.balance():
                raise RuntimeError("Not enough demo margin")
            self.cash -= fee
        pos.fees += fee
        pos.pnl -= fee                     # entry fee; exit fees are booked by the base class
        from ..core.risk import liquidation_price
        pos.liq = liquidation_price(pos.side, fill, pos.leverage) if pos.market == "futures" else None
        self._save()
        return pos

    def _exec_close(self, pos: Position, qty: float, price: float, reason: str) -> tuple:
        fill = price if reason == "liquidated" else price * (1 - pos.sgn * self.slippage)
        fee = fill * qty * self.fee
        if pos.market == "spot":
            self.cash += fill * qty - fee
        else:
            pnl = pos.sgn * (fill - pos.entry) * qty
            if reason == "liquidated":     # isolated margin: you lose at most the margin
                pnl = max(pnl, -pos.entry * qty / max(1, pos.leverage))
            self.cash += pnl - fee
        self._save()
        return fill, fee

    def _on_sl_moved(self, pos):
        pass
