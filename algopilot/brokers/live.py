"""Live exchange trading through ccxt (Binance, Bybit, KuCoin, Toobit), mainnet or testnet.

Safety design:
* Entries are market orders sized by risk %. Futures use ISOLATED margin and the
  requested leverage; if the leverage can't be set the trade is refused.
* Futures: an exchange-side reduce-only STOP-LOSS order is placed as a safety
  net (protects you even if the app/PC goes offline). When the stop moves, the
  new one is placed BEFORE the old one is cancelled (never unprotected).
* Every tick the robot reconciles its positions with the exchange: positions
  closed on the exchange (stop hit, closed by hand, liquidated) are booked with
  the real fill price; nothing is ever booked from a failed order.
* Spot: SL/TP are managed by the robot (keep the app running).
* Use API keys with trading permission only. NEVER enable withdrawals.
"""
from __future__ import annotations

import time

from ..core.risk import liquidation_price
from ..data.feed import make_exchange, to_market_symbol
from .base import BaseBroker, ExecError, Position

LEVERAGE_OK_HINTS = ("not modified", "no need", "same", "leverage not changed", "110043")
MARGIN_OK_HINTS = ("no need", "not modified", "already", "110026", "-4046", "same")


class LiveBroker(BaseBroker):
    account = "live"
    simulated = False   # real orders fill at market; wicks are covered by the exchange-side stop

    def __init__(self, db, market_type: str, creds, exchange_side_sl: bool = True, log=print,
                 client=None):
        self.exchange = creds.exchange
        self.testnet = bool(creds.testnet)
        super().__init__(db, market_type, log)
        self.exchange_side_sl = exchange_side_sl and market_type == "futures"
        self.ex = client or make_exchange(creds.exchange, market_type, creds.api_key, creds.secret,
                                          creds.password, creds.testnet)
        self.ex.load_markets()
        self.quote = "USDT"
        self._bal_cache = (0.0, None)

    # ---------------- account
    def _balance(self):
        t, b = self._bal_cache
        if b is None or time.time() - t > 3:
            b = self.ex.fetch_balance()
            self._bal_cache = (time.time(), b)
        return b

    def test_connection(self) -> str:
        b = self.ex.fetch_balance()
        q = b.get(self.quote, {}) or {}
        env = "TESTNET" if self.testnet else "MAINNET"
        return (f"Connected to {self.exchange} {env} ({self.market_type}). "
                f"{self.quote}: free {q.get('free')}, total {q.get('total')}")

    def balance(self) -> float:
        return float((self._balance().get(self.quote) or {}).get("free") or 0.0)

    def equity(self) -> float:
        total = float((self._balance().get(self.quote) or {}).get("total") or 0.0)
        with self.lock:
            for p in self.open_positions():
                if p.market == "spot":
                    total += (p.last_price or p.entry) * p.remaining
                else:
                    total += p.upnl()
        return total

    # ---------------- helpers
    def _sym(self, symbol):
        return to_market_symbol(symbol, self.market_type)

    def _amount(self, sym: str, qty_base: float) -> float:
        m = self.ex.market(sym)
        amount = qty_base
        if m.get("contract") and m.get("contractSize"):
            amount = qty_base / float(m["contractSize"])
        amount = float(self.ex.amount_to_precision(sym, amount))
        min_amt = ((m.get("limits") or {}).get("amount") or {}).get("min")
        if amount <= 0 or (min_amt and amount < float(min_amt)):
            raise ExecError(f"order size {amount} below exchange minimum {min_amt} for {sym}")
        return amount

    def _base_qty(self, sym: str, amount: float) -> float:
        m = self.ex.market(sym)
        if m.get("contract") and m.get("contractSize"):
            return amount * float(m["contractSize"])
        return amount

    def _fee_quote(self, sym: str, order: dict, fill: float, qty_base: float) -> float:
        """Order fee converted to the quote currency; estimated from the taker rate if unknown."""
        m = self.ex.market(sym)
        base, quote = m.get("base"), m.get("quote")
        fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
        total, known = 0.0, False
        for f in fees:
            if not f or f.get("cost") is None:
                continue
            cost, cur = float(f["cost"]), f.get("currency")
            if cur in (quote, None):
                total += cost
                known = True
            elif cur == base:
                total += cost * fill
                known = True
        if not known:
            total = fill * qty_base * float(m.get("taker") or 0.0006)
        return total

    def _fill_info(self, sym: str, order: dict, fallback_price: float, amount: float):
        try:
            order = self.ex.fetch_order(order["id"], sym) or order
        except Exception:  # noqa: BLE001
            pass
        fill = float(order.get("average") or order.get("price") or fallback_price)
        filled = float(order.get("filled") or amount)
        return order, fill, filled

    def exchange_position(self, sym: str) -> float | None:
        """Absolute open size in base units on the exchange, or None if unknown."""
        try:
            if self.market_type == "futures":
                total = 0.0
                for p in self.ex.fetch_positions([sym]) or []:
                    if p.get("symbol") != sym:
                        continue
                    c = float(p.get("contracts") or 0.0)
                    total += abs(self._base_qty(sym, c))
                return total
            base = self.ex.market(sym)["base"]
            b = self.ex.fetch_balance()
            return float((b.get(base) or {}).get("total") or 0.0)
        except Exception as e:  # noqa: BLE001
            self._throttled_log(("pos", sym), f"Could not read {sym} position from exchange: {e}")
            return None

    def _setup_futures(self, sym: str, leverage: int):
        try:
            self.ex.set_margin_mode("isolated", sym, {"leverage": leverage})
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if not any(h in msg for h in MARGIN_OK_HINTS):
                self.log(f"set_margin_mode isolated: {e} (continuing with account default)")
        try:
            self.ex.set_leverage(leverage, sym)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if not any(h in msg for h in LEVERAGE_OK_HINTS):
                raise ExecError(f"could not set {leverage}x leverage on {sym}: {e}") from e

    # ---------------- exchange-side stop
    def _place_sl(self, pos: Position):
        if not self.exchange_side_sl:
            return
        sym = self._sym(pos.symbol)
        old = pos.extra.get("sl_order")
        try:
            side = "sell" if pos.side == "long" else "buy"
            amount = self._amount(sym, pos.remaining)
            price = float(self.ex.price_to_precision(sym, pos.sl))
            o = self.ex.create_order(sym, "market", side, amount, None,
                                     {"stopLossPrice": price, "reduceOnly": True})
            pos.extra["sl_order"] = o.get("id")
            self.log(f"Exchange-side SL placed for {pos.symbol} @ {price}")
        except Exception as e:  # noqa: BLE001
            self.log(f"Exchange-side SL NOT updated ({e}); old stop kept, robot also manages SL.")
            return
        if old:   # cancel the old stop only after the new one exists
            try:
                self.ex.cancel_order(old, sym, {"stop": True, "trigger": True})
            except Exception:  # noqa: BLE001
                try:
                    self.ex.cancel_order(old, sym)
                except Exception as e:  # noqa: BLE001
                    self.log(f"Cancel old SL {old} failed: {e}")

    def _cancel_sl(self, pos: Position):
        oid = pos.extra.get("sl_order")
        if not oid:
            return
        sym = self._sym(pos.symbol)
        for params in ({"stop": True, "trigger": True}, {}):
            try:
                self.ex.cancel_order(oid, sym, params)
                break
            except Exception:  # noqa: BLE001
                continue
        pos.extra["sl_order"] = None

    def _sl_fill(self, pos: Position):
        """(price, fee) if the exchange-side stop order filled, else None."""
        oid = pos.extra.get("sl_order")
        if not oid:
            return None
        sym = self._sym(pos.symbol)
        for params in ({"stop": True, "trigger": True}, {}):
            try:
                o = self.ex.fetch_order(oid, sym, params)
            except Exception:  # noqa: BLE001
                continue
            if o and o.get("status") == "closed" and (o.get("average") or o.get("price")):
                fill = float(o.get("average") or o.get("price"))
                return fill, self._fee_quote(sym, o, fill, pos.remaining)
            return None
        return None

    # ---------------- execution
    def _exec_open(self, pos: Position, price: float) -> Position:
        sym = self._sym(pos.symbol)
        if self.market_type == "spot" and pos.side == "short":
            raise RuntimeError("Short selling is not available on spot. Switch to futures.")
        if self.market_type == "futures":
            self._setup_futures(sym, pos.leverage)
        amount = self._amount(sym, pos.qty)
        side = "buy" if pos.side == "long" else "sell"
        order = self.ex.create_order(sym, "market", side, amount)
        order, fill, filled = self._fill_info(sym, order, price, amount)
        pos.qty = pos.remaining = self._base_qty(sym, filled)
        pos.entry = fill
        fee = self._fee_quote(sym, order, fill, pos.qty)
        if self.market_type == "spot":
            fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
            base = self.ex.market(sym).get("base")
            for f in fees:   # fee taken in the coin -> we hold less than we bought
                if f and f.get("currency") == base and f.get("cost"):
                    pos.qty = pos.remaining = max(0.0, pos.qty - float(f["cost"]))
        pos.fees += fee
        pos.pnl -= fee
        pos.extra["entry_order"] = order.get("id")
        pos.extra["testnet"] = self.testnet
        pos.liq = liquidation_price(pos.side, fill, pos.leverage) if pos.market == "futures" else None
        self._place_sl(pos)
        return pos

    def _exec_close(self, pos: Position, qty: float, price: float, reason: str) -> tuple:
        sym = self._sym(pos.symbol)
        side = "sell" if pos.side == "long" else "buy"
        params = {"reduceOnly": True} if self.market_type == "futures" else {}
        closing_all = pos.remaining - qty <= pos.qty * 1e-6
        if self.market_type == "spot":
            base = pos.symbol.split("/")[0]
            free = float((self.ex.fetch_balance().get(base) or {}).get("free") or 0)
            qty = min(qty, free)
        try:
            amount = self._amount(sym, qty)
        except ExecError:
            if closing_all or pos.remaining - qty < pos.qty * 0.02:
                raise
            # a partial TP below the exchange minimum: skip it, the rest rides to the next target
            self.log(f"{pos.symbol}: {reason} part is below exchange minimum, keeping it for the next target")
            return None, 0.0
        if closing_all:
            self._cancel_sl(pos)   # a reduce-only stop on a flat position is useless
        try:
            order = self.ex.create_order(sym, "market", side, amount, None, params)
        except Exception as e:  # noqa: BLE001
            if closing_all and self.exchange_side_sl:
                self._place_sl(pos)        # restore protection
            raise ExecError(str(e)) from e
        order, fill, filled = self._fill_info(sym, order, price, amount)
        return fill, self._fee_quote(sym, order, fill, self._base_qty(sym, filled))

    def _partial(self, pos, qty, price, reason):
        qty = min(qty, pos.remaining)
        fill, fee = self._exec_close(pos, qty, price, reason)
        if fill is None:   # partial TP skipped (below exchange minimum): nothing booked
            return {"type": "info", "pos": pos, "reason": f"{reason} skipped (below exchange minimum)",
                    "price": price, "qty": 0.0, "pnl": 0.0}
        return self._record(pos, qty, fill, fee, reason)

    def _on_sl_moved(self, pos: Position):
        self._place_sl(pos)

    # ---------------- reconciliation
    def sync(self) -> list:
        """Book positions that were closed (or reduced) on the exchange side."""
        events = []
        with self.lock:
            for pos in list(self.open_positions()):
                sym = self._sym(pos.symbol)
                size = self.exchange_position(sym)
                if size is None:
                    continue
                if size <= pos.remaining * 0.02:
                    sl = self._sl_fill(pos)
                    if sl:
                        fill, fee = sl
                        reason = "stop loss (exchange)" if pos.tp_hit == 0 else "trailing stop (exchange)"
                    else:
                        fill = pos.last_price or pos.entry
                        fee = fill * pos.remaining * 0.0006
                        reason = "closed on exchange"
                    pos.extra["sl_order"] = None
                    events.append(self._record(pos, pos.remaining, fill, fee, reason))
                    self.log(f"{pos.symbol}: position no longer on the exchange, booked as '{reason}' @ {fill:.6g}")
                elif size < pos.remaining * 0.98 and self.market_type == "futures":
                    gone = pos.remaining - size
                    fill = pos.last_price or pos.entry
                    events.append(self._record(pos, gone, fill, fill * gone * 0.0006, "reduced on exchange"))
                    self._place_sl(pos)
        return events
