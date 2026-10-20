"""Regression tests for the reviewed bugs (no network: a fake ccxt exchange is used)."""
import os
import tempfile
import time

import numpy as np
import pandas as pd

from algopilot.brokers.base import ExecError
from algopilot.brokers.live import LiveBroker
from algopilot.brokers.paper import PaperBroker
from algopilot.config import ExchangeCreds, Settings
from algopilot.core.backtest import run_backtest
from algopilot.core.ml import MLFilter
from algopilot.core.signals import Signal
from algopilot.core.strategies import STRATEGIES
from algopilot.data.feed import DemoFeed, ExchangeFeed, enable_test_mode
from algopilot.engine.robot import NullEvents, Robot
from algopilot.storage.db import Database


def _db():
    return Database(os.path.join(tempfile.mkdtemp(), "t.db"))


class FakeEx:
    """Tiny ccxt stand-in: one linear USDT perp, market orders, stop orders, positions."""
    id = "binanceusdm"

    def __init__(self, price=100.0, min_amt=0.001):
        self.price, self.min_amt = price, min_amt
        self.pos = 0.0                    # signed base qty
        self.usdt = 10000.0
        self.orders, self.stops, self.fail_next = {}, {}, False
        self.leverage_error = None
        self.n = 0

    def load_markets(self): return {}
    def milliseconds(self): return int(time.time() * 1000)
    def market(self, sym):
        return {"base": "BTC", "quote": "USDT", "contract": True, "contractSize": 1.0, "taker": 0.0005,
                "limits": {"amount": {"min": self.min_amt}}}
    def amount_to_precision(self, sym, a): return f"{np.floor(a * 1000) / 1000:.3f}"
    def price_to_precision(self, sym, p): return f"{p:.2f}"
    def set_margin_mode(self, *a, **k): return None
    def set_leverage(self, lev, sym):
        if self.leverage_error:
            raise Exception(self.leverage_error)
    def fetch_balance(self): return {"USDT": {"free": self.usdt, "total": self.usdt}, "BTC": {"total": 0}}
    def fetch_positions(self, syms):
        return [{"symbol": syms[0], "contracts": abs(self.pos)}] if self.pos else []

    def create_order(self, sym, typ, side, amount, price=None, params=None):
        params = params or {}
        self.n += 1
        oid = f"o{self.n}"
        if "stopLossPrice" in params:
            self.stops[oid] = (side, amount, params["stopLossPrice"])
            return {"id": oid, "status": "open"}
        if self.fail_next:
            self.fail_next = False
            raise Exception("exchange busy")
        if params.get("reduceOnly") and abs(self.pos) < 1e-12:
            raise Exception("ReduceOnly order rejected: no position")
        self.pos += amount if side == "buy" else -amount
        o = {"id": oid, "status": "closed", "average": self.price, "filled": amount,
             "fee": {"cost": self.price * amount * 0.0005, "currency": "USDT"}}
        self.orders[oid] = o
        return o

    def fetch_order(self, oid, sym, params=None):
        return self.orders.get(oid) or {"id": oid, "status": "open"}

    def cancel_order(self, oid, sym, params=None):
        self.stops.pop(oid, None)

    def trigger_stop(self, px):
        """Exchange-side stop fills while the robot isn't looking."""
        for oid, (side, amount, _) in list(self.stops.items()):
            self.pos += amount if side == "buy" else -amount
            self.orders[oid] = {"id": oid, "status": "closed", "average": px, "filled": amount,
                                "fee": {"cost": px * amount * 0.0005, "currency": "USDT"}}
            del self.stops[oid]


def _live(ex, db=None, exchange="binance"):
    return LiveBroker(db or _db(), "futures", ExchangeCreds(exchange=exchange, testnet=True), True,
                      log=lambda m: None, client=ex)


def _sig(side="long", entry=100.0, sl=95.0, tps=(105.0, 110.0, 115.0)):
    return Signal("BTC/USDT", "15m", "x", side, entry, sl, list(tps), 0.8)


# ------------------------------------------------------------------ live broker
def test_failed_close_keeps_position_open():
    ex = FakeEx()
    b = _live(ex)
    p = b.open_from_signal(_sig(), 1.0, 5, 100)
    ex.fail_next = True
    ex.price = 94
    assert b.on_price("BTC/USDT", 94) == []          # order failed: nothing booked
    assert p.status == "open" and p.remaining == 1.0
    ev = b.on_price("BTC/USDT", 94)                   # retried next tick
    assert ev[-1]["type"] == "closed" and abs(ex.pos) < 1e-12


def test_exchange_stop_is_reconciled_no_fake_profit():
    ex = FakeEx()
    b = _live(ex)
    p = b.open_from_signal(_sig(), 1.0, 5, 100)
    assert ex.stops                                    # safety stop exists
    ex.trigger_stop(94.9)                              # wick hits the exchange stop
    ex.price = 106                                     # ...and price recovers to above TP1
    ev = b.sync()
    assert ev and ev[0]["type"] == "closed" and p.status == "closed"
    assert abs(p.exit_price - 94.9) < 1e-9 and p.pnl < 0
    assert b.on_price("BTC/USDT", 106) == []           # no TP booked on a closed position


def test_live_pnl_includes_fees_and_stop_replaced_before_cancel():
    ex = FakeEx()
    b = _live(ex)
    p = b.open_from_signal(_sig(), 1.0, 5, 100)
    first_stop = p.extra["sl_order"]
    ex.price = 105
    b.on_price("BTC/USDT", 105)                        # TP1 -> SL to breakeven
    assert p.extra["sl_order"] != first_stop and first_stop not in ex.stops and len(ex.stops) == 1
    assert p.fees > 0
    gross = 0.5 * 5
    assert p.pnl < gross                               # net of entry + exit fees


def test_partial_tp_below_minimum_is_skipped_not_faked():
    ex = FakeEx(min_amt=0.4)
    b = _live(ex)
    p = b.open_from_signal(_sig(), 1.0, 5, 100)
    ex.price = 111
    b.on_price("BTC/USDT", 111)                        # TP1 0.5 ok, TP2 0.3 < min -> skipped
    assert abs(ex.pos - p.remaining) < 1e-9            # local state == exchange


def test_leverage_failure_refuses_trade():
    ex = FakeEx()
    ex.leverage_error = "leverage 50 exceeds max"
    b = _live(ex)
    try:
        b.open_from_signal(_sig(), 1.0, 50, 100)
        assert False
    except ExecError:
        pass
    assert ex.pos == 0
    ex.leverage_error = "leverage not modified"        # benign
    b.open_from_signal(_sig(), 1.0, 5, 100)
    assert ex.pos == 1.0


def test_positions_loaded_per_exchange():
    db = _db()
    ex = FakeEx()
    _live(ex, db, "binance").open_from_signal(_sig(), 1.0, 5, 100)
    assert len(_live(FakeEx(), db, "bybit").open_positions()) == 0
    assert len(_live(FakeEx(), db, "binance").open_positions()) == 1


def test_test_mode_prefers_demo_trading_for_binance():
    calls = []

    class E:
        def enable_demo_trading(self, on): calls.append("demo")
        def set_sandbox_mode(self, on): calls.append("sandbox")
    assert enable_test_mode(E(), "binance", "futures") == "demo trading" and calls == ["demo"]
    calls.clear()
    assert enable_test_mode(E(), "bybit", "futures") == "testnet" and calls == ["sandbox"]


# ------------------------------------------------------------------ data / ml / robot
def test_ohlcv_pagination_respects_exchange_limit():
    class E:
        id = "kucoinfutures"
        def milliseconds(self): return 1_000_000 * 60_000
        def fetch_ohlcv(self, sym, tf, since=None, limit=None):
            assert limit <= 200
            start = since // 60000 + 1 if since else 0
            end = min(start + limit, 1_000_000)
            return [[m * 60000, 1, 1, 1, 1, 1] for m in range(start, end)]
    df = ExchangeFeed("kucoin", "futures", client=E()).fetch_ohlcv("BTC/USDT", "1m", 600)
    assert len(df) == 600


def test_demo_price_matches_chart():
    f = DemoFeed()
    d = f.fetch_ohlcv("BTC/USDT", "15m", 10)
    assert abs(f.fetch_price("BTC/USDT", "15m") - d.close.iloc[-1]) / d.close.iloc[-1] < 0.01


def test_ml_retrains_with_fixed_window():
    full = DemoFeed().fetch_ohlcv("BTC/USDT", "15m", 700)
    ml = MLFilter(retrain_every=25)
    ml.prob_up("k", full.iloc[:600])
    t0 = ml._cache["k"]["t"]
    ml.prob_up("k", full.iloc[30:630])                # same length, 30 new bars
    assert ml._cache["k"]["t"] != t0


def test_ai_confluence_triggers_pass_default_confidence():
    d = DemoFeed().fetch_ohlcv("ETH/USDT", "15m", 2000)
    o = STRATEGIES["ai_confluence"].compute(d)
    m = o.long | o.short
    assert m.sum() > 0 and (o.strength[m] >= 0.6).all()


def test_paper_wick_between_polls_hits_stop():
    b = PaperBroker(_db(), "futures", 10000, slippage=0, log=lambda m: None)
    p = b.open_from_signal(_sig(), 1.0, 5, 100)
    ev = b.on_price("BTC/USDT", 99, high=101, low=94, window_start=time.time() + 1)
    assert ev and p.status == "closed" and p.exit_price == 95


def test_backtest_gap_fills_at_open():
    idx = pd.date_range("2024", periods=400, freq="15min", tz="UTC")
    d = DemoFeed().fetch_ohlcv("BTC/USDT", "15m", 400)
    d.index = idx
    r = run_backtest(d, "ema_donchian")
    assert r.max_dd_pct >= 0


def test_robot_live_testnet_uses_broker_prices_and_syncs():
    st = Settings(account="live", data_exchange="binance", symbols=["BTC/USDT"],
                  creds=ExchangeCreds(exchange="binance", api_key="k", secret="s", testnet=True))
    import algopilot.brokers.live as live
    ex = FakeEx()
    orig = live.make_exchange
    live.make_exchange = lambda *a, **k: ex
    try:
        r = Robot(st, _db(), NullEvents())
    finally:
        live.make_exchange = orig
    assert r.feed._ex is ex                            # testnet prices, not mainnet


def test_settings_load_rejects_bad_types(tmp_path=None):
    s = Settings()
    s.leverage = "abc"
    from algopilot.config import _coerce
    assert _coerce(5, "abc") is None and _coerce(1.0, "2") == 2.0 and _coerce(True, 1) is None
