import os
import tempfile

import numpy as np
import pandas as pd

from algopilot.brokers.paper import PaperBroker
from algopilot.config import Settings
from algopilot.core import indicators as ind
from algopilot.core.analyzer import Analyzer
from algopilot.core.backtest import rank_strategies, run_backtest
from algopilot.core.ml import MLFilter
from algopilot.core.risk import liquidation_price, position_size
from algopilot.core.signals import Signal
from algopilot.core.strategies import STRATEGIES
from algopilot.data.feed import DemoFeed
from algopilot.engine.robot import NullEvents, Robot
from algopilot.storage.db import Database


def df():
    return DemoFeed().fetch_ohlcv("BTC/USDT", "15m", 1200)


def test_indicators_shapes():
    d = df()
    for s in [ind.rsi(d.close), ind.atr(d), ind.supertrend(d)[0], ind.ut_bot(d)[0],
              ind.range_filter(d.close)[0], ind.squeeze_momentum(d)[0], ind.wavetrend(d)[0],
              ind.kalman_price(d.close), ind.vwap(d), ind.hma(d.close)]:
        assert len(s) == len(d)
        assert s.iloc[-50:].notna().all()
    r = ind.rsi(d.close).dropna()
    assert r.between(0, 100).all()


def test_strategies_and_backtest():
    d = df()
    for k, s in STRATEGIES.items():
        out = s.compute(d)
        assert {"long", "short", "strength", "sl"} <= set(out.columns)
        assert out.long.dtype == bool
        r = run_backtest(d, k)
        assert r.n >= 0
    ranks = rank_strategies({"BTC/USDT": d}, "daily")
    assert ranks and ranks[0]["score"] >= ranks[-1]["score"]


def test_signal_plan_geometry():
    d = df()
    an = Analyzer("ai_confluence", MLFilter(), True)
    sig, a, ph, pl = an.prepare(d)
    for i in np.where(sig.long | sig.short)[0]:
        if i < 210:
            continue
        side = "long" if sig.long.iloc[i] else "short"
        p = an.plan(d, i, side, 0.7, a, ph, pl, "BTC/USDT", "15m")
        if side == "long":
            assert p.sl < p.entry < p.tps[0] < p.tps[1] < p.tps[2]
        else:
            assert p.sl > p.entry > p.tps[0] > p.tps[1] > p.tps[2]


def test_risk():
    q = position_size(10000, 1, 100, 98, "futures", 10)  # fee=0 default
    assert abs(q * 2 - 100) < 1e-9
    assert liquidation_price("long", 100, 10) < 100 < liquidation_price("short", 100, 10)


def test_paper_broker_tp_flow():
    db = Database(os.path.join(tempfile.mkdtemp(), "t.db"))
    b = PaperBroker(db, "futures", 10000, slippage=0)
    s = Signal("BTC/USDT", "15m", "x", "long", 100, 95, [105, 110, 115], 0.8)
    p = b.open_from_signal(s, 10, 5, 100)
    assert b.on_price("BTC/USDT", 106)[0]["reason"] == "TP1"
    assert p.sl == p.entry                                   # breakeven
    b.on_price("BTC/USDT", 111)
    assert p.sl == 105                                       # trailed to TP1
    ev = b.on_price("BTC/USDT", 104)
    assert ev[-1]["type"] == "closed" and p.status == "closed"
    assert p.pnl > 0
    assert len(db.history("paper")) == 1


def test_robot_tick_demo():
    db = Database(os.path.join(tempfile.mkdtemp(), "t.db"))
    st = Settings(data_exchange="demo", symbols=["BTC/USDT", "ETH/USDT"], trade_mode="auto")
    r = Robot(st, db, NullEvents())
    r.scanning = True
    r.tick()
    assert set(r.last_df) == {"BTC/USDT", "ETH/USDT"}
