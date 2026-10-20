"""v3 tests: new indicators, strategies, auto-tuner and the demo open/close path (no network)."""
import os
import tempfile

import numpy as np

from algopilot.config import Settings
from algopilot.core import indicators as ind
from algopilot.core.luxalgo import signals_overlays
from algopilot.core.smc import smc
from algopilot.core.sp2l import sp2l
from algopilot.core.signals import Signal
from algopilot.core.strategies import STRATEGIES
from algopilot.core.tuner import AutoTuner, autopilot_params
from algopilot.data.feed import DemoFeed
from algopilot.engine.robot import NullEvents, Robot
from algopilot.storage.db import Database


def _df(tf="15m", n=1500, sym="BTC/USDT"):
    return DemoFeed().fetch_ohlcv(sym, tf, n)


def test_linreg_and_wma_match_reference():
    d = _df(n=300)
    x = d["close"]
    ref = x.rolling(20).apply(lambda y: np.polyval(np.polyfit(np.arange(20), y, 1), 19), raw=True)
    assert np.nanmax(np.abs(ref - ind.linreg(x, 20))) < 1e-6
    w = np.arange(1, 11)
    ref2 = x.rolling(10).apply(lambda y: np.dot(y, w) / w.sum(), raw=True)
    assert np.nanmax(np.abs(ref2 - ind.wma(x, 10))) < 1e-6


def test_lazybear_squeeze_states():
    q = ind.squeeze_lazybear(_df())
    assert set(np.unique(q["dot_color"].values)) <= {0, 1, 2}
    assert set(np.unique(q["bar_color"].values)) <= {-2, -1, 0, 1, 2}
    assert not (q["sqz_on"] & q["sqz_off"]).any()


def test_new_indicators_are_causal():
    """Changing the future must not change any past value (no look-ahead)."""
    d = _df(n=900)
    cut = 700
    a, b = d.iloc[:cut], d.copy()
    b.iloc[cut:, b.columns.get_loc("close")] *= 1.5
    b.iloc[cut:, b.columns.get_loc("high")] *= 1.5
    b.iloc[cut:, b.columns.get_loc("low")] *= 1.5
    for fn in (lambda df: smc(df)["swing_trend"], lambda df: sp2l(df)["long"].astype(float),
               lambda df: signals_overlays(df)["buy"].astype(float)):
        va, vb = fn(a).iloc[:cut - 60], fn(b).iloc[:cut - 60]
        assert np.allclose(np.asarray(va, float), np.asarray(vb, float), equal_nan=True)


def test_all_strategies_return_valid_frames():
    d = _df()
    for k, s in STRATEGIES.items():
        p = {"mode": "daily"} if k == "auto_pilot" else None
        o = s.compute(d, p)
        assert len(o) == len(d)
        assert o.long.dtype == bool and o.short.dtype == bool
        assert not (o.long & o.short).any()
        assert o.strength.between(0, 1).all()


def test_sp2l_stop_below_long_entry():
    d = _df(n=2000)
    o = STRATEGIES["sp2l"].compute(d, STRATEGIES["sp2l"].params_for("daily"))
    idx = np.where(o.long.values)[0]
    for i in idx:
        assert o["sl"].iloc[i] < d["close"].iloc[i]


def test_tuner_runs_and_mode_params_are_valid():
    d = _df(n=1200)
    res = AutoTuner(max_combos=4).tune_mode(d, "daily", keys=["smc", "squeeze"])
    assert set(res) == {"smc", "squeeze"}
    for r in res.values():
        assert r.status in ("validated", "unproven", "failed", "untuned")
    ap = autopilot_params(res, "daily", "quality")
    assert set(ap["members"]) == {"smc", "squeeze"}


def test_robot_demo_open_and_close_flow():
    """Regression: manual open + close on the Demo account must update positions and history."""
    db = Database(os.path.join(tempfile.mkdtemp(), "t.db"))
    st = Settings(data_exchange="demo", symbols=["BTC/USDT"], trade_mode="notify", timeframe="15m", mode="daily")
    r = Robot(st, db, NullEvents())
    r.scanning = True
    r.tick()
    px = r.feed.fetch_price("BTC/USDT", "15m")
    sig = Signal("BTC/USDT", "15m", "test", "long", px, px * 0.98, [px * 1.01, px * 1.02, px * 1.03], 0.8)
    pos = r.execute_signal(sig)
    assert len(r.broker.open_positions()) == 1 and pos.status == "open"
    r.close_position(pos.id)
    assert r.broker.open_positions() == []
    assert len(db.history("paper")) == 1


def test_min_confidence_frequency_shift():
    st = Settings(min_confidence=0.55, signal_frequency="high")
    db = Database(os.path.join(tempfile.mkdtemp(), "t.db"))
    r = Robot(st, db, NullEvents())
    assert abs(r.min_confidence() - 0.48) < 1e-9
