"""Trading strategies.

Each strategy returns a DataFrame aligned to the candles with columns
`long`, `short` (bool, signal on that closed bar), `strength` (0..1) and `sl`
(optional strategy stop price, NaN = let the planner pick a swing/ATR stop).
All strategies stack noise filters (ATR activity, volume, trend/ADX or
efficiency ratio) to avoid trading chop.

Every strategy has tunable `defaults`, per-style `presets` (scalp / daily / long) and a
`space` the auto-tuner searches (see tuner.py). `compute(df, p)` takes the params dict.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from . import indicators as ind
from .luxalgo import signals_overlays
from .smc import smc
from .sp2l import sp2l


def _b(s) -> pd.Series:
    return pd.Series(s).fillna(False).astype(bool)


def _cross_up(a, b):
    return _b((a > b) & (a.shift() <= b.shift()))


def _cross_dn(a, b):
    return _b((a < b) & (a.shift() >= b.shift()))


_CACHE: "OrderedDict" = OrderedDict()


def cached(name, df, fn, **kw):
    """Memoize heavy indicator passes (SMC, Signals & Overlays...) for the same candles+params,
    so Auto Pilot / AI Confluence / the chart don't recompute them several times per tick."""
    key = (name, id(df), len(df), df.index[-1] if len(df) else None,
           float(df["close"].iloc[-1]) if len(df) else 0.0, tuple(sorted(kw.items())))
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    val = fn(df, **kw)
    _CACHE[key] = val
    while len(_CACHE) > 96:
        _CACHE.popitem(last=False)
    return val


class Strategy:
    key = "base"
    name = "Base"
    modes = ("scalp", "daily", "long")
    description = ""
    defaults: dict = {}
    presets: dict = {}          # mode -> param overrides (starting point before tuning)
    space: dict = {}            # mode or "*" -> {param: [candidates]} searched by the auto-tuner

    def params_for(self, mode: str | None = None, overrides: dict | None = None) -> dict:
        p = dict(self.defaults)
        if mode:
            p.update(self.presets.get(mode, {}))
        if overrides:
            p.update({k: v for k, v in overrides.items() if k in p or not self.defaults})
        return p

    def search_space(self, mode: str) -> dict:
        return dict(self.space.get(mode) or self.space.get("*") or {})

    def compute(self, df: pd.DataFrame, p: dict | None = None) -> pd.DataFrame:
        params = dict(self.defaults)
        if p:
            params.update(p)
        return self._compute(df, params)

    def _compute(self, df, p):
        raise NotImplementedError

    @staticmethod
    def _out(df, long, short, strength, sl=None):
        out = pd.DataFrame({"long": _b(pd.Series(long, index=df.index)),
                            "short": _b(pd.Series(short, index=df.index)),
                            "strength": pd.Series(strength, index=df.index).astype(float).fillna(0).clip(0, 1)},
                           index=df.index)
        out["sl"] = np.nan if sl is None else pd.Series(sl, index=df.index).astype(float)
        out.loc[~(out["long"] | out["short"]), "sl"] = np.nan
        return out


class UTBotTrend(Strategy):
    key, name = "ut_bot_trend", "UT Bot + EMA200 Trend"
    modes = ("scalp", "daily")
    description = "UT Bot ATR-trailing-stop flips, only with the EMA200 trend and ADX filter."
    defaults = {"key": 1.5, "n": 10, "adx_min": 18, "trend_len": 200}
    presets = {"scalp": {"key": 1.2, "n": 10, "adx_min": 16}, "daily": {"key": 1.5, "n": 10}}
    space = {"*": {"key": [1.0, 1.5, 2.0, 2.5], "n": [7, 10, 14], "adx_min": [14, 18, 22],
                   "trend_len": [100, 200]}}

    def _compute(self, df, p):
        c = df["close"]
        _, buy, sell = ind.ut_bot(df, key=p["key"], n=int(p["n"]))
        tr = ind.ema(c, int(p["trend_len"]))
        adx, _, _ = ind.adx(df)
        kslope = ind.slope(ind.kalman_price(c), 5)
        ok = ind.noise_ok(df) & (adx > p["adx_min"])
        long = buy & (c > tr) & ok
        short = sell & (c < tr) & ok
        strength = 0.55 + 0.2 * (adx > 25) + 0.15 * (np.sign(kslope) == np.where(long, 1, -1))
        return self._out(df, long, short, strength)


class RangeFilterBreakout(Strategy):
    key, name = "range_filter", "Range Filter Breakout"
    modes = ("scalp", "daily")
    description = "Range Filter direction change with volume above average (noise-filtered trend)."
    defaults = {"n": 50, "mult": 2.5, "trend_len": 200}
    presets = {"scalp": {"n": 30, "mult": 2.0}}
    space = {"*": {"n": [20, 30, 50, 100], "mult": [1.5, 2.0, 2.5, 3.0], "trend_len": [100, 200]}}

    def _compute(self, df, p):
        c = df["close"]
        filt, d = ind.range_filter(c, int(p["n"]), p["mult"])
        vol_ok = df["volume"] > ind.sma(df["volume"], 20)
        tr = ind.ema(c, int(p["trend_len"]))
        ok = ind.noise_ok(df) & vol_ok
        long = (d == 1) & (d.shift() != 1) & (c > filt) & (c > tr) & ok
        short = (d == -1) & (d.shift() != -1) & (c < filt) & (c < tr) & ok
        er = ind.efficiency_ratio(c)
        return self._out(df, long, short, 0.55 + 0.4 * er)


class SqueezeLazyBear(Strategy):
    key, name = "squeeze", "Squeeze Momentum [LazyBear]"
    modes = ("scalp", "daily", "long")
    description = ("LazyBear Squeeze Momentum: trade the squeeze release (black dots -> gray) in the "
                   "direction of the momentum histogram (lime = strong up, red = strong down), with a "
                   "trend filter. Optional zero-line cross entries. Lengths auto-tuned per coin/timeframe.")
    defaults = {"length": 20, "mult": 2.0, "length_kc": 20, "mult_kc": 1.5, "release_bars": 2,
                "trend_len": 50, "zero_cross": False}
    presets = {"scalp": {"length": 20, "length_kc": 20, "release_bars": 2, "trend_len": 50, "zero_cross": True},
               "daily": {"release_bars": 2, "trend_len": 50},
               "long": {"release_bars": 3, "trend_len": 100}}
    space = {"*": {"length": [14, 20, 26], "length_kc": [14, 20, 26], "mult_kc": [1.25, 1.5, 2.0],
                   "release_bars": [1, 2, 4], "trend_len": [50, 100, 200], "zero_cross": [False, True]}}

    def _compute(self, df, p):
        c = df["close"]
        q = cached("sqz", df, ind.squeeze_lazybear, length=int(p["length"]), mult=float(p["mult"]),
                   length_kc=int(p["length_kc"]), mult_kc=float(p["mult_kc"]))
        val, on, bar = q["val"], q["sqz_on"], q["bar_color"]
        fired = _b(on.shift(1)) & ~on                      # black dot -> gray/blue = release
        recent = fired.copy()
        for k in range(1, int(p["release_bars"])):
            recent |= _b(fired.shift(k))
        tr = ind.ema(c, int(p["trend_len"]))
        ok = ind.noise_ok(df)
        long = recent & (bar == 2) & (c > tr) & ok
        short = recent & (bar == -1) & (c < tr) & ok
        if p.get("zero_cross"):
            long |= _cross_up(val, pd.Series(0.0, index=df.index)) & (bar == 2) & (c > tr) & ok & ~on
            short |= _cross_dn(val, pd.Series(0.0, index=df.index)) & (bar == -1) & (c < tr) & ok & ~on
        long &= ~_b(long.shift(1))
        short &= ~_b(short.shift(1))
        mag = (val.abs() / ind.atr(df)).clip(0, 2) / 2
        return self._out(df, long, short, 0.58 + 0.32 * mag + 0.05 * recent)


class SupertrendADXMACD(Strategy):
    key, name = "supertrend_adx", "Supertrend + ADX + MACD"
    modes = ("daily", "long")
    description = "Supertrend direction confirmed by ADX and MACD histogram turning."
    defaults = {"n": 10, "mult": 3.0, "adx_min": 20}
    space = {"*": {"n": [7, 10, 14], "mult": [2.0, 2.5, 3.0, 4.0], "adx_min": [16, 20, 25]}}

    def _compute(self, df, p):
        c = df["close"]
        _, d = ind.supertrend(df, int(p["n"]), float(p["mult"]))
        adx, _, _ = ind.adx(df)
        _, _, h = ind.macd(c)
        flip = d != d.shift()
        ok = ind.noise_ok(df) & (adx > p["adx_min"])
        long = (d == 1) & (h > 0) & (flip | (h.shift() <= 0)) & ok
        short = (d == -1) & (h < 0) & (flip | (h.shift() >= 0)) & ok
        return self._out(df, long, short, 0.5 + (adx.clip(20, 45) - 20) / 50)


class WaveTrendReversal(Strategy):
    key, name = "wavetrend", "WaveTrend Reversal + RSI"
    modes = ("scalp", "daily")
    description = "WaveTrend cross in oversold/overbought zones, confirmed by RSI and candle direction."
    defaults = {"n1": 10, "n2": 21, "level": 53}
    space = {"*": {"n1": [9, 10, 12], "n2": [18, 21, 26], "level": [45, 53, 60]}}

    def _compute(self, df, p):
        c = df["close"]
        wt1, wt2 = ind.wavetrend(df, int(p["n1"]), int(p["n2"]))
        r = ind.rsi(c)
        ok = ind.noise_ok(df)
        lv = float(p["level"])
        long = _cross_up(wt1, wt2) & (wt1 < -lv) & (r < 45) & (c > df["open"]) & ok
        short = _cross_dn(wt1, wt2) & (wt1 > lv) & (r > 55) & (c < df["open"]) & ok
        depth = (wt1.abs().clip(lv, 90) - lv) / 74
        return self._out(df, long, short, 0.55 + depth)


class IchimokuTrend(Strategy):
    key, name = "ichimoku", "Ichimoku Cloud Trend"
    modes = ("daily", "long")
    description = "Tenkan/Kijun cross above/below the cloud with Chikou confirmation."
    defaults = {"set": "9/26/52"}
    presets = {"daily": {"set": "9/26/52"}, "long": {"set": "10/30/60"}}
    space = {"*": {"set": ["9/26/52", "10/30/60", "20/60/120"]}}

    def _compute(self, df, p):
        t_, k_, s_ = (int(x) for x in str(p["set"]).split("/"))
        c = df["close"]
        t, k, a, b = ind.ichimoku(df, t_, k_, s_)
        top, bot = np.maximum(a, b), np.minimum(a, b)
        ok = ind.noise_ok(df)
        long = _cross_up(t, k) & (c > top) & (c > c.shift(k_)) & ok
        short = _cross_dn(t, k) & (c < bot) & (c < c.shift(k_)) & ok
        return self._out(df, long, short, 0.65 + 0.25 * ind.efficiency_ratio(c))


class EMADonchian(Strategy):
    key, name = "ema_donchian", "EMA 50/200 + Donchian Breakout"
    modes = ("daily", "long")
    description = "Turtle-style breakout only in the EMA 50/200 trend direction."
    defaults = {"n": 20, "fast": 50, "slow": 200}
    space = {"*": {"n": [20, 30, 55], "fast": [20, 50], "slow": [100, 200]}}

    def _compute(self, df, p):
        c = df["close"]
        e50, e200 = ind.ema(c, int(p["fast"])), ind.ema(c, int(p["slow"]))
        hi, lo = ind.donchian(df, int(p["n"]))
        ok = ind.noise_ok(df)
        long = (e50 > e200) & (c > hi.shift()) & (c.shift() <= hi.shift(2)) & ok
        short = (e50 < e200) & (c < lo.shift()) & (c.shift() >= lo.shift(2)) & ok
        return self._out(df, long, short, 0.55 + 0.4 * ind.efficiency_ratio(c, 40))


class VWAPPullback(Strategy):
    key, name = "vwap_pullback", "VWAP Pullback"
    modes = ("scalp",)
    description = "Trend pullback that taps session VWAP and rejects it."
    defaults = {"fast": 50, "slow": 200}
    space = {"*": {"fast": [20, 50], "slow": [100, 200]}}

    def _compute(self, df, p):
        c, o = df["close"], df["open"]
        vw = ind.vwap(df)
        e50, e200 = ind.ema(c, int(p["fast"])), ind.ema(c, int(p["slow"]))
        r = ind.rsi(c)
        a = ind.atr(df)
        ok = ind.noise_ok(df)
        long = (e50 > e200) & (df["low"] <= vw + 0.1 * a) & (c > vw) & (c > o) & r.between(40, 62) & ok
        short = (e50 < e200) & (df["high"] >= vw - 0.1 * a) & (c < vw) & (c < o) & r.between(38, 60) & ok
        long &= ~_b(long.shift(1))
        short &= ~_b(short.shift(1))
        return self._out(df, long, short, 0.6)


class SmartMoneyConcepts(Strategy):
    key, name = "smc", "Smart Money Concepts [LuxAlgo]"
    modes = ("scalp", "daily", "long")
    description = ("LuxAlgo SMC logic: internal/swing BOS & CHoCH, order-block and fair-value-gap "
                   "retests, premium/discount filter. Stop goes beyond the order block / swing. "
                   "Swing & internal lengths are auto-tuned per coin/timeframe.")
    defaults = {"swing_len": 50, "internal_len": 5, "choch": True, "bos": True, "ob": True, "fvg": True,
                "zone_filter": True}
    presets = {"scalp": {"swing_len": 30, "internal_len": 5}, "daily": {"swing_len": 50, "internal_len": 5},
               "long": {"swing_len": 50, "internal_len": 8}}
    space = {"*": {"swing_len": [20, 30, 50], "internal_len": [3, 5, 8], "bos": [True, False],
                   "ob": [True, False], "fvg": [True, False], "zone_filter": [True, False]}}

    def _compute(self, df, p):
        c, o = df["close"], df["open"]
        m = cached("smc", df, smc, swing_len=int(p["swing_len"]), internal_len=int(p["internal_len"]))
        a = m["atr"]
        ie, sw, it = m["int_event"], m["swing_trend"], m["int_trend"]
        zp = m["zone_pos"].fillna(0.5)
        zl = (zp < 0.62) if p["zone_filter"] else pd.Series(True, index=df.index)
        zs = (zp > 0.38) if p["zone_filter"] else pd.Series(True, index=df.index)
        ok = ind.noise_ok(df)
        ev_long = ((ie == 2) & p["choch"]) | ((ie == 1) & p["bos"])
        ev_short = ((ie == -2) & p["choch"]) | ((ie == -1) & p["bos"])
        long_a = ev_long & (sw >= 0) & zl
        short_a = ev_short & (sw <= 0) & zs
        ob_l = m["bull_ob_touch"].notna() & (c > o) & (it == 1) & bool(p["ob"])
        ob_s = m["bear_ob_touch"].notna() & (c < o) & (it == -1) & bool(p["ob"])
        fv_l = m["bull_fvg_touch"].notna() & (c > o) & (it == 1) & (sw >= 0) & bool(p["fvg"])
        fv_s = m["bear_fvg_touch"].notna() & (c < o) & (it == -1) & (sw <= 0) & bool(p["fvg"])
        long = (long_a | ob_l | fv_l) & ok
        short = (short_a | ob_s | fv_s) & ok
        long &= ~_b(long.shift(1)) & ~_b(long.shift(2))
        short &= ~_b(short.shift(1)) & ~_b(short.shift(2))
        # stop: beyond the order block / FVG that was retested, else beyond the internal swing
        sl_l = m["bull_ob_touch"].combine_first(m["bull_fvg_touch"]).combine_first(m["int_low"]) - 0.2 * a
        sl_s = m["bear_ob_touch"].combine_first(m["bear_fvg_touch"]).combine_first(m["int_high"]) + 0.2 * a
        sl = pd.Series(np.where(long, sl_l, np.where(short, sl_s, np.nan)), index=df.index)
        strength = (0.6 + 0.1 * (ie.abs() == 2) + 0.1 * ((sw == 1) & long | (sw == -1) & short)
                    + 0.1 * ((ob_l | fv_l) & long_a | (ob_s | fv_s) & short_a)
                    + 0.05 * ((zp < 0.45) & long | (zp > 0.55) & short))
        return self._out(df, long, short, strength, sl)


class LuxSignalsOverlays(Strategy):
    key, name = "lux_signals", "Signals & Overlays (LuxAlgo-style)"
    modes = ("scalp", "daily", "long")
    description = ("Confirmation signals (normal + strong) with Smart Trail stop, Trend Catcher, "
                   "Trend Tracer, Neo Cloud and Reversal Zones. Clean-room version of LuxAlgo "
                   "Signals & Overlays (the original is closed-source). Sensitivity auto-tuned.")
    defaults = {"sensitivity": 6.0, "atr_len": 11, "strong_only": False, "contrarian": False,
                "trail_mult": 3.0}
    presets = {"scalp": {"sensitivity": 7.0, "atr_len": 10}, "daily": {"sensitivity": 6.0},
               "long": {"sensitivity": 5.0, "atr_len": 14}}
    space = {"*": {"sensitivity": [3.0, 4.5, 6.0, 7.5, 9.0], "atr_len": [10, 11, 14],
                   "strong_only": [False, True], "contrarian": [False, True], "trail_mult": [2.5, 3.0, 3.5]}}

    def _compute(self, df, p):
        c = df["close"]
        x = cached("lux", df, signals_overlays, sensitivity=float(p["sensitivity"]), atr_len=int(p["atr_len"]),
                   trail_mult=float(p["trail_mult"]))
        ok = ind.noise_ok(df)
        if p["strong_only"]:
            long, short = x["strong_buy"], x["strong_sell"]
        else:
            long, short = x["buy"], x["sell"]
        long = long & ok
        short = short & ok
        if p["contrarian"]:
            long |= x["contra_buy"] & ok & (x["cloud_dir"] >= 0)
            short |= x["contra_sell"] & ok & (x["cloud_dir"] <= 0)
        a = ind.atr(df)
        # stop: Smart Trail when it sits on the right side, else the confirmation line
        sl_l = np.where(x["trail"] < c, x["trail"], x["st_line"]) - 0.1 * a
        sl_s = np.where(x["trail"] > c, x["trail"], x["st_line"]) + 0.1 * a
        sl = pd.Series(np.where(long, sl_l, np.where(short, sl_s, np.nan)), index=df.index)
        strong = (x["strong_buy"] & long) | (x["strong_sell"] & short)
        strength = 0.6 + 0.18 * strong + 0.12 * (x["strength"] / 100)
        return self._out(df, long, short, strength, sl)


class SP2LStrategy(Strategy):
    key, name = "sp2l", "SP2L Spike + 2 Legs (Poursamadi)"
    modes = ("scalp", "daily", "long")
    description = ("Spike of consecutive higher lows / lower highs (with FVG + doji filters), entry on the "
                   "retest of a spike HL/LH, stop beyond the spike origin, targets 1R/2R/3R. "
                   "Spike length/power auto-tuned per coin/timeframe.")
    defaults = {"min_bars": 3, "power": 2.0, "gap_filter": True, "expiry": 12, "doji_body": 0.25,
                "trend_filter": True}
    presets = {"scalp": {"min_bars": 3, "power": 2.0, "expiry": 10}, "daily": {"min_bars": 3, "power": 2.0},
               "long": {"min_bars": 3, "power": 1.6, "expiry": 8}}
    space = {"*": {"min_bars": [3, 4, 5], "power": [1.5, 2.0, 2.5, 3.0], "gap_filter": [True, False],
                   "expiry": [6, 12, 20], "trend_filter": [True, False]}}

    def _compute(self, df, p):
        r = cached("sp2l", df, sp2l, min_bars=int(p["min_bars"]), power=float(p["power"]),
                   gap_filter=bool(p["gap_filter"]), expiry=int(p["expiry"]),
                   doji_body=float(p["doji_body"]), trend_filter=bool(p["trend_filter"]))
        ok = ind.noise_ok(df)
        return self._out(df, r["long"] & ok, r["short"] & ok, r["strength"], r["sl"])


class AIConfluence(Strategy):
    key, name = "ai_confluence", "AI Confluence (all indicators + ML)"
    modes = ("scalp", "daily", "long")
    description = ("Weighted vote of 15 trend/momentum/Algo indicators (now incl. SMC structure, "
                   "Signals & Overlays trend + Smart Trail and LazyBear squeeze), regime-filtered by ADX "
                   "and efficiency ratio. Threshold auto-tuned. Pair with the ML filter.")
    defaults = {"threshold": 0.55, "adx_min": 18, "er_min": 0.2, "smc_len": 50, "lux_sens": 6.0}
    presets = {"scalp": {"threshold": 0.5, "adx_min": 16, "smc_len": 30, "lux_sens": 7.0},
               "long": {"threshold": 0.55, "lux_sens": 5.0}}
    space = {"*": {"threshold": [0.4, 0.45, 0.5, 0.55, 0.6], "adx_min": [14, 18, 22], "er_min": [0.1, 0.2, 0.3]}}
    threshold = 0.55

    def votes(self, df, p=None) -> pd.Series:
        p = self.params_for(None, p)
        c = df["close"]
        _, st_dir = ind.supertrend(df)
        ut_stop, _, _ = ind.ut_bot(df, 1.5, 10)
        _, rf_dir = ind.range_filter(c, 50, 2.5)
        e20, e50, e200 = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
        _, _, hist = ind.macd(c)
        r = ind.rsi(c)
        wt1, wt2 = ind.wavetrend(df)
        _, _, sa, sb = ind.ichimoku(df)
        q = cached("sqz", df, ind.squeeze_lazybear, length=20, mult=2.0, length_kc=20, mult_kc=1.5)
        sq = q["val"]
        kal = ind.kalman_price(c)
        h = ind.hma(c, 55)
        m = cached("smc", df, smc, swing_len=int(p["smc_len"]), internal_len=5)
        x = cached("lux", df, signals_overlays, sensitivity=float(p["lux_sens"]), atr_len=11, trail_mult=3.0)
        stack = np.sign(e20 - e50) * 0.5 + np.sign(e50 - e200) * 0.5
        cloud = np.where(c > np.maximum(sa, sb), 1, np.where(c < np.minimum(sa, sb), -1, 0))
        parts = [
            (st_dir, 1.0), (np.sign(c - ut_stop), 1.0), (rf_dir, 1.0), (stack, 1.0),
            (np.sign(hist), 0.7), (np.sign(r - 50), 0.5), (np.sign(wt1 - wt2), 0.5),
            (pd.Series(cloud, index=df.index), 0.8),
            (np.sign(sq) * (np.sign(sq.diff()) == np.sign(sq)), 0.7),
            (np.sign(ind.slope(kal, 3)), 1.0), (np.sign(h.diff()), 0.6),
            (m["int_trend"], 0.8), (m["swing_trend"], 0.6),
            (x["st_dir"], 0.9), (x["trail_dir"], 0.6),
        ]
        total = sum(w for _, w in parts)
        return sum(pd.Series(v, index=df.index).astype(float).fillna(0) * w for v, w in parts) / total

    def _compute(self, df, p):
        c = df["close"]
        score = self.votes(df, p)
        adx, _, _ = ind.adx(df)
        er = ind.efficiency_ratio(c)
        regime = (adx > p["adx_min"]) & (er > p["er_min"])
        ok = ind.noise_ok(df) & regime
        thr = float(p["threshold"])
        long = (score >= thr) & (score.shift() < thr) & ok
        short = (score <= -thr) & (score.shift() > -thr) & ok
        # map |score| in [thr, 1] to [0.6, 0.9] (+ up to 0.1 for a clean trend)
        over = ((score.abs() - thr) / (1 - thr)).clip(0, 1)
        strength = 0.6 + 0.3 * over + 0.1 * er.clip(0, 1)
        return self._out(df, long, short, strength)


class AutoPilot(Strategy):
    """Runs every strategy of the trading style with its auto-tuned settings and takes the best
    signal on each bar. Members whose tuning failed out-of-sample are switched off by the tuner."""
    key, name = "auto_pilot", "Auto Pilot (all strategies, auto-tuned)"
    modes = ("scalp", "daily", "long")
    description = ("Scans with ALL strategies of your style at once (SMC, SP2L, Signals & Overlays, "
                   "Squeeze LazyBear, AI Confluence, UT Bot...), each with settings auto-tuned on the "
                   "coin's own chart. Only strategies that passed an out-of-sample test can trade. "
                   "Several agreeing strategies raise the confidence. Most trades with quality control.")
    defaults = {}

    def _compute(self, df, p):
        mode = p.get("mode", "daily")
        members = p.get("members") or {}
        enabled = p.get("enabled")
        bonus = p.get("bonus") or {}
        keys = [s.key for s in strategies_for_mode(mode) if s.key != self.key]
        if enabled is not None:
            keys = [k for k in keys if k in enabled]
        n = len(df)
        best_l = np.zeros(n)
        best_s = np.zeros(n)
        sl_l = np.full(n, np.nan)
        sl_s = np.full(n, np.nan)
        src_l = np.array([""] * n, dtype=object)
        src_s = np.array([""] * n, dtype=object)
        cnt_l = np.zeros(n)
        cnt_s = np.zeros(n)
        for k in keys:
            st = STRATEGIES[k]
            try:
                o = st.compute(df, st.params_for(mode, members.get(k)))
            except Exception:       # one broken member must not kill the scan
                continue
            b = float(bonus.get(k, 0.0))
            sl_ = o["strength"].values + b
            L, S = o["long"].values, o["short"].values
            cnt_l += L
            cnt_s += S
            up = L & (sl_ > best_l)
            best_l = np.where(up, sl_, best_l)
            sl_l = np.where(up, o["sl"].values, sl_l)
            src_l = np.where(up, k, src_l)
            dn = S & (sl_ > best_s)
            best_s = np.where(dn, sl_, best_s)
            sl_s = np.where(dn, o["sl"].values, sl_s)
            src_s = np.where(dn, k, src_s)
        long = (cnt_l > 0) & (cnt_s == 0)
        short = (cnt_s > 0) & (cnt_l == 0)
        strength = np.where(long, best_l + 0.05 * (cnt_l - 1), np.where(short, best_s + 0.05 * (cnt_s - 1), 0))
        out = self._out(df, long, short, strength, np.where(long, sl_l, np.where(short, sl_s, np.nan)))
        out["source"] = np.where(long, src_l, np.where(short, src_s, ""))
        out["agree"] = np.where(long, cnt_l, np.where(short, cnt_s, 0)).astype(int)
        return out


STRATEGIES = {s.key: s for s in (AutoPilot(), AIConfluence(), SmartMoneyConcepts(), SP2LStrategy(),
                                 LuxSignalsOverlays(), SqueezeLazyBear(), UTBotTrend(), RangeFilterBreakout(),
                                 SupertrendADXMACD(), WaveTrendReversal(), IchimokuTrend(),
                                 EMADonchian(), VWAPPullback())}

TUNABLE = [k for k, s in STRATEGIES.items() if s.space]


def strategies_for_mode(mode: str):
    return [s for s in STRATEGIES.values() if mode in s.modes]
