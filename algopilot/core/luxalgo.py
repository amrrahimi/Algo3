"""Signals & Overlays toolkit, modelled on "LuxAlgo Signals & Overlays".

IMPORTANT: the original LuxAlgo Signals & Overlays is a paid, closed-source (invite-only) script,
so its exact formulas are not public. This is a clean-room re-implementation of the documented
behaviour of each component, built from standard, well-understood building blocks:

  Confirmation signals  ATR trend flips with a Sensitivity input (higher = more signals);
                        "strong" (+) when the flip agrees with the trend cloud and trend strength.
  Contrarian signals    exhaustion: price closes back inside after piercing an outer reversal zone.
  Smart Trail           adaptive ATR trailing stop (direction + level), used as dynamic stop.
  Trend Catcher         fast low-lag trend line (Hull MA), coloured by slope.
  Trend Tracer          slower trend line (range midpoint).
  Neo Cloud             long-term trend cloud (two-line cloud, causal).
  Reversal Zones        inner/outer volatility bands where reversals are likely.
  Trend Strength        0..100 (ADX based).

Signals will NOT match LuxAlgo's arrows bar-for-bar; they follow the same ideas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _trail(df: pd.DataFrame, n: int, mult: float):
    """ATR trailing stop with direction (+1 long / -1 short)."""
    a = (ind.atr(df, n) * mult).values
    c = df["close"].values
    h, l = df["high"].values, df["low"].values
    line = np.full(len(c), np.nan)
    dirn = np.zeros(len(c))
    d, lv = 1, np.nan
    for i in range(len(c)):
        if not np.isfinite(a[i]):
            continue
        if np.isnan(lv):
            lv, d = c[i] - a[i], 1
        if d == 1:
            lv = max(lv, h[i] - a[i])
            if c[i] < lv:
                d, lv = -1, l[i] + a[i]
        else:
            lv = min(lv, l[i] + a[i])
            if c[i] > lv:
                d, lv = 1, h[i] - a[i]
        line[i], dirn[i] = lv, d
    return pd.Series(line, index=df.index), pd.Series(dirn, index=df.index)


def signals_overlays(df: pd.DataFrame, sensitivity: float = 6.0, atr_len: int = 11,
                     trail_len: int = 22, trail_mult: float = 3.0, catcher_len: int = 24,
                     tracer_len: int = 55, cloud_fast: int = 26, cloud_slow: int = 52,
                     rz_len: int = 50, rz_inner: float = 2.0, rz_outer: float = 3.2,
                     strong_adx: float = 20.0) -> dict:
    c = df["close"]
    sens = float(np.clip(sensitivity, 1, 10))
    factor = 1.0 + (10.0 - sens) * 0.45                     # sens 10 -> 1.0x ATR, sens 1 -> 5.05x
    st_line, st_dir = ind.supertrend(df, atr_len, factor)
    flip_up = (st_dir == 1) & (st_dir.shift() == -1)
    flip_dn = (st_dir == -1) & (st_dir.shift() == 1)

    trail, trail_dir = _trail(df, trail_len, trail_mult)
    catcher = ind.hma(c, catcher_len)
    catcher_dir = np.sign(catcher.diff()).fillna(0)
    hh = df["high"].rolling(tracer_len, min_periods=tracer_len).max()
    ll = df["low"].rolling(tracer_len, min_periods=tracer_len).min()
    tracer = (hh + ll) / 2
    mid = lambda k: (df["high"].rolling(k, min_periods=k).max() + df["low"].rolling(k, min_periods=k).min()) / 2
    cloud_a = (mid(cloud_fast) + mid(cloud_fast // 3 or 1)) / 2
    cloud_b = mid(cloud_slow)
    cloud_dir = pd.Series(np.where(cloud_a > cloud_b, 1, np.where(cloud_a < cloud_b, -1, 0)), index=df.index)

    basis = ind.ema(ind.hlc3(df), rz_len)
    width = ind.rma(ind.true_range(df), rz_len)
    rz = {"basis": basis, "up_in": basis + rz_inner * width, "up_out": basis + rz_outer * width,
          "dn_in": basis - rz_inner * width, "dn_out": basis - rz_outer * width}

    adx, _, _ = ind.adx(df)
    strength = adx.clip(0, 60) / 60 * 100

    trend_ok_up = (c > tracer) & (cloud_dir >= 0)
    trend_ok_dn = (c < tracer) & (cloud_dir <= 0)
    strong_up = flip_up & trend_ok_up & (adx > strong_adx) & (catcher_dir > 0)
    strong_dn = flip_dn & trend_ok_dn & (adx > strong_adx) & (catcher_dir < 0)
    # contrarian: wick through the outer zone and close back inside the inner band
    contra_up = (df["low"] < rz["dn_out"]) & (c > rz["dn_in"]) & (c > df["open"])
    contra_dn = (df["high"] > rz["up_out"]) & (c < rz["up_in"]) & (c < df["open"])

    f = lambda s: s.fillna(False).astype(bool)
    return {"buy": f(flip_up), "sell": f(flip_dn), "strong_buy": f(strong_up), "strong_sell": f(strong_dn),
            "contra_buy": f(contra_up), "contra_sell": f(contra_dn),
            "st_line": st_line, "st_dir": st_dir, "trail": trail, "trail_dir": trail_dir,
            "catcher": catcher, "catcher_dir": catcher_dir, "tracer": tracer,
            "cloud_a": cloud_a, "cloud_b": cloud_b, "cloud_dir": cloud_dir,
            "rz": rz, "strength": strength, "adx": adx}
