"""SP2L (Spike - 2 Legs) price-action setup by Mohammad Ali Poursamadi.

Rules (from the published description of the strategy / TradingFinder indicator):
  1. Spike: >= `min_bars` consecutive candles in one direction making Higher Lows (bullish) or
     Lower Highs (bearish), with enough "movement power" (net move in ATRs), a limited share of
     doji candles, and (optional gap filter) at least one Fair Value Gap inside the spike.
  2. Each Higher Low (Lower High) of the spike is a potential entry level.
  3. Entry triggers when price comes back and retests one of those levels and holds it.
  4. Stop loss beyond the candle where the spike originated (+ a small threshold).
  5. Targets as R multiples (1:1, 1:2, more in strong trends), handled by the trade planner.
  Optional trend filter: share of dojis in the last `trend_lookback` candles must be low.

Causal: the signal at bar i only uses bars <= i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def sp2l(df: pd.DataFrame, min_bars: int = 3, power: float = 2.0, doji_body: float = 0.25,
         max_doji_spike: float = 0.34, gap_filter: bool = True, sl_threshold: float = 0.15,
         expiry: int = 12, hold_tol: float = 0.15, trend_filter: bool = True,
         trend_lookback: int = 20, max_doji_trend: float = 0.45) -> dict:
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    a = ind.atr(df, 14).bfill().values
    rng = np.maximum(h - l, 1e-12)
    doji = np.abs(c - o) / rng <= doji_body
    doji_share = pd.Series(doji.astype(float)).rolling(trend_lookback, min_periods=5).mean().values

    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)
    sl = np.full(n, np.nan)
    strength = np.zeros(n)
    setups = []             # for the chart: (x_origin, x_end, origin_px, levels, bull, triggered_at)

    def scan(bull: bool):
        sgn = 1 if bull else -1
        run_start, run_len = -1, 0
        active = None       # dict(levels, origin, end, power, x0)
        for i in range(1, n):
            cont = (l[i] > l[i - 1] and (c[i] > o[i] or doji[i])) if bull else \
                   (h[i] < h[i - 1] and (c[i] < o[i] or doji[i]))
            # ---- check for an entry on an existing (finished) spike first
            if active is not None and not cont:
                lv = np.array(active["levels"])
                if bull:
                    touched = lv[lv >= l[i]]
                    broke = c[i] < active["origin"]
                else:
                    touched = lv[lv <= h[i]]
                    broke = c[i] > active["origin"]
                if broke or i - active["end"] > expiry:
                    active = None
                elif len(touched):
                    deep = touched.min() if bull else touched.max()
                    held = (c[i] >= deep - hold_tol * a[i]) if bull else (c[i] <= deep + hold_tol * a[i])
                    trend_ok = (not trend_filter) or not (doji_share[i] > max_doji_trend)
                    if held and trend_ok:
                        (long_sig if bull else short_sig)[i] = True
                        sl[i] = active["origin"] - sgn * sl_threshold * a[i]
                        strength[i] = 0.6 + 0.25 * min(1.0, (active["power"] - power) / max(power, 1e-9)) \
                            + (0.1 if len(touched) == 1 else 0.0)        # first HL retest = cleanest
                        setups.append((active["x0"], i, active["origin"], list(active["levels"]), bull, i))
                        active = None
            # ---- spike tracking
            if cont:
                if run_len == 0:
                    run_start = i
                run_len += 1
            else:
                run_len = 0
            if run_len >= min_bars:
                s0 = run_start
                idx = np.arange(s0, i + 1)
                origin = min(l[s0 - 1], l[s0]) if bull else max(h[s0 - 1], h[s0])
                move = (c[i] - origin) * sgn / max(a[i], 1e-12)
                dj = doji[idx].mean()
                gap = True
                if gap_filter:
                    gap = False
                    for j in range(max(s0, 2), i + 1):
                        if (bull and l[j] > h[j - 2]) or (not bull and h[j] < l[j - 2]):
                            gap = True
                            break
                if move >= power and dj <= max_doji_spike and gap:
                    levels = list(l[idx]) if bull else list(h[idx])
                    active = {"levels": levels, "origin": float(origin), "end": i,
                              "power": float(move), "x0": int(s0 - 1)}
        return

    scan(True)
    scan(False)
    strength = np.clip(strength, 0, 1)
    return {"long": pd.Series(long_sig, index=df.index), "short": pd.Series(short_sig, index=df.index),
            "sl": pd.Series(sl, index=df.index), "strength": pd.Series(strength, index=df.index),
            "setups": setups[-12:]}
