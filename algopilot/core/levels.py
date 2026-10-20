"""Support/resistance detection and automatic SL / TP selection."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_flags(df: pd.DataFrame, left: int = 5, right: int = 5):
    """Boolean arrays marking swing highs / lows. A pivot at bar j is only
    *confirmed* at bar j + right (no look-ahead when used causally)."""
    w = left + right + 1
    hi = df["high"].rolling(w, center=True).max()
    lo = df["low"].rolling(w, center=True).min()
    ph = (df["high"] == hi).values
    pl = (df["low"] == lo).values
    return ph, pl


def sr_levels(highs: np.ndarray, lows: np.ndarray, atr_val: float, tol_mult: float = 0.5):
    """Cluster pivot prices into levels. Returns list of (price, touches)."""
    pts = np.sort(np.concatenate([highs, lows]))
    if len(pts) == 0 or not np.isfinite(atr_val) or atr_val <= 0:
        return []
    tol = atr_val * tol_mult
    levels, cluster = [], [pts[0]]
    for p in pts[1:]:
        if p - cluster[-1] <= tol:
            cluster.append(p)
        else:
            levels.append((float(np.mean(cluster)), len(cluster)))
            cluster = [p]
    levels.append((float(np.mean(cluster)), len(cluster)))
    return levels


def levels_at(df: pd.DataFrame, i: int, ph, pl, atr_val: float, lookback: int = 300, right: int = 5):
    """Causal S/R levels as known at bar i."""
    end = max(0, i - right + 1)
    start = max(0, end - lookback)
    h = df["high"].values[start:end][ph[start:end]]
    l = df["low"].values[start:end][pl[start:end]]
    return sr_levels(h, l, atr_val)


def choose_sl(df: pd.DataFrame, i: int, side: str, entry: float, atr_val: float, pl, ph,
              swing_lookback: int = 30, right: int = 5, default_mult: float = 1.5) -> tuple:
    """Stop beyond the most recent swing, clamped to 0.8..3 ATR."""
    end = max(0, i - right + 1)
    start = max(0, end - swing_lookback)
    reason = "ATR stop"
    if side == "long":
        lows = df["low"].values[start:end][pl[start:end]]
        lows = lows[lows < entry]
        if len(lows):
            sl, reason = float(lows[-1]) - 0.2 * atr_val, "below swing low"
        else:
            sl = entry - default_mult * atr_val
        raw = entry - sl
        dist = min(max(raw, 0.8 * atr_val), 3.0 * atr_val)
        if dist != raw:
            reason = "ATR stop (swing too far)" if raw > dist else "min 0.8 ATR stop"
        return entry - dist, reason
    highs = df["high"].values[start:end][ph[start:end]]
    highs = highs[highs > entry]
    if len(highs):
        sl, reason = float(highs[-1]) + 0.2 * atr_val, "above swing high"
    else:
        sl = entry + default_mult * atr_val
    raw = sl - entry
    dist = min(max(raw, 0.8 * atr_val), 3.0 * atr_val)
    if dist != raw:
        reason = "ATR stop (swing too far)" if raw > dist else "min 0.8 ATR stop"
    return entry + dist, reason


def choose_tps(side: str, entry: float, sl: float, levels: list, atr_val: float,
               ladder=(1.0, 2.0, 3.0)) -> tuple:
    """Pick 3 take-profit levels. Prefers real S/R levels (front-run slightly),
    falls back to R-multiples when no level fits the ladder."""
    risk = abs(entry - sl)
    sgn = 1 if side == "long" else -1
    cands = []
    for price, touches in levels:
        target = price - sgn * 0.1 * atr_val          # exit just before the level
        rr = sgn * (target - entry) / risk if risk > 0 else 0
        if rr > 0:
            cands.append((rr, target, touches))
    cands.sort()
    tps, notes, last_rr = [], [], 0.0
    for want in ladder:
        pick = None
        for rr, target, touches in cands:
            if rr > last_rr + 0.3 and want * 0.85 <= rr <= want * 1.6:
                pick = (rr, target, touches)
                break
        if pick:
            tps.append(pick[1])
            notes.append(f"S/R {pick[1]:.6g} ({pick[2]} touches)")
            last_rr = pick[0]
        else:
            rr = max(want, last_rr + 0.5)
            tps.append(entry + sgn * rr * risk)
            notes.append(f"{rr:.1f}R")
            last_rr = rr
    return tps, notes
