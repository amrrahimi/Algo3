"""Smart Money Concepts, a Python port of the logic of "Smart Money Concepts [LuxAlgo]".

Everything here is causal: a value at bar i only uses bars <= i, so it is safe for the robot
and the backtester. Implemented parts of the original script:
  * swing + internal market structure (BOS / CHoCH), Lux "os" swing detection
  * order blocks (swing + internal) with the high-volatility bar filter and mitigation
  * fair value gaps with the "auto threshold" filter and mitigation
  * equal highs / equal lows
  * premium / equilibrium / discount zones and strong / weak highs and lows

`smc(df, ...)` returns per-bar arrays (used by strategies) plus an `overlay` dict with the
currently active objects (used by the chart).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class _Box:
    top: float
    bottom: float
    x: int            # bar where the object starts
    bull: bool
    created: int      # bar where it became known (no look-ahead)
    kind: str = ""


def _os_swings(high: np.ndarray, low: np.ndarray, length: int):
    """Lux swing detection. At bar i a top/bottom located at bar i-length gets confirmed.
    Returns arrays top_px, btm_px (NaN when nothing is confirmed at i)."""
    n = len(high)
    top_px = np.full(n, np.nan)
    btm_px = np.full(n, np.nan)
    if n <= length:
        return top_px, btm_px
    up = pd.Series(high).rolling(length, min_periods=length).max().values
    lo = pd.Series(low).rolling(length, min_periods=length).min().values
    os_ = 0
    for i in range(length, n):
        prev = os_
        if high[i - length] > up[i]:
            os_ = 0
        elif low[i - length] < lo[i]:
            os_ = 1
        if os_ == 0 and prev != 0:
            top_px[i] = high[i - length]
        if os_ == 1 and prev != 1:
            btm_px[i] = low[i - length]
    return top_px, btm_px


def smc(df: pd.DataFrame, swing_len: int = 50, internal_len: int = 5, eq_len: int = 3,
        eq_threshold: float = 0.1, fvg_auto: bool = True, max_obs: int = 5, max_fvgs: int = 10,
        ob_mitigation: str = "close") -> dict:
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    a200 = ind.atr(df, 200).bfill().values
    a14 = ind.atr(df, 14).bfill().values
    # high-volatility bars: Lux swaps their high/low when looking for order blocks
    hv = (h - l) >= 2 * a200
    ph = np.where(hv, l, h)
    pl = np.where(hv, h, l)

    s_top, s_btm = _os_swings(h, l, swing_len)
    i_top, i_btm = _os_swings(h, l, internal_len)
    e_top, e_btm = _os_swings(h, l, eq_len)

    out = {k: np.zeros(n, dtype=int) for k in ("swing_trend", "int_trend", "swing_event", "int_event")}
    for k in ("bull_ob_touch", "bear_ob_touch", "bull_fvg_touch", "bear_fvg_touch"):
        out[k] = np.full(n, np.nan)       # SL anchor (OB / FVG far edge) when touched, else NaN
    out["zone_pos"] = np.full(n, np.nan)  # 0 = bottom of range (discount) .. 1 = top (premium)
    out["int_low"] = np.full(n, np.nan)   # last internal swing low / high (for stops)
    out["int_high"] = np.full(n, np.nan)

    events = []                           # (bar, from_x, level, "BOS"/"CHoCH", bull, "swing"/"internal")
    eqs = []                              # (x1, x2, level, "EQH"/"EQL")
    obs: list[_Box] = []
    fvgs: list[_Box] = []

    # structure state, one dict per level
    st = {lvl: {"top": np.nan, "top_x": 0, "top_cross": True, "btm": np.nan, "btm_x": 0,
                "btm_cross": True, "trend": 0} for lvl in ("swing", "internal")}
    eq_top, eq_top_x, eq_btm, eq_btm_x = np.nan, 0, np.nan, 0
    trail_up, trail_dn = np.nan, np.nan
    trail_up_x = trail_dn_x = 0
    delta_cum, delta_n = 0.0, 0

    for i in range(n):
        # ---- confirm new swing points (pivot sits at i - len)
        for lvl, tops, btms, ln in (("swing", s_top, s_btm, swing_len), ("internal", i_top, i_btm, internal_len)):
            s = st[lvl]
            if np.isfinite(tops[i]):
                s["top"], s["top_x"], s["top_cross"] = tops[i], i - ln, False
                if lvl == "swing":
                    trail_up, trail_up_x = tops[i], i - ln
            if np.isfinite(btms[i]):
                s["btm"], s["btm_x"], s["btm_cross"] = btms[i], i - ln, False
                if lvl == "swing":
                    trail_dn, trail_dn_x = btms[i], i - ln
        if np.isfinite(e_top[i]):
            if np.isfinite(eq_top) and abs(e_top[i] - eq_top) < a200[i] * eq_threshold:
                eqs.append((eq_top_x, i - eq_len, float(e_top[i]), "EQH"))
            eq_top, eq_top_x = e_top[i], i - eq_len
        if np.isfinite(e_btm[i]):
            if np.isfinite(eq_btm) and abs(e_btm[i] - eq_btm) < a200[i] * eq_threshold:
                eqs.append((eq_btm_x, i - eq_len, float(e_btm[i]), "EQL"))
            eq_btm, eq_btm_x = e_btm[i], i - eq_len

        # ---- trailing extremes for premium / discount
        if np.isfinite(trail_up) and h[i] > trail_up:
            trail_up, trail_up_x = h[i], i
        if np.isfinite(trail_dn) and l[i] < trail_dn:
            trail_dn, trail_dn_x = l[i], i
        if np.isfinite(trail_up) and np.isfinite(trail_dn) and trail_up > trail_dn:
            out["zone_pos"][i] = (c[i] - trail_dn) / (trail_up - trail_dn)

        # ---- structure breaks (internal first, then swing, like the original)
        for lvl in ("internal", "swing"):
            s = st[lvl]
            other = st["swing"]
            if np.isfinite(s["top"]) and not s["top_cross"] and c[i] > s["top"] and \
                    (lvl == "swing" or s["top"] != other["top"]):
                kind = "CHoCH" if s["trend"] < 0 else "BOS"
                s["trend"], s["top_cross"] = 1, True
                out[f"{'swing' if lvl == 'swing' else 'int'}_event"][i] = 2 if kind == "CHoCH" else 1
                events.append((i, s["top_x"], float(s["top"]), kind, True, lvl))
                lo_x = s["top_x"]
                if i - lo_x >= 1:
                    k = lo_x + int(np.argmin(pl[lo_x:i]))
                    obs.append(_Box(float(ph[k]), float(pl[k]), k, True, i, lvl))
            if np.isfinite(s["btm"]) and not s["btm_cross"] and c[i] < s["btm"] and \
                    (lvl == "swing" or s["btm"] != other["btm"]):
                kind = "CHoCH" if s["trend"] > 0 else "BOS"
                s["trend"], s["btm_cross"] = -1, True
                out[f"{'swing' if lvl == 'swing' else 'int'}_event"][i] = -2 if kind == "CHoCH" else -1
                events.append((i, s["btm_x"], float(s["btm"]), kind, False, lvl))
                lo_x = s["btm_x"]
                if i - lo_x >= 1:
                    k = lo_x + int(np.argmax(ph[lo_x:i]))
                    obs.append(_Box(float(ph[k]), float(pl[k]), k, False, i, lvl))
        out["swing_trend"][i] = st["swing"]["trend"]
        out["int_trend"][i] = st["internal"]["trend"]
        out["int_low"][i] = st["internal"]["btm"]
        out["int_high"][i] = st["internal"]["top"]

        # ---- fair value gaps (bar i completes the 3-candle pattern)
        if i >= 2 and o[i - 1] > 0:
            dp = (c[i - 1] - o[i - 1]) / o[i - 1] * 100
            delta_cum += abs(dp)
            delta_n += 1
            thr = (delta_cum / delta_n * 2) if fvg_auto else 0.0
            if l[i] > h[i - 2] and c[i - 1] > h[i - 2] and dp > thr:
                fvgs.append(_Box(float(l[i]), float(h[i - 2]), i - 2, True, i, "fvg"))
            if h[i] < l[i - 2] and c[i - 1] < l[i - 2] and -dp > thr:
                fvgs.append(_Box(float(l[i - 2]), float(h[i]), i - 2, False, i, "fvg"))

        # ---- touches (retests) of objects known BEFORE this bar, then mitigation
        keep = []
        for b in obs:
            if b.created >= i:
                keep.append(b)
                continue
            if b.bull:
                if l[i] <= b.top and c[i] >= b.bottom:
                    prev = out["bull_ob_touch"][i]
                    out["bull_ob_touch"][i] = b.bottom if np.isnan(prev) else min(prev, b.bottom)
                gone = (c[i] if ob_mitigation == "close" else l[i]) < b.bottom
            else:
                if h[i] >= b.bottom and c[i] <= b.top:
                    prev = out["bear_ob_touch"][i]
                    out["bear_ob_touch"][i] = b.top if np.isnan(prev) else max(prev, b.top)
                gone = (c[i] if ob_mitigation == "close" else h[i]) > b.top
            if not gone:
                keep.append(b)
        obs = keep[-4 * max_obs:]
        keep = []
        for b in fvgs:
            if b.created >= i:
                keep.append(b)
                continue
            if b.bull:
                if l[i] <= b.top and c[i] >= b.bottom:
                    out["bull_fvg_touch"][i] = b.bottom
                gone = l[i] < b.bottom
            else:
                if h[i] >= b.bottom and c[i] <= b.top:
                    out["bear_fvg_touch"][i] = b.top
                gone = h[i] > b.top
            if not gone:
                keep.append(b)
        fvgs = keep[-4 * max_fvgs:]

    for k in ("bull_ob_touch", "bear_ob_touch", "bull_fvg_touch", "bear_fvg_touch"):
        out[k] = pd.Series(out[k], index=df.index)
    for k in ("swing_trend", "int_trend", "swing_event", "int_event"):
        out[k] = pd.Series(out[k], index=df.index)
    for k in ("zone_pos", "int_low", "int_high"):
        out[k] = pd.Series(out[k], index=df.index)

    zones = None
    if np.isfinite(trail_up) and np.isfinite(trail_dn) and trail_up > trail_dn:
        x0 = min(trail_up_x, trail_dn_x)
        zones = {"x": int(x0), "top": float(trail_up), "bottom": float(trail_dn),
                 "premium": (float(trail_up), float(0.95 * trail_up + 0.05 * trail_dn)),
                 "equilibrium": (float(0.525 * trail_up + 0.475 * trail_dn),
                                 float(0.525 * trail_dn + 0.475 * trail_up)),
                 "discount": (float(0.95 * trail_dn + 0.05 * trail_up), float(trail_dn)),
                 "strong_high": st["swing"]["trend"] < 0, "strong_low": st["swing"]["trend"] > 0}
    int_obs = [b for b in obs if b.kind == "internal"][-max_obs:]
    sw_obs = [b for b in obs if b.kind == "swing"][-max_obs:]
    out["overlay"] = {
        "events": events[-30:],
        "obs": [(b.x, b.top, b.bottom, b.bull, b.kind) for b in int_obs + sw_obs],
        "fvgs": [(b.x, b.top, b.bottom, b.bull) for b in fvgs[-max_fvgs:]],
        "eqs": eqs[-10:],
        "zones": zones,
    }
    out["atr"] = pd.Series(a14, index=df.index)
    return out
