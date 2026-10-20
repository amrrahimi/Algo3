"""Turns raw strategy triggers into full trade plans (entry / SL / TP1-3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .levels import choose_sl, choose_tps, levels_at, pivot_flags
from .signals import Signal
from .strategies import STRATEGIES


# How much the signal-frequency setting shifts the confidence gate
FREQUENCY_SHIFT = {"high": -0.07, "balanced": 0.0, "quality": 0.07}


class Analyzer:
    def __init__(self, strategy_key: str, ml_filter=None, use_ml: bool = True, mode: str = "daily",
                 params: dict | None = None):
        self.strategy = STRATEGIES.get(strategy_key, STRATEGIES["auto_pilot"])
        self.ml = ml_filter
        self.use_ml = use_ml and ml_filter is not None
        self.mode = mode
        self.params = params        # tuned params (None -> style preset)

    def strategy_params(self, params=None) -> dict:
        p = params if params is not None else self.params
        if self.strategy.key == "auto_pilot":
            p = dict(p or {})
            p.setdefault("mode", self.mode)
            return p
        return self.strategy.params_for(self.mode, p)

    def prepare(self, df: pd.DataFrame, params=None):
        sig = self.strategy.compute(df, self.strategy_params(params))
        a = ind.atr(df, 14)
        ph, pl = pivot_flags(df)
        return sig, a, ph, pl

    def plan(self, df, i, side, strength, a, ph, pl, symbol="", timeframe="", strat_sl=None,
             source="") -> Signal | None:
        atr_val = float(a.iloc[i])
        if not np.isfinite(atr_val) or atr_val <= 0:
            return None
        entry = float(df["close"].iloc[i])
        sl, sl_note = None, ""
        if strat_sl is not None and np.isfinite(strat_sl):
            dist = (entry - strat_sl) if side == "long" else (strat_sl - entry)
            # strategy stop (order block, spike origin, Smart Trail...) if it is sane
            if 0.3 * atr_val <= dist <= 5.0 * atr_val:
                sl, sl_note = float(strat_sl), "strategy stop"
            elif dist > 0:
                d = min(max(dist, 0.5 * atr_val), 5.0 * atr_val)
                sl = entry - d if side == "long" else entry + d
                sl_note = "strategy stop (clamped to 0.5..5 ATR)"
        if sl is None:
            sl, sl_note = choose_sl(df, i, side, entry, atr_val, pl, ph)
        lv = levels_at(df, i, ph, pl, atr_val)
        tps, tp_notes = choose_tps(side, entry, sl, lv, atr_val)
        name = self.strategy.name
        if source and source in STRATEGIES:
            name = f"Auto Pilot · {STRATEGIES[source].name}"
        return Signal(symbol=symbol, timeframe=timeframe, strategy=name, side=side,
                      entry=entry, sl=float(sl), tps=[float(t) for t in tps],
                      confidence=float(min(1.0, strength)),
                      reasons=[f"SL {sl_note}", "TPs: " + ", ".join(tp_notes)],
                      bar_time=float(df.index[i].timestamp()))

    def plan_row(self, df, i, sig, a, ph, pl, symbol="", timeframe=""):
        row = sig.iloc[i]
        side = "long" if row["long"] else "short" if row["short"] else None
        if side is None:
            return None
        src = row["source"] if "source" in sig.columns else ""
        s = self.plan(df, i, side, float(row["strength"]), a, ph, pl, symbol, timeframe,
                      row.get("sl", np.nan), src)
        if s is not None and "agree" in sig.columns and int(row["agree"]) > 1:
            s.reasons.append(f"{int(row['agree'])} strategies agree")
        return s

    def latest_signal(self, df: pd.DataFrame, symbol: str, timeframe: str, fresh_bars: int = 1,
                      params: dict | None = None):
        """Evaluate the last CLOSED candles of df (the newest `fresh_bars`, newest first).
        A signal from an older bar is only returned while price is still near its entry.
        Returns (Signal|None, sigframe)."""
        sig, a, ph, pl = self.prepare(df, params)
        n = len(df)
        s = None
        for back in range(max(1, int(fresh_bars))):
            i = n - 1 - back
            if i < 0:
                break
            s = self.plan_row(df, i, sig, a, ph, pl, symbol, timeframe)
            if s is None:
                continue
            if back > 0:
                last = float(df["close"].iloc[-1])
                drift = (last - s.entry) * (1 if s.side == "long" else -1)
                if abs(drift) > 0.35 * float(a.iloc[-1]):
                    s = None            # price ran away from that entry: too late
                    continue
                s.reasons.append(f"signal {back} bar(s) ago, price still near entry")
                s.entry = last          # plan from the current price; SL/TP stay where they are
            break
        if s is None:
            return None, sig
        if self.use_ml:
            p_up, acc = self.ml.prob_up((symbol, timeframe), df)
            if p_up is not None:
                p_side = p_up if s.side == "long" else 1 - p_up
                s.ml_prob = p_side
                s.reasons.append(f"ML {p_side:.0%} (holdout acc {acc:.0%})")
                if p_side < 0.45:          # ML veto
                    s.reasons.append("ML veto")
                    s.confidence = min(s.confidence, p_side)
                    return s, sig
                s.confidence = 0.5 * s.confidence + 0.5 * min(1.0, p_side + 0.1)
        return s, sig
