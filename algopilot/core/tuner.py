"""Auto-tuner: picks each indicator's settings from the coin's own chart.

For one (coin, timeframe, style) it backtests a sample of parameter combinations of a strategy on
the older part of the history (train, ~65%) and keeps the best one, then checks it on the most
recent part it has never seen (test, ~35%, walk-forward). Results:

  validated  best params also made money out-of-sample -> used, confidence bonus
  unproven   too few out-of-sample trades to judge     -> used, small penalty
  failed     lost money out-of-sample                  -> style preset kept, strategy muted in
             Auto Pilot (unless signal frequency = "More trades")

Nothing here is a profit guarantee: it only avoids settings that clearly don't fit the market.
"""
from __future__ import annotations

import hashlib
import itertools
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .backtest import run_backtest
from .strategies import STRATEGIES, strategies_for_mode


@dataclass
class TuneResult:
    strategy: str
    params: dict
    preset: dict
    status: str                    # validated | unproven | failed | untuned
    train: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    combos: int = 0
    bars: int = 0
    tuned_at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @property
    def changed(self) -> dict:
        return {k: v for k, v in self.params.items() if self.preset.get(k) != v}


def _stats(r) -> dict:
    return {"n": int(r.n), "win": round(float(r.win_rate), 3), "pf": round(float(min(r.profit_factor, 99)), 2),
            "exp_r": round(float(r.expectancy_r), 3), "net_pct": round(float(r.net_pct), 2),
            "dd": round(float(r.max_dd_pct), 2)}


def _train_score(r, min_trades: int) -> float:
    if r.n < min_trades:
        return -1e9 + r.n
    pf = min(r.profit_factor, 5.0)
    return float(r.expectancy_r * np.sqrt(r.n) * (0.5 + 0.1 * pf) * (1 - min(r.max_dd_pct / 100, 0.9)))


class AutoTuner:
    def __init__(self, max_combos: int = 24, min_trades: int = 4, train_frac: float = 0.65):
        self.max_combos = max_combos
        self.min_trades = min_trades
        self.train_frac = train_frac

    def _candidates(self, key: str, mode: str) -> list:
        st = STRATEGIES[key]
        preset = st.params_for(mode)
        space = st.search_space(mode)
        if not space:
            return [preset]
        names = list(space)
        grid = [dict(zip(names, vals)) for vals in itertools.product(*(space[k] for k in names))]
        rnd = random.Random(int(hashlib.md5(f"{key}:{mode}".encode()).hexdigest()[:8], 16))
        rnd.shuffle(grid)
        out, seen = [preset], {tuple(sorted(preset.items()))}
        for g in grid:
            p = dict(preset)
            p.update(g)
            t = tuple(sorted(p.items()))
            if t not in seen:
                seen.add(t)
                out.append(p)
            if len(out) >= self.max_combos:
                break
        return out

    def tune_strategy(self, df, key: str, mode: str, market_type: str = "futures",
                      leverage: int = 1, stop=None) -> TuneResult:
        st = STRATEGIES[key]
        preset = st.params_for(mode)
        n = len(df)
        warm = 210
        split = warm + int((n - warm) * self.train_frac)
        if n < warm + 300:
            return TuneResult(key, preset, preset, "untuned", bars=n)
        best, best_score, best_train = preset, -1e18, None
        cands = self._candidates(key, mode)
        tried = 0
        for p in cands:
            if stop is not None and stop():
                break
            tried += 1
            try:
                r = run_backtest(df, key, market_type=market_type, leverage=leverage, mode=mode, params=p,
                                 end=split)
            except Exception:  # noqa: BLE001  (a bad combo must not stop tuning)
                continue
            sc = _train_score(r, self.min_trades)
            if sc > best_score:
                best, best_score, best_train = p, sc, r
        if best_train is None:
            return TuneResult(key, preset, preset, "untuned", combos=tried, bars=n)
        test = run_backtest(df, key, market_type=market_type, leverage=leverage, mode=mode, params=best,
                            start=split)
        if best_train.expectancy_r <= 0:
            status = "failed"            # even the best combo lost money in training
        elif test.n >= 3 and test.expectancy_r > 0:
            status = "validated"
        elif test.n < 3 and test.expectancy_r >= 0:
            status = "unproven"
        else:
            status = "failed"
        params = best
        if status == "failed" and best != preset:
            # fall back to the style preset if it does better on the unseen data
            pt = run_backtest(df, key, market_type=market_type, leverage=leverage, mode=mode, params=preset,
                              start=split)
            if pt.expectancy_r > test.expectancy_r:
                params, test = preset, pt
                status = "unproven" if (pt.n >= 3 and pt.expectancy_r > 0) else "failed"
        return TuneResult(key, params, preset, status, _stats(best_train), _stats(test), tried, n)

    def tune_mode(self, df, mode: str, keys=None, market_type: str = "futures", leverage: int = 1,
                  stop=None, progress=None) -> dict:
        keys = keys or [s.key for s in strategies_for_mode(mode) if s.key != "auto_pilot"]
        out = {}
        for k in keys:
            if stop is not None and stop():
                break
            if k == "auto_pilot" or k not in STRATEGIES:
                continue
            out[k] = self.tune_strategy(df, k, mode, market_type, leverage, stop)
            if progress:
                progress(out[k])
        return out


def bonus_for(r: TuneResult) -> float:
    if r.status == "validated":
        pf = r.test.get("pf", 1)
        return 0.06 if pf >= 1.5 else 0.03
    if r.status == "unproven":
        return -0.02
    if r.status == "failed":
        return -0.10
    return 0.0


def autopilot_params(results: dict, mode: str, frequency: str = "balanced") -> dict:
    """Build Auto Pilot params from per-strategy tuning results."""
    members, bonus, enabled = {}, {}, []
    allowed = {"quality": ("validated",), "balanced": ("validated", "unproven", "untuned"),
               "high": ("validated", "unproven", "untuned", "failed")}[frequency if frequency in
                                                                       ("quality", "balanced", "high") else "balanced"]
    for k, r in results.items():
        members[k] = r.params
        bonus[k] = bonus_for(r)
        if r.status in allowed:
            enabled.append(k)
    if not results:            # not tuned yet: everything on, style presets
        enabled = None
    return {"mode": mode, "members": members, "enabled": enabled, "bonus": bonus}
