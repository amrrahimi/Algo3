"""Event-driven backtester with partial take-profits, breakeven stop,
fees, slippage and conservative intrabar fills (SL assumed first)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analyzer import Analyzer

TP_FRACTIONS = (0.5, 0.3, 0.2)


@dataclass
class BTResult:
    strategy: str
    symbol: str
    trades: list = field(default_factory=list)
    net_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    expectancy_r: float = 0.0
    n: int = 0
    score: float = 0.0


def run_backtest(df: pd.DataFrame, strategy_key: str, symbol: str = "", market_type: str = "futures",
                 risk_pct: float = 1.0, fee: float = 0.0006, slippage: float = 0.0002,
                 warmup: int = 210, max_bars_in_trade: int = 300, leverage: int = 1,
                 min_confidence: float = 0.0, mode: str = "daily", params: dict | None = None,
                 start: int | None = None, end: int | None = None, prepared=None) -> BTResult:
    """Fills are conservative: stop before targets inside a bar, and a bar that OPENS
    beyond the stop fills at the open (gap). Position size is capped like the live robot
    (notional <= 95% balance x leverage), so R results are scaled by the risk actually taken."""
    an = Analyzer(strategy_key, use_ml=False, mode=mode, params=params)
    sig, a, ph, pl = prepared if prepared is not None else an.prepare(df)
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    res = BTResult(strategy=an.strategy.name, symbol=symbol)
    lev = max(1, leverage) if market_type == "futures" else 1
    equity, peak, max_dd = 1.0, 1.0, 0.0
    i = max(warmup, start or 0)
    n_entry = min(len(df), end) if end else len(df)     # signals only inside [start, end)
    n = len(df)
    while i < n_entry - 2:
        row = sig.iloc[i]
        side = "long" if row["long"] else "short" if (row["short"] and market_type == "futures") else None
        if side is None or float(row["strength"]) < min_confidence:
            i += 1
            continue
        plan = an.plan_row(df, i, sig, a, ph, pl, symbol)
        if plan is None or plan.side != side:
            i += 1
            continue
        sgn = 1 if side == "long" else -1
        entry = o[i + 1] * (1 + sgn * slippage)
        # re-anchor SL/TP distances to the actual fill
        shift = entry - plan.entry
        sl = plan.sl + shift
        tps = [t + shift for t in plan.tps]
        risk = abs(entry - sl)
        if risk <= 0:
            i += 1
            continue
        # risk actually taken (% of equity): capped by max notional, like position_size()
        risk_taken = min(risk_pct / 100.0, 0.95 * lev * risk / entry)
        remaining, r_total, tp_idx, worst_r = 1.0, 0.0, 0, 0.0
        j = i + 1
        exit_bar = n - 1
        while j < n:
            # adverse excursion for intra-trade drawdown
            adv = l[j] if sgn == 1 else h[j]
            worst_r = min(worst_r, r_total + remaining * sgn * (adv - entry) / risk)
            if (sgn == 1 and l[j] <= sl) or (sgn == -1 and h[j] >= sl):
                gap = (sgn == 1 and o[j] < sl) or (sgn == -1 and o[j] > sl)
                px = (o[j] if gap else sl) * (1 - sgn * slippage)
                r_total += remaining * sgn * (px - entry) / risk
                remaining = 0
            else:
                while tp_idx < 3 and ((sgn == 1 and h[j] >= tps[tp_idx]) or (sgn == -1 and l[j] <= tps[tp_idx])):
                    frac = TP_FRACTIONS[tp_idx] if tp_idx < 2 else remaining
                    frac = min(frac, remaining)
                    r_total += frac * sgn * (tps[tp_idx] - entry) / risk
                    remaining -= frac
                    tp_idx += 1
                    sl = entry if tp_idx == 1 else tps[tp_idx - 2]   # breakeven, then trail
                if remaining <= 1e-9:
                    remaining = 0
            if remaining == 0 or j - i >= max_bars_in_trade:
                if remaining > 0:   # time exit
                    r_total += remaining * sgn * (c[j] - entry) / risk
                    remaining = 0
                exit_bar = j
                break
            j += 1
        if remaining > 0 and exit_bar == n - 1:   # data ended with the trade open: mark to market
            r_total += remaining * sgn * (c[-1] - entry) / risk
        fee_r = 2 * fee * entry / risk
        r_net = r_total - fee_r
        scale = risk_taken * 100.0 / risk_pct if risk_pct > 0 else 1.0   # <1 when size was capped
        low_eq = equity * (1 + (worst_r - fee_r / 2) * risk_taken)
        max_dd = max(max_dd, (peak - low_eq) / peak)
        equity *= (1 + r_net * risk_taken)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        res.trades.append({"time": df.index[i], "side": side, "entry": entry, "r": r_net * scale,
                           "exit_time": df.index[exit_bar], "tp_hit": tp_idx})
        i = exit_bar + 1
    rs = np.array([t["r"] for t in res.trades])
    res.n = len(rs)
    if res.n:
        res.win_rate = float((rs > 0).mean())
        gains, losses = rs[rs > 0].sum(), -rs[rs < 0].sum()
        res.profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
        res.expectancy_r = float(rs.mean())
        res.net_pct = (equity - 1) * 100
        res.max_dd_pct = max_dd * 100
        conf = min(1.0, res.n / 20.0)
        res.score = float(res.expectancy_r * np.sqrt(res.n) * conf * (1 - min(max_dd, 0.9)))
    return res


def rank_strategies(dfs: dict, mode: str, market_type: str = "futures", risk_pct: float = 1.0,
                    leverage: int = 1, min_confidence: float = 0.0, tuned: dict | None = None):
    """dfs: {symbol: DataFrame}. Returns list of dicts sorted by score.
    Uses the same min-confidence gate as the robot (the ML filter is not simulated)."""
    from .strategies import strategies_for_mode
    out = []
    for strat in strategies_for_mode(mode):
        results = [run_backtest(df, strat.key, sym, market_type, risk_pct, leverage=leverage,
                                min_confidence=min_confidence, mode=mode,
                                params=(tuned or {}).get((sym, strat.key)))
                   for sym, df in dfs.items()]
        n = sum(r.n for r in results)
        if n == 0:
            out.append({"key": strat.key, "name": strat.name, "trades": 0, "win_rate": 0, "pf": 0,
                        "net_pct": 0, "max_dd": 0, "score": -999})
            continue
        all_r = np.concatenate([[t["r"] for t in r.trades] for r in results if r.trades])
        gains, losses = all_r[all_r > 0].sum(), -all_r[all_r < 0].sum()
        out.append({"key": strat.key, "name": strat.name, "trades": n,
                    "win_rate": float((all_r > 0).mean()),
                    "pf": float(gains / losses) if losses > 0 else 99.0,
                    "net_pct": float(np.mean([r.net_pct for r in results])),
                    "max_dd": float(max(r.max_dd_pct for r in results)),
                    "score": float(np.mean([r.score for r in results]))})
    out.sort(key=lambda d: d["score"], reverse=True)
    return out
