from __future__ import annotations


def position_size(balance: float, risk_pct: float, entry: float, sl: float,
                  market_type: str = "futures", leverage: int = 1, fee: float = 0.0,
                  free_balance: float | None = None) -> float:
    """Base-asset quantity so that hitting SL loses `risk_pct`% of `balance`
    including entry + exit fees. The notional is capped by the free balance
    (x leverage on futures)."""
    dist = abs(entry - sl)
    if dist <= 0 or entry <= 0 or balance <= 0:
        return 0.0
    qty = balance * (risk_pct / 100.0) / (dist + 2 * fee * entry)
    free = balance if free_balance is None else max(0.0, free_balance)
    lev = max(1, leverage) if market_type == "futures" else 1.0
    max_notional = free * lev * 0.95 / (1 + fee * lev)
    return max(0.0, min(qty, max_notional / entry))


def liquidation_price(side: str, entry: float, leverage: int, mmr: float = 0.005) -> float | None:
    """Approximate isolated-margin liquidation price."""
    if leverage <= 1:
        return None
    if side == "long":
        return entry * (1 - 1.0 / leverage + mmr)
    return entry * (1 + 1.0 / leverage - mmr)


def sl_safe_with_leverage(side: str, entry: float, sl: float, leverage: int) -> bool:
    """SL must trigger before liquidation."""
    liq = liquidation_price(side, entry, leverage)
    if liq is None:
        return True
    return sl > liq if side == "long" else sl < liq
