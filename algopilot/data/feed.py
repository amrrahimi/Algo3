"""Market data: live from exchanges via ccxt, or synthetic demo data offline."""
from __future__ import annotations

import hashlib
import threading
import time

import numpy as np
import pandas as pd

from ..config import TF_SECONDS

CCXT_IDS = {
    ("binance", "spot"): "binance", ("binance", "futures"): "binanceusdm",
    ("bybit", "spot"): "bybit", ("bybit", "futures"): "bybit",
    ("kucoin", "spot"): "kucoin", ("kucoin", "futures"): "kucoinfutures",
    ("toobit", "spot"): "toobit", ("toobit", "futures"): "toobit",
}


def to_market_symbol(symbol: str, market_type: str) -> str:
    """'BTC/USDT' -> 'BTC/USDT:USDT' for USDT-margined perpetuals."""
    if market_type == "futures" and ":" not in symbol:
        quote = symbol.split("/")[1]
        return f"{symbol}:{quote}"
    return symbol


def make_exchange(exchange_id: str, market_type: str, api_key: str = "", secret: str = "",
                  password: str = "", testnet: bool = False):
    import ccxt  # lazy import keeps the core testable without ccxt
    cid = CCXT_IDS.get((exchange_id, market_type), exchange_id)
    cls = getattr(ccxt, cid, None)
    if cls is None:
        raise RuntimeError(f"{exchange_id} ({market_type}) is not supported by the installed ccxt "
                           f"version {ccxt.__version__}. Update ccxt (pip install -U ccxt).")
    cfg = {"enableRateLimit": True, "timeout": 20000, "options": {}}
    if cid in ("bybit", "toobit"):
        cfg["options"]["defaultType"] = "swap" if market_type == "futures" else "spot"
    if api_key:
        cfg["apiKey"], cfg["secret"] = api_key, secret
        if password:
            cfg["password"] = password
    ex = cls(cfg)
    if testnet:
        enable_test_mode(ex, exchange_id, market_type)
    return ex


# Binance retired its futures testnet in favour of "Demo Trading" (demo.binance.com);
# ccxt exposes it as enable_demo_trading(). Bybit testnet (testnet.bybit.com) works through
# the classic sandbox URLs.
DEMO_TRADING_FIRST = {"binance"}


def enable_test_mode(ex, exchange_id: str, market_type: str) -> str:
    """Switch a ccxt client to the exchange's test environment. Returns its name."""
    errors = []
    order = ["demo", "sandbox"] if exchange_id in DEMO_TRADING_FIRST else ["sandbox", "demo"]
    for kind in order:
        try:
            if kind == "demo":
                fn = getattr(ex, "enable_demo_trading", None)
                if fn is None:
                    raise RuntimeError("this ccxt version has no demo trading (pip install -U ccxt)")
                fn(True)
                return "demo trading"
            ex.set_sandbox_mode(True)
            return "testnet"
        except Exception as e:  # noqa: BLE001
            errors.append(f"{kind}: {e}")
    raise RuntimeError(f"{exchange_id} ({market_type}) has no testnet/demo trading in ccxt "
                       f"({'; '.join(errors)}). Use the built-in Demo account instead.")


# Max candles per fetch_ohlcv request (ccxt ids). Unknown exchanges use a safe 200.
OHLCV_MAX = {"binance": 1000, "binanceusdm": 1000, "bybit": 1000, "kucoin": 1500,
             "kucoinfutures": 200, "toobit": 1000}


def ohlcv_to_df(rows) -> pd.DataFrame:
    rows = [r[:6] for r in rows]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").astype(float)
    return df


class ExchangeFeed:
    def __init__(self, exchange_id: str, market_type: str, client=None):
        """`client`: optional ready ccxt client (e.g. the live/testnet broker's), so prices
        come from the same venue the orders go to."""
        self.exchange_id = exchange_id
        self.market_type = market_type
        self._ex = client
        self._lock = threading.Lock()

    def _client(self):
        with self._lock:
            if self._ex is None:
                ex = make_exchange(self.exchange_id, self.market_type)
                ex.load_markets()
                self._ex = ex
            return self._ex

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        ex = self._client()
        sym = to_market_symbol(symbol, self.market_type)
        per_call = min(limit, OHLCV_MAX.get(getattr(ex, "id", ""), 200))
        rows, since = [], None
        if limit > per_call:      # paginate forward from (now - limit bars)
            since = ex.milliseconds() - (limit + 2) * TF_SECONDS[timeframe] * 1000
        for _ in range(50):
            batch = ex.fetch_ohlcv(sym, timeframe, since=since, limit=per_call)
            if not batch:
                break
            rows.extend(batch)
            if since is None or len(batch) < per_call:
                break
            nxt = batch[-1][0] + 1
            if nxt <= since:
                break
            since = nxt
        if not rows:
            raise RuntimeError(f"no candles returned for {sym} {timeframe}")
        return ohlcv_to_df(rows).sort_index().iloc[-limit:]

    def fetch_price(self, symbol: str, timeframe: str | None = None) -> float:
        t = self._client().fetch_ticker(to_market_symbol(symbol, self.market_type))
        return float(t.get("last") or t.get("close"))

    def list_symbols(self, quote: str = "USDT") -> list:
        ex = self._client()
        out = []
        for m in ex.markets.values():
            if m.get("quote") != quote or not m.get("active", True):
                continue
            if self.market_type == "futures" and m.get("swap") and m.get("linear"):
                out.append(f"{m['base']}/{m['quote']}")
            elif self.market_type == "spot" and m.get("spot"):
                out.append(f"{m['base']}/{m['quote']}")
        return sorted(set(out))


class DemoFeed:
    """Deterministic synthetic market (trends, ranges, volatility regimes) so the
    app can be tested with no internet. New candles appear in real time."""

    BASES = {"BTC": 65000, "ETH": 3200, "SOL": 150, "BNB": 580, "XRP": 0.6, "DOGE": 0.15}

    def __init__(self, market_type: str = "futures"):
        self.market_type = market_type

    def _seed(self, symbol):
        return int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        tf = TF_SECONDS[timeframe]
        now = time.time()
        last = int(now // tf)
        idx = np.arange(last - limit + 1, last + 1)
        seed = self._seed(symbol)
        base = self.BASES.get(symbol.split("/")[0], 10 + seed % 90)
        t = idx.astype(float)
        ph = (seed % 1000) / 1000 * 6.28
        logp = (0.06 * np.sin(t / 97 + ph) + 0.035 * np.sin(t / 31 + 2 * ph) +
                0.12 * np.sin(t / 401 + 3 * ph) + 0.015 * np.sin(t / 9 + ph))
        noise = np.array([np.random.default_rng([seed, int(k)]).normal() for k in idx])
        vol = 0.004 * (1.2 + np.sin(t / 53 + ph))
        close = base * np.exp(logp + noise * vol)
        # live partial candle: progress within current bar
        frac = (now % tf) / tf
        close[-1] = close[-2] + (close[-1] - close[-2]) * frac
        open_ = np.concatenate([[close[0]], close[:-1]])
        rng_hl = np.abs(np.array([np.random.default_rng([seed, int(k), 7]).normal() for k in idx])) * vol * base
        high = np.maximum(open_, close) + rng_hl * 0.5
        low = np.minimum(open_, close) - rng_hl * 0.5
        volume = 1000 * (1 + np.abs(noise)) * (1.5 + np.sin(t / 13))
        ts = pd.to_datetime(idx * tf, unit="s", utc=True)
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                             "volume": volume}, index=ts)

    def fetch_price(self, symbol: str, timeframe: str | None = None) -> float:
        # the synthetic series depends on the timeframe: price from the same series as the chart
        return float(self.fetch_ohlcv(symbol, timeframe or "15m", 3)["close"].iloc[-1])

    def list_symbols(self, quote: str = "USDT") -> list:
        return [f"{b}/{quote}" for b in self.BASES]


def make_feed(source: str, market_type: str, client=None):
    if source == "demo":
        return DemoFeed(market_type)
    return ExchangeFeed(source, market_type, client)
