"""Technical indicators (pure pandas/numpy).

Every function takes a OHLCV DataFrame (columns: open, high, low, close, volume,
DatetimeIndex) or a Series and returns Series aligned to the input index.
Includes classic indicators plus popular "Algo" indicators (UT Bot, Range
Filter, Squeeze Momentum, WaveTrend, Chandelier Exit, Kalman filter).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- basics
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _rolling_dot(s: pd.Series, w: np.ndarray) -> pd.Series:
    """sum(w[k] * s[t-n+1+k]) for each full window (vectorized, NaN if the window has a NaN)."""
    n = len(w)
    x = s.values.astype(float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        bad = np.isnan(x)
        xf = np.where(bad, 0.0, x)
        out[n - 1:] = np.convolve(xf, w[::-1], mode="valid")
        nb = np.convolve(bad.astype(float), np.ones(n), mode="valid")
        out[n - 1:][nb > 0] = np.nan
    return pd.Series(out, index=s.index)


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return _rolling_dot(s, w / w.sum())


def hma(s: pd.Series, n: int = 55) -> pd.Series:
    half = max(int(n / 2), 1)
    sq = max(int(np.sqrt(n)), 1)
    return wma(2 * wma(s, half) - wma(s, n), sq)


def hlc3(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                     axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(dn != 0, 100.0)
    return out.where(up.notna())


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return m, sig, m - sig


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def keltner(df: pd.DataFrame, n: int = 20, k: float = 1.5):
    mid = ema(df["close"], n)
    a = atr(df, n)
    return mid + k * a, mid, mid - k * a


def donchian(df: pd.DataFrame, n: int = 20):
    return df["high"].rolling(n, min_periods=n).max(), df["low"].rolling(n, min_periods=n).min()


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = rma(true_range(df), n).replace(0, np.nan)
    pdi = 100 * rma(plus, n) / tr
    mdi = 100 * rma(minus, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n), pdi, mdi


def stoch_rsi(s: pd.Series, n: int = 14, k: int = 3, d: int = 3):
    r = rsi(s, n)
    lo = r.rolling(n, min_periods=n).min()
    hi = r.rolling(n, min_periods=n).max()
    st = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    kk = sma(st.fillna(50), k)
    return kk, sma(kk, d)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = hlc3(df)
    m = sma(tp, n)
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = hlc3(df)
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(), 0.0)
    neg = rmf.where(tp < tp.shift(), 0.0)
    ratio = pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP, resets every UTC day."""
    tp = hlc3(df)
    day = df.index.normalize()
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return (pv / vv).fillna(tp)


def ichimoku(df: pd.DataFrame, t: int = 9, k: int = 26, s: int = 52):
    def mid(n):
        return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2
    tenkan, kijun = mid(t), mid(k)
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = mid(s).shift(k)
    return tenkan, kijun, span_a, span_b


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = np.zeros(len(df))
    o, c = df["open"].values, ha_close.values
    for i in range(len(df)):
        ha_open[i] = (o[0] + c[0]) / 2 if i == 0 else (ha_open[i - 1] + c[i - 1]) / 2
    ha_open = pd.Series(ha_open, index=df.index)
    return pd.DataFrame({"open": ha_open,
                         "high": pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1),
                         "low": pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1),
                         "close": ha_close, "volume": df["volume"]}, index=df.index)


def linreg(s: pd.Series, n: int) -> pd.Series:
    """Value of the least-squares line at the last point of each window (Pine ta.linreg, offset 0)."""
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    b = _rolling_dot(s, (x - xm) / denom)
    ym = _rolling_dot(s, np.full(n, 1.0 / n))
    return ym + b * (n - 1 - xm)


def slope(s: pd.Series, n: int = 10) -> pd.Series:
    return (s - s.shift(n)) / n


# ------------------------------------------------------- "Algo" indicators
def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    """Returns (line, direction) where direction is +1 bullish / -1 bearish."""
    a = atr(df, n).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    c = df["close"].values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    L = len(c)
    fu, fl, st = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)
    dirn = np.zeros(L)
    for i in range(L):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(fu[i - 1]):
            fu[i], fl[i], dirn[i] = ub[i], lb[i], 1
            st[i] = fl[i]
            continue
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if dirn[i - 1] == 1:
            dirn[i] = -1 if c[i] < fl[i] else 1
        else:
            dirn[i] = 1 if c[i] > fu[i] else -1
        st[i] = fl[i] if dirn[i] == 1 else fu[i]
    return pd.Series(st, index=df.index), pd.Series(dirn, index=df.index)


def ut_bot(df: pd.DataFrame, key: float = 1.0, n: int = 10):
    """UT Bot Alerts (ATR trailing stop). Returns (stop, buy, sell)."""
    src = df["close"].values
    loss = key * atr(df, n).values
    stop = np.full(len(src), np.nan)
    for i in range(1, len(src)):
        if np.isnan(loss[i]):
            continue
        prev = stop[i - 1] if not np.isnan(stop[i - 1]) else src[i - 1]
        if src[i] > prev and src[i - 1] > prev:
            stop[i] = max(prev, src[i] - loss[i])
        elif src[i] < prev and src[i - 1] < prev:
            stop[i] = min(prev, src[i] + loss[i])
        elif src[i] > prev:
            stop[i] = src[i] - loss[i]
        else:
            stop[i] = src[i] + loss[i]
    stop_s = pd.Series(stop, index=df.index)
    c = df["close"]
    buy = (c > stop_s) & (c.shift() <= stop_s.shift())
    sell = (c < stop_s) & (c.shift() >= stop_s.shift())
    return stop_s, buy.fillna(False), sell.fillna(False)


def range_filter(s: pd.Series, n: int = 100, mult: float = 3.0):
    """Range Filter (DonovanWall). Returns (filter, direction +1/-1/0)."""
    avrng = ema(s.diff().abs(), n)
    smrng = (ema(avrng, n * 2 - 1) * mult).values
    x = s.values
    f = np.full(len(x), np.nan)
    for i in range(len(x)):
        if np.isnan(smrng[i]):
            continue
        prev = f[i - 1] if i > 0 and not np.isnan(f[i - 1]) else x[i]
        r = smrng[i]
        if x[i] > prev:
            f[i] = prev if x[i] - r < prev else x[i] - r
        else:
            f[i] = prev if x[i] + r > prev else x[i] + r
    filt = pd.Series(f, index=s.index)
    dirn = np.sign(filt.diff()).replace(0, np.nan).ffill().fillna(0)
    return filt, dirn


def squeeze_lazybear(df: pd.DataFrame, length: int = 20, mult: float = 2.0, length_kc: int = 20,
                     mult_kc: float = 1.5, use_tr: bool = True, faithful: bool = True) -> dict:
    """Squeeze Momentum Indicator [LazyBear], ported 1:1 from the TradingView script.

    faithful=True reproduces the original script exactly, including its well-known quirk that the
    Bollinger deviation uses the KC multiplier (dev = multKC * stdev). Returns a dict with
    val (momentum histogram), sqz_on / sqz_off / no_sqz (bool), bar_color and dot_color codes:
      bar_color: 2 lime (up, rising) 1 green (up, falling) -1 red (down, falling) -2 maroon (down, rising)
      dot_color: 0 black (squeeze ON) 1 gray (squeeze OFF, just released) 2 blue (no squeeze)
    """
    c = df["close"]
    basis = sma(c, length)
    sd = c.rolling(length, min_periods=length).std(ddof=0)   # Pine stdev = population
    dev = (mult_kc if faithful else mult) * sd
    upper_bb, lower_bb = basis + dev, basis - dev
    ma = sma(c, length_kc)
    rng = true_range(df) if use_tr else (df["high"] - df["low"])
    rangema = sma(rng, length_kc)
    upper_kc, lower_kc = ma + rangema * mult_kc, ma - rangema * mult_kc
    sqz_on = ((lower_bb > lower_kc) & (upper_bb < upper_kc)).fillna(False)
    sqz_off = ((lower_bb < lower_kc) & (upper_bb > upper_kc)).fillna(False)
    no_sqz = ~sqz_on & ~sqz_off
    hh = df["high"].rolling(length_kc, min_periods=length_kc).max()
    ll = df["low"].rolling(length_kc, min_periods=length_kc).min()
    avg = ((hh + ll) / 2 + ma) / 2
    val = linreg(c - avg, length_kc)
    prev = val.shift().fillna(0)
    bar = np.where(val > 0, np.where(val > prev, 2, 1), np.where(val < prev, -1, -2))
    bar = pd.Series(np.where(val.isna(), 0, bar), index=df.index)
    dot = pd.Series(np.where(no_sqz, 2, np.where(sqz_on, 0, 1)), index=df.index)
    return {"val": val, "sqz_on": sqz_on, "sqz_off": sqz_off, "no_sqz": no_sqz,
            "bar_color": bar, "dot_color": dot}


def squeeze_momentum(df: pd.DataFrame, bb_n: int = 20, bb_k: float = 2.0,
                     kc_n: int = 20, kc_k: float = 1.5):
    """LazyBear Squeeze Momentum. Returns (momentum, squeeze_on bool)."""
    r = squeeze_lazybear(df, bb_n, bb_k, kc_n, kc_k)
    return r["val"], r["sqz_on"]


def wavetrend(df: pd.DataFrame, n1: int = 10, n2: int = 21):
    ap = hlc3(df)
    esa = ema(ap, n1)
    d = ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ema(ci, n2)
    return wt1, sma(wt1, 4)


def chandelier_exit(df: pd.DataFrame, n: int = 22, mult: float = 3.0):
    a = atr(df, n) * mult
    return df["high"].rolling(n).max() - a, df["low"].rolling(n).min() + a


def kalman(s: pd.Series, q: float = 0.01, r: float = 1.0) -> pd.Series:
    """1-D Kalman filter: a low-lag noise filter for price."""
    x = s.values.astype(float)
    out = np.full(len(x), np.nan)
    est, p = np.nan, 1.0
    for i, z in enumerate(x):
        if np.isnan(z):
            out[i] = est
            continue
        if np.isnan(est):
            est = z
        p += q
        k = p / (p + r)
        est += k * (z - est)
        p *= (1 - k)
        out[i] = est
    return pd.Series(out, index=s.index)


def kalman_price(s: pd.Series, smooth: float = 0.05) -> pd.Series:
    """Kalman filter scaled to price volatility (smooth: lower = smoother)."""
    return kalman(s, q=smooth, r=1.0)


# ------------------------------------------------------ noise / regime
def noise_ok(df: pd.DataFrame, atr_n: int = 14) -> pd.Series:
    """True when the market is moving enough and has volume (avoid dead chop)."""
    a = atr(df, atr_n) / df["close"]
    act = a > a.rolling(200, min_periods=50).quantile(0.15)
    vol = df["volume"] >= 0.6 * sma(df["volume"], 20)
    return (act & vol).fillna(False)


def efficiency_ratio(s: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman efficiency ratio 0..1 (1 = clean trend, 0 = pure noise)."""
    change = (s - s.shift(n)).abs()
    vol = s.diff().abs().rolling(n).sum()
    return (change / vol.replace(0, np.nan)).fillna(0)
