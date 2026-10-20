"""Machine-learning signal filter.

Trains a gradient-boosting classifier (scikit-learn) on indicator features to
estimate the probability that price moves up vs. down over the next N bars.
Falls back to a pure-numpy logistic regression if scikit-learn is missing.
The model is validated on a time-ordered holdout; if it has no edge its
output is shrunk toward 0.5 so it cannot fake confidence.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from . import indicators as ind

try:  # optional dependency
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    a = ind.atr(df, 14).replace(0, np.nan)
    e20, e50, e200 = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
    _, _, hist = ind.macd(c)
    adx, pdi, mdi = ind.adx(df)
    ub, mid, lb = ind.bollinger(c)
    wt1, wt2 = ind.wavetrend(df)
    k, _ = ind.stoch_rsi(c)
    _, st_dir = ind.supertrend(df)
    kal = ind.kalman_price(c)
    vol_z = (df["volume"] - ind.sma(df["volume"], 20)) / df["volume"].rolling(20).std().replace(0, np.nan)
    X = pd.DataFrame({
        "r1": c.pct_change(1), "r3": c.pct_change(3), "r6": c.pct_change(6), "r12": c.pct_change(12),
        "rsi": ind.rsi(c) / 100, "hist_atr": hist / a,
        "c_e20": (c - e20) / a, "c_e50": (c - e50) / a, "e20_e50": (e20 - e50) / a,
        "c_e200": (c - e200) / a, "adx": adx / 100, "di": (pdi - mdi) / 100,
        "bb_b": (c - lb) / (ub - lb).replace(0, np.nan), "bb_w": (ub - lb) / c,
        "atr_pct": a / c, "vol_z": vol_z, "wt1": wt1 / 100, "wt_d": (wt1 - wt2) / 100,
        "stoch": k / 100, "st_dir": st_dir, "kal_slope": ind.slope(kal, 5) / a,
        "er": ind.efficiency_ratio(c), "cci": ind.cci(df) / 200,
        "body": (c - df["open"]) / a, "wick_up": (df["high"] - np.maximum(c, df["open"])) / a,
        "wick_dn": (np.minimum(c, df["open"]) - df["low"]) / a,
    }, index=df.index)
    return X.replace([np.inf, -np.inf], np.nan)


def build_labels(df: pd.DataFrame, horizon: int, thr_atr: float = 0.3) -> pd.Series:
    c = df["close"]
    fwd = c.shift(-horizon) - c
    thr = thr_atr * ind.atr(df, 14)
    y = pd.Series(np.nan, index=df.index)
    y[fwd > thr] = 1.0
    y[fwd < -thr] = 0.0
    return y


class _NumpyLogit:
    def fit(self, X, y, iters=400, lr=0.1, l2=1e-3):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            g = p - y
            self.w -= lr * (Z.T @ g / len(y) + l2 * self.w)
            self.b -= lr * g.mean()
        return self

    def predict_proba(self, X):
        Z = (X - self.mu) / self.sd
        p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
        return np.column_stack([1 - p, p])


def _make_model():
    if HAVE_SKLEARN:
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                              l2_regularization=1.0, min_samples_leaf=20)
    return _NumpyLogit()


class MLFilter:
    """Per (symbol, timeframe) model cache with periodic retraining."""

    def __init__(self, horizon: int = 8, retrain_every: int = 25, min_samples: int = 150):
        self.horizon = horizon
        self.retrain_every = retrain_every
        self.min_samples = min_samples
        self._cache: dict = {}
        self._lock = threading.Lock()

    def _train(self, X: pd.DataFrame, y: pd.Series):
        mask = y.notna() & X.notna().all(axis=1)
        Xv, yv = X[mask].values, y[mask].values
        if len(yv) < self.min_samples or len(np.unique(yv)) < 2:
            return None
        split = int(len(yv) * 0.8)
        test_start = split + self.horizon      # purge: labels of the last train rows overlap the test window
        acc, acc_lb = 0.5, 0.5
        if split > 50 and len(np.unique(yv[:split])) == 2 and len(yv) - test_start > 20:
            m = _make_model().fit(Xv[:split], yv[:split])
            pred = m.predict_proba(Xv[test_start:])[:, 1] > 0.5
            n_test = len(yv) - test_start
            acc = float((pred == (yv[test_start:] > 0.5)).mean())
            acc_lb = acc - 1.0 * np.sqrt(0.25 / n_test)   # ~1 std error: don't trust lucky holdouts
        model = _make_model().fit(Xv, yv)
        return model, acc, acc_lb

    def prob_up(self, key, df: pd.DataFrame):
        """Probability (0..1) that price is higher in `horizon` bars, plus holdout accuracy."""
        X = build_features(df)
        y = build_labels(df, self.horizon)
        y.iloc[-self.horizon:] = np.nan  # no future leakage
        with self._lock:
            entry = self._cache.get(key)
            last_t = df.index[-1]
            # retrain after `retrain_every` NEW bars (the window length stays fixed, so count by time)
            stale = entry is None or last_t < entry["t"] or \
                int((df.index > entry["t"]).sum()) >= self.retrain_every
            if stale:
                trained = self._train(X, y)
                if trained is None:
                    return None, 0.5
                entry = {"model": trained[0], "acc": trained[1], "acc_lb": trained[2], "t": last_t}
                self._cache[key] = entry
        x_last = X.iloc[[-1]]
        if x_last.isna().any(axis=1).iloc[0]:
            return None, entry["acc"]
        p = float(entry["model"].predict_proba(x_last.values)[0, 1])
        # shrink toward 0.5 when the model has no validated edge
        edge = float(np.clip((entry["acc_lb"] - 0.5) / 0.1, 0.0, 1.0))
        return 0.5 + (p - 0.5) * edge, entry["acc"]
