"""The robot: scans charts, plans trades, notifies, optionally auto-trades,
and manages open positions. Runs in a background thread; talks to the UI
only through the `events` callback object (Qt-agnostic)."""
from __future__ import annotations

import queue
import threading
import time
import traceback
from datetime import datetime, timezone

import numpy as np

from ..brokers.paper import PaperBroker
from ..config import TF_SECONDS
from ..core import indicators as ind
from ..core.analyzer import FREQUENCY_SHIFT, Analyzer
from ..core.luxalgo import signals_overlays
from ..core.ml import MLFilter
from ..core.risk import position_size, sl_safe_with_leverage
from ..core.smc import smc
from ..core.sp2l import sp2l
from ..core.strategies import STRATEGIES, cached, strategies_for_mode
from ..core.tuner import AutoTuner, TuneResult, autopilot_params
from ..data.feed import make_feed

FRESH_BARS = {"quality": 1, "balanced": 2, "high": 3}
TUNE_BARS = 1500


class NullEvents:
    def log(self, msg): print(msg)
    def chart(self, symbol, payload): pass
    def signal(self, sig): pass
    def positions(self, positions): pass
    def trade_event(self, ev): pass
    def account(self, info): pass
    def status(self, text): pass
    def tuning(self, info): pass


class Robot:
    def __init__(self, settings, db, events=None):
        self.db = db
        self.events = events or NullEvents()
        self.ml = MLFilter()
        self.settings = settings.copy()
        self.scanning = False
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.RLock()
        self._seen = {}
        self._hl = {}                       # symbol -> (bar_ts, high, low, checked_at)
        self._exec_lock = threading.Lock()
        self._thread = None
        self.feed = None
        self.broker = None
        self.last_df = {}
        self.tuner = AutoTuner()
        self.tuned = self._load_tuning()   # "SYM|TF|MODE|MARKET" -> {"at", "results": {key: TuneResult}}
        self._tune_q = queue.Queue()
        self._tune_pending = set()
        self._tune_thread = None
        self._scan_started = 0.0
        self._last_hb = 0.0
        self._stats = {"signals": 0, "filtered": 0, "opened": 0, "skipped": 0}
        self._build()

    # ------------------------------------------------------------ config
    @staticmethod
    def _heavy_key(s):
        c = s.creds
        return (s.data_exchange, s.market_type, s.account, c.exchange, c.api_key, c.secret,
                c.password, c.testnet, s.exchange_side_sl)

    def needs_rebuild(self, settings) -> bool:
        return self._heavy_key(self.settings) != self._heavy_key(settings)

    def _make(self, s):
        """Create (feed, broker) for settings. Network calls happen here: never under the lock."""
        if s.account == "live":
            from ..brokers.live import LiveBroker
            broker = LiveBroker(self.db, s.market_type, s.creds, s.exchange_side_sl, self.events.log)
            # prices for SL/TP management and signals must come from the venue we trade on
            feed = make_feed(s.creds.exchange, s.market_type, client=broker.ex)
            if s.data_exchange != s.creds.exchange or s.creds.testnet:
                self.events.log(f"Live account: using {s.creds.exchange}"
                                f"{' TESTNET' if s.creds.testnet else ''} prices for charts and signals.")
        else:
            feed = make_feed(s.data_exchange, s.market_type)
            broker = PaperBroker(self.db, s.market_type, s.paper_start_balance, log=self.events.log)
        return feed, broker

    def _build(self):
        s = self.settings
        self.feed, self.broker = self._make(s)
        self.analyzer = Analyzer(s.strategy, self.ml, s.use_ml, s.mode)

    def reconfigure(self, settings):
        """Apply new settings. May block on the network (call from a worker thread when
        switching accounts). Raises if the new broker can't be created; the old one stays."""
        new = settings.copy()
        if self.needs_rebuild(new):
            feed, broker = self._make(new)
            with self._lock:
                self.settings, self.feed, self.broker = new, feed, broker
                self.analyzer = Analyzer(new.strategy, self.ml, new.use_ml, new.mode)
                self.last_df.clear()
                self._hl.clear()
        else:
            with self._lock:
                old = self.settings
                self.settings = new
                self.analyzer = Analyzer(new.strategy, self.ml, new.use_ml, new.mode)
                if old.timeframe != new.timeframe:
                    self.last_df.clear()
                    self._hl.clear()
        self._wake.set()

    # ------------------------------------------------------------ control
    def start_engine(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="robot", daemon=True)
        self._thread.start()
        self._tune_thread = threading.Thread(target=self._tune_loop, name="tuner", daemon=True)
        self._tune_thread.start()

    def shutdown(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def set_scanning(self, on: bool):
        self.scanning = on
        self.events.status("Robot running" if on else "Robot stopped (monitoring positions)")
        if on:
            s = self.settings
            self._scan_started = time.time()
            self._last_hb = time.time()
            self._stats = {"signals": 0, "filtered": 0, "opened": 0, "skipped": 0}
            strat = STRATEGIES.get(s.strategy)
            self.events.log(f"Scanning {len(s.symbols)} coin(s) on {s.timeframe} with "
                            f"{strat.name if strat else s.strategy}. Signals come on CANDLE CLOSE "
                            f"(every {s.timeframe}); arrows already on the chart are history.")
            if s.trade_mode == "notify":
                self.events.log("Semi-auto mode: I alert you and the trade opens only when you click "
                                "Open in the Signals tab. Choose 'Full auto' to let the robot open/close by itself.")
            else:
                self.events.log(f"Full auto: the robot opens trades itself (max {s.max_positions} open, "
                                f"{s.risk_pct}% risk each).")
        self._wake.set()

    # ------------------------------------------------------------ auto-tuning
    @staticmethod
    def _tkey(sym, s):
        return f"{sym}|{s.timeframe}|{s.mode}|{s.market_type}"

    def _load_tuning(self) -> dict:
        out = {}
        try:
            raw = self.db.get("tuning_v3", {}) or {}
            for k, e in raw.items():
                out[k] = {"at": float(e["at"]),
                          "results": {sk: TuneResult.from_dict(r) for sk, r in e["results"].items()}}
        except Exception:  # noqa: BLE001  (old / corrupt cache: just retune)
            out = {}
        return out

    def _save_tuning(self):
        try:
            data = {k: {"at": e["at"], "results": {sk: r.to_dict() for sk, r in e["results"].items()}}
                    for k, e in list(self.tuned.items())[-200:]}
            self.db.set("tuning_v3", data)
        except Exception as e:  # noqa: BLE001
            self.events.log(f"Could not save tuning cache: {e}")

    def _tune_keys(self, s):
        if s.strategy == "auto_pilot":
            return [st.key for st in strategies_for_mode(s.mode) if st.key != "auto_pilot"]
        st = STRATEGIES.get(s.strategy)
        return [s.strategy] if st and st.space else []

    def _retune_after(self, tf):
        return min(max(200 * TF_SECONDS.get(tf, 900), 4 * 3600), 7 * 86400)

    def _maybe_tune(self, sym, s):
        if not s.auto_tune:
            return
        keys = self._tune_keys(s)
        if not keys:
            return
        k = self._tkey(sym, s)
        e = self.tuned.get(k)
        fresh = e and time.time() - e["at"] < self._retune_after(s.timeframe) and all(x in e["results"] for x in keys)
        if fresh or k in self._tune_pending:
            return
        self._tune_pending.add(k)
        self._tune_q.put((sym, s.copy(), keys))

    def retune(self, symbols=None):
        """Forget cached tuning (all or some coins) so it runs again now."""
        with self._lock:
            s = self.settings
            for sym in (symbols or s.symbols):
                self.tuned.pop(self._tkey(sym, s), None)
        self._save_tuning()
        self._wake.set()

    def _tune_loop(self):
        while not self._stop.is_set():
            try:
                sym, s, keys = self._tune_q.get(timeout=1.0)
            except queue.Empty:
                continue
            k = self._tkey(sym, s)
            try:
                with self._lock:
                    feed, cur = self.feed, self.settings
                if self._tkey(sym, cur) != k:      # settings changed while queued
                    continue
                self.events.log(f"Auto-tuning {sym} {s.timeframe} ({s.mode}): testing indicator settings on "
                                f"the last {TUNE_BARS} candles…")
                df = feed.fetch_ohlcv(sym, s.timeframe, TUNE_BARS).iloc[:-1]
                lev = s.leverage if s.market_type == "futures" else 1
                res = self.tuner.tune_mode(df, s.mode, keys, s.market_type, lev, stop=self._stop.is_set)
                if self._stop.is_set():
                    return
                self.tuned[k] = {"at": time.time(), "results": res}
                self._save_tuning()
                ok = [r for r in res.values() if r.status == "validated"]
                bad = [r for r in res.values() if r.status == "failed"]
                parts = []
                for r in sorted(res.values(), key=lambda r: -r.test.get("exp_r", -9)):
                    ch = ", ".join(f"{a}={b}" for a, b in r.changed.items()) or "preset"
                    parts.append(f"{STRATEGIES[r.strategy].name}: {r.status} ({ch})")
                self.events.log(f"Tuned {sym} {s.timeframe}: {len(ok)} validated, {len(bad)} muted. "
                                + " | ".join(parts[:4]))
                self.events.tuning({"key": k, "symbol": sym, "timeframe": s.timeframe, "mode": s.mode,
                                    "results": res})
                self._wake.set()
            except Exception as e:  # noqa: BLE001
                self.events.log(f"Auto-tune {sym} failed (using style presets): {e}")
                self.tuned[k] = {"at": time.time() - self._retune_after(s.timeframe) + 1800, "results": {}}
            finally:
                self._tune_pending.discard(k)

    def params_for(self, sym, s=None):
        """Strategy params for this coin: tuned when available, otherwise the style preset."""
        s = s or self.settings
        e = self.tuned.get(self._tkey(sym, s)) if s.auto_tune else None
        res = e["results"] if e else {}
        if s.strategy == "auto_pilot":
            return autopilot_params(res, s.mode, s.signal_frequency)
        r = res.get(s.strategy)
        return r.params if r else None

    def tuning_table(self):
        s = self.settings
        rows = []
        for sym in s.symbols:
            e = self.tuned.get(self._tkey(sym, s))
            if not e:
                continue
            for r in e["results"].values():
                rows.append((sym, s.timeframe, r, e["at"]))
        return rows

    # ------------------------------------------------------------ loop
    def _run(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.tick()
            except Exception as e:
                self.events.log(f"Engine error: {e}\n{traceback.format_exc(limit=2)}")
            wait = max(1.0, self.settings.poll_seconds - (time.time() - t0))
            self._wake.wait(wait)
            self._wake.clear()

    def _range_since_last(self, sym, df):
        """High/low traded since the previous tick (None on first look)."""
        cur_ts, h, l = df.index[-1], float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
        prev = self._hl.get(sym)
        now = time.time()
        self._hl[sym] = (cur_ts, h, l, now)
        if prev is None:
            return None, None, None
        p_ts, p_h, p_l, p_at = prev
        newer = df[df.index > p_ts]
        his, los = list(newer["high"]), list(newer["low"])
        if p_ts in df.index:
            if df.at[p_ts, "high"] > p_h:
                his.append(float(df.at[p_ts, "high"]))
            if df.at[p_ts, "low"] < p_l:
                los.append(float(df.at[p_ts, "low"]))
        return (max(his) if his else None), (min(los) if los else None), p_at

    def tick(self):
        with self._lock:
            s, feed, broker, analyzer = self.settings, self.feed, self.broker, self.analyzer
        try:
            for ev in broker.sync():
                self.events.trade_event(ev)
        except Exception as e:
            self.events.log(f"Position sync error: {e}")
        open_syms = [p.symbol for p in broker.open_positions()]
        symbols = list(dict.fromkeys(s.symbols + open_syms))
        for sym in symbols:
            if self._stop.is_set():
                return
            try:
                df = feed.fetch_ohlcv(sym, s.timeframe, 600)
            except Exception as e:
                self.events.log(f"{sym}: data error: {e}")
                if sym in open_syms:          # still protect the open position
                    try:
                        px = feed.fetch_price(sym, s.timeframe)
                        for ev in broker.on_price(sym, px):
                            self.events.trade_event(ev)
                    except Exception as e2:
                        self.events.log(f"{sym}: price error, position NOT checked: {e2}")
                continue
            price = float(df["close"].iloc[-1])
            hi, lo, since = self._range_since_last(sym, df)
            for ev in broker.on_price(sym, price, hi, lo, since):
                self.events.trade_event(ev)
            if len(df) < 250:
                self.events.log(f"{sym}: not enough history ({len(df)} bars)")
                continue
            closed = df.iloc[:-1]           # last candle is still forming
            sig, sigframe = None, None
            params = None
            if sym in s.symbols:
                self._maybe_tune(sym, s)
                params = self.params_for(sym, s)
                try:
                    sig, sigframe = analyzer.latest_signal(closed, sym, s.timeframe,
                                                           FRESH_BARS.get(s.signal_frequency, 2), params)
                except Exception as e:
                    self.events.log(f"{sym}: analysis error: {e}")
                if sig and self.scanning:
                    self._handle_signal(sig, price)
            self.last_df[sym] = df
            try:
                payload = self._chart_payload(sym, df, sigframe, s)
            except Exception as e:  # noqa: BLE001  (never let a drawing problem stop trading)
                self.events.log(f"{sym}: chart overlay error: {e}")
                payload = self._chart_payload(sym, df, sigframe, s, overlays=False)
            self.events.chart(sym, payload)
        self.events.positions(broker.open_positions())
        self._heartbeat(s, broker)
        try:
            self.events.account({"balance": broker.balance(), "equity": broker.equity(),
                                 "account": s.account,
                                 "testnet": bool(s.account == "live" and s.creds.testnet)})
        except Exception as e:
            self.events.log(f"Balance error: {e}")

    def _heartbeat(self, s, broker):
        if not self.scanning or time.time() - self._last_hb < 300:
            return
        self._last_hb = time.time()
        st = self._stats
        tuned = sum(1 for sym in s.symbols if self._tkey(sym, s) in self.tuned)
        nxt = TF_SECONDS.get(s.timeframe, 900) - time.time() % TF_SECONDS.get(s.timeframe, 900)
        self.events.log(f"Robot alive: {len(s.symbols)} coins on {s.timeframe}, tuned {tuned}/{len(s.symbols)} · "
                        f"signals {st['signals']} (filtered {st['filtered']}) · opened {st['opened']} · "
                        f"skipped {st['skipped']} · open now {len(broker.open_positions())} · "
                        f"next candle close in {int(nxt // 60)}m{int(nxt % 60):02d}s")

    # ------------------------------------------------------------ signals
    def min_confidence(self, s=None):
        s = s or self.settings
        return float(np.clip(s.min_confidence + FREQUENCY_SHIFT.get(s.signal_frequency, 0.0), 0.0, 1.0))

    def _handle_signal(self, sig, price):
        key = (sig.symbol, sig.timeframe, sig.bar_time, sig.side)
        if key in self._seen:
            return
        self._seen[key] = time.time()
        if len(self._seen) > 2000:          # keep memory bounded
            cutoff = time.time() - 3 * 86400
            self._seen = {k: t for k, t in self._seen.items() if t >= cutoff}
        s = self.settings
        gate = self.min_confidence(s)
        if "ML veto" in sig.reasons or sig.confidence < gate:
            self._stats["filtered"] += 1
            why = "ML veto" if "ML veto" in sig.reasons else f"confidence {sig.confidence:.0%} < {gate:.0%}"
            self.events.log(f"{sig.symbol} {sig.side} ({sig.strategy}) filtered: {why}")
            return
        self._stats["signals"] += 1
        self.db.add_signal(sig)
        self.events.signal(sig)
        if s.trade_mode == "auto":
            try:
                self.execute_signal(sig, price, auto=True)
                self._stats["opened"] += 1
            except Exception as e:
                self._stats["skipped"] += 1
                self.events.log(f"Auto-trade skipped for {sig.symbol}: {e}")

    def daily_loss_hit(self) -> bool:
        s = self.settings
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        realized = self.db.realized_since(self.broker.account, start)
        eq = (self.broker.equity() or 0) - realized     # equity at the start of the day (approx.)
        if eq <= 0:
            return realized < 0
        return realized < 0 and -realized / eq * 100 >= s.max_daily_loss_pct

    def execute_signal(self, sig, price=None, auto=False):
        """Open a trade from a signal (robot auto-trade or user click). Serialized so two
        threads can't both pass the max-positions / one-per-coin checks."""
        with self._exec_lock:
            return self._execute(sig, price, auto)

    def _execute(self, sig, price=None, auto=False):
        with self._lock:
            s, broker, feed = self.settings, self.broker, self.feed
        if sig.timeframe and sig.timeframe != s.timeframe:
            raise RuntimeError(f"signal is for {sig.timeframe} but the robot now runs {s.timeframe}")
        if s.market_type == "spot" and sig.side == "short":
            raise RuntimeError("short signals can't be traded on spot")
        if len(broker.open_positions()) >= s.max_positions:
            raise RuntimeError(f"max open positions ({s.max_positions}) reached")
        if any(p.symbol == sig.symbol for p in broker.open_positions()):
            raise RuntimeError(f"already in a {sig.symbol} position")
        if self.daily_loss_hit():
            raise RuntimeError("daily loss limit reached, trading paused until tomorrow (UTC)")
        price = price or feed.fetch_price(sig.symbol, s.timeframe)
        # signal must still be valid at the current price
        if (sig.side == "long" and price <= sig.sl) or (sig.side == "short" and price >= sig.sl):
            raise RuntimeError("price already beyond stop loss, signal expired")
        if (sig.side == "long" and price >= sig.tps[0]) or (sig.side == "short" and price <= sig.tps[0]):
            raise RuntimeError("price already passed TP1, signal expired")
        lev = s.leverage if s.market_type == "futures" else 1
        if not sl_safe_with_leverage(sig.side, price, sig.sl, lev):
            raise RuntimeError(f"stop loss is beyond liquidation at {lev}x, lower leverage")
        qty = position_size(broker.equity() if s.market_type == "futures" else broker.balance(),
                            s.risk_pct, price, sig.sl, s.market_type, lev, fee=0.0006,
                            free_balance=broker.balance())
        if qty <= 0:
            raise RuntimeError("position size is zero (check balance)")
        pos = broker.open_from_signal(sig, qty, lev, price)
        self.db.mark_signal(sig.id)
        self.events.trade_event({"type": "opened", "pos": pos, "reason": "auto" if auto else "manual",
                                 "price": pos.entry, "qty": pos.qty, "pnl": 0.0})
        self.events.positions(broker.open_positions())
        return pos

    def close_position(self, pos_id: str):
        from ..brokers.base import ExecError
        with self._lock:
            broker, feed, tf = self.broker, self.feed, self.settings.timeframe
        pos = broker.positions.get(pos_id)
        if not pos:
            return
        price = feed.fetch_price(pos.symbol, tf)
        try:
            for ev in broker.close(pos_id, price, "manual"):
                self.events.trade_event(ev)
        except ExecError as e:
            raise RuntimeError(f"close order failed, position is still open: {e}") from e
        self.events.positions(broker.open_positions())

    # ------------------------------------------------------------ chart data
    def _chart_payload(self, sym, df, sigframe, s, overlays=True):
        c = df["close"]
        st_line, st_dir = ind.supertrend(df)
        payload = {"symbol": sym, "timeframe": s.timeframe, "df": df,
                   "ema20": ind.ema(c, 20), "ema50": ind.ema(c, 50), "ema200": ind.ema(c, 200),
                   "supertrend": st_line, "st_dir": st_dir, "rsi": ind.rsi(c),
                   "markers": [], "positions": [p for p in self.broker.open_positions() if p.symbol == sym],
                   "show": {"smc": s.show_smc, "lux": s.show_lux, "sp2l": s.show_sp2l, "emas": s.show_emas,
                            "osc": s.osc_pane}}
        if sigframe is not None:
            n = len(sigframe)
            since = self._scan_started if self.scanning else float("inf")
            for i in range(max(0, n - 400), n):
                live = df.index[i].timestamp() >= since - TF_SECONDS.get(s.timeframe, 900)
                if sigframe["long"].iloc[i]:
                    payload["markers"].append((i, "long", float(df["low"].iloc[i]), live))
                elif sigframe["short"].iloc[i]:
                    payload["markers"].append((i, "short", float(df["high"].iloc[i]), live))
        if not overlays:
            return payload
        members = (self.params_for(sym, s) or {}).get("members", {}) if s.strategy == "auto_pilot" else {}
        own = self.params_for(sym, s) if s.strategy != "auto_pilot" else None

        def tuned(key):
            if s.strategy == key and own:
                return STRATEGIES[key].params_for(s.mode, own)
            return STRATEGIES[key].params_for(s.mode, members.get(key))
        q = tuned("squeeze")
        sq = cached("sqz", df, ind.squeeze_lazybear, length=int(q["length"]), mult=float(q["mult"]),
                    length_kc=int(q["length_kc"]), mult_kc=float(q["mult_kc"]))
        payload["sqz"] = {"val": sq["val"].values, "bar": sq["bar_color"].values, "dot": sq["dot_color"].values}
        if s.show_smc:
            m = tuned("smc")
            payload["smc"] = cached("smc", df, smc, swing_len=int(m["swing_len"]),
                                    internal_len=int(m["internal_len"]))["overlay"]
        if s.show_lux:
            x_ = tuned("lux_signals")
            x = cached("lux", df, signals_overlays, sensitivity=float(x_["sensitivity"]),
                       atr_len=int(x_["atr_len"]), trail_mult=float(x_["trail_mult"]))
            payload["lux"] = {"trail": x["trail"].values, "trail_dir": x["trail_dir"].values,
                              "catcher": x["catcher"].values, "catcher_dir": x["catcher_dir"].values,
                              "up_in": x["rz"]["up_in"].values, "up_out": x["rz"]["up_out"].values,
                              "dn_in": x["rz"]["dn_in"].values, "dn_out": x["rz"]["dn_out"].values,
                              "cloud_a": x["cloud_a"].values, "cloud_b": x["cloud_b"].values,
                              "buy": np.where(x["buy"].values)[0], "sell": np.where(x["sell"].values)[0],
                              "strong_buy": np.where(x["strong_buy"].values)[0],
                              "strong_sell": np.where(x["strong_sell"].values)[0]}
        if s.show_sp2l:
            p2 = tuned("sp2l")
            r = cached("sp2l", df.iloc[:-1], sp2l, min_bars=int(p2["min_bars"]), power=float(p2["power"]),
                       gap_filter=bool(p2["gap_filter"]), expiry=int(p2["expiry"]),
                       doji_body=float(p2["doji_body"]), trend_filter=bool(p2["trend_filter"]))
            payload["sp2l"] = r["setups"]
        return payload
