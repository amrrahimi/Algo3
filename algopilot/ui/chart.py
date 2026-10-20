"""TradingView-style resizable candlestick charts (pyqtgraph)."""
from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPicture
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from .theme import BG, DN, UP

pg.setConfigOptions(antialias=True, background=BG, foreground="#8f96a3")


class CandleItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self._pic = QPicture()
        self._bounds = QRectF()

    def set_data(self, o, h, l, c):
        pic = QPicture()
        p = QPainter(pic)
        w = 0.35
        pens = {True: pg.mkPen(UP), False: pg.mkPen(DN)}
        brushes = {True: pg.mkBrush(UP), False: pg.mkBrush(DN)}
        for i in range(len(c)):
            up = bool(c[i] >= o[i])
            p.setPen(pens[up])
            p.setBrush(brushes[up])
            p.drawLine(QPointF(i, float(l[i])), QPointF(i, float(h[i])))
            top, bot = float(max(o[i], c[i])), float(min(o[i], c[i]))
            p.drawRect(QRectF(i - w, bot, 2 * w, max(top - bot, 1e-12)))
        p.end()
        self.prepareGeometryChange()
        self._pic = pic
        lo, hi = float(np.nanmin(l)), float(np.nanmax(h))
        self._bounds = QRectF(-1.0, lo, len(c) + 1.0, hi - lo)
        self.update()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self._pic)

    def boundingRect(self):
        return self._bounds


class TimeAxis(pg.AxisItem):
    def __init__(self):
        super().__init__(orientation="bottom")
        self.times = []
        self.fmt = "%d %b %H:%M"

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            i = int(round(v))
            out.append(self.times[i].strftime(self.fmt) if 0 <= i < len(self.times) else "")
        return out


class ChartWidget(QWidget):
    symbolChanged = Signal(object, str)
    maximizeRequested = Signal(object)

    def __init__(self, symbols: list, symbol: str, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.n = 0
        self.df = None
        self.levels = []
        self.follow = True
        self._pos_items = []

        top = QHBoxLayout()
        top.setContentsMargins(4, 2, 4, 2)
        self.combo = QComboBox()
        self.set_symbols(symbols, symbol)
        self.combo.currentTextChanged.connect(self._on_combo)
        self.price_lbl = QLabel("-")
        self.price_lbl.setStyleSheet("font-weight:bold; font-size:14px;")
        self.info_lbl = QLabel("")
        self.info_lbl.setStyleSheet("color:#8f96a3;")
        self.max_btn = QPushButton("⤢")
        self.max_btn.setFixedWidth(30)
        self.max_btn.setToolTip("Maximize / restore this chart")
        self.max_btn.clicked.connect(lambda: self.maximizeRequested.emit(self))
        top.addWidget(self.combo)
        top.addWidget(self.price_lbl)
        top.addWidget(self.info_lbl, 1)
        top.addWidget(self.max_btn)

        self.glw = pg.GraphicsLayoutWidget()
        self.axis, self.axis2 = TimeAxis(), TimeAxis()
        self.price = self.glw.addPlot(row=0, col=0, axisItems={"bottom": self.axis})
        self.osc = self.glw.addPlot(row=1, col=0, axisItems={"bottom": self.axis2})
        self.glw.ci.layout.setRowStretchFactor(0, 4)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        self.osc.setXLink(self.price)
        for plot in (self.price, self.osc):
            plot.showAxis("right")
            plot.hideAxis("left")
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.setMouseEnabled(x=True, y=False)
            plot.setMenuEnabled(False)
        self.price.hideAxis("bottom")
        self.osc.setYRange(-100, 100)
        self.osc.setMaximumHeight(180)

        self.candles = CandleItem()
        self.price.addItem(self.candles)
        self.ema20 = self.price.plot(pen=pg.mkPen("#f6c343", width=1), name="EMA20")
        self.ema50 = self.price.plot(pen=pg.mkPen("#42a5f5", width=1), name="EMA50")
        self.ema200 = self.price.plot(pen=pg.mkPen("#ab47bc", width=1.5), name="EMA200")
        self.st_up = self.price.plot(pen=pg.mkPen(UP, width=1.5, style=Qt.PenStyle.DotLine))
        self.st_dn = self.price.plot(pen=pg.mkPen(DN, width=1.5, style=Qt.PenStyle.DotLine))
        self.long_m = pg.ScatterPlotItem(symbol="t1", size=13, brush=pg.mkBrush(UP), pen=pg.mkPen(None))
        self.short_m = pg.ScatterPlotItem(symbol="t", size=13, brush=pg.mkBrush(DN), pen=pg.mkPen(None))
        self.price.addItem(self.long_m)
        self.price.addItem(self.short_m)
        self.rsi = self.osc.plot(pen=pg.mkPen("#b39ddb", width=1.2))
        for y in (30, 70):
            self.osc.addLine(y=y, pen=pg.mkPen("#555", style=Qt.PenStyle.DashLine))
        self.sqz_hist = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush=pg.mkBrush("#888"))
        self.sqz_dots = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None))
        self.osc.addItem(self.sqz_hist)
        self.osc.addItem(self.sqz_dots)
        self.lux_items = []          # pyqtgraph items for overlays (rebuilt each update)
        self.sp2l_items = []
        self.smc_items = []

        dash = pg.mkPen("#787b86", style=Qt.PenStyle.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=dash)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=dash,
                                     label="{value:.6g}", labelOpts={"position": 0.98, "color": "#d1d4dc",
                                                                      "fill": (42, 46, 57, 220)})
        self.price.addItem(self.vline, ignoreBounds=True)
        self.price.addItem(self.hline, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.price.scene().sigMouseMoved, rateLimit=60, slot=self._mouse)
        self.price.getViewBox().sigXRangeChanged.connect(self._autoscale_y)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(top)
        lay.addWidget(self.glw, 1)

    # ------------------------------------------------------------------
    def set_symbols(self, symbols, current=None):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(symbols)
        cur = current or self.symbol
        if cur in symbols:
            self.combo.setCurrentText(cur)
        self.combo.blockSignals(False)

    def _on_combo(self, sym):
        if sym and sym != self.symbol:
            self.symbol = sym
            self.reset()
            self.symbolChanged.emit(self, sym)

    def reset(self):
        self.n = 0
        self.df = None
        self.price_lbl.setText("loading…")

    def _mouse(self, evt):
        pos = evt[0]
        if self.df is None or not self.price.sceneBoundingRect().contains(pos):
            return
        mp = self.price.getViewBox().mapSceneToView(pos)
        self.vline.setPos(mp.x())
        self.hline.setPos(mp.y())
        i = int(round(mp.x()))
        if 0 <= i < self.n:
            r = self.df.iloc[i]
            chg = (r.close / r.open - 1) * 100 if r.open else 0
            self.info_lbl.setText(f"O {r.open:.6g}  H {r.high:.6g}  L {r.low:.6g}  C {r.close:.6g}  "
                                  f"({chg:+.2f}%)  V {r.volume:,.0f}")

    def _autoscale_y(self, *args):
        if self.df is None or self.n == 0:
            return
        vb = self.price.getViewBox()
        xr = vb.viewRange()[0]
        self.follow = xr[1] >= self.n - 3
        i0, i1 = max(0, int(xr[0])), min(self.n, int(xr[1]) + 1)
        if i1 - i0 < 2:
            return
        lo = float(np.nanmin(self.df["low"].values[i0:i1]))
        hi = float(np.nanmax(self.df["high"].values[i0:i1]))
        for lv in self.levels:
            if lo - (hi - lo) * 0.6 < lv < hi + (hi - lo) * 0.6:
                lo, hi = min(lo, lv), max(hi, lv)
        pad = (hi - lo) * 0.07 or hi * 0.01
        vb.setYRange(lo - pad, hi + pad, padding=0)

    def update_data(self, payload: dict):
        df = payload["df"]
        if df is None or len(df) == 0:
            return
        first = self.n == 0
        self.df = df
        self.n = n = len(df)
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        self.candles.set_data(o, h, l, c)
        x = np.arange(n, dtype=float)
        self.ema20.setData(x, payload["ema20"].values, connect="finite")
        self.ema50.setData(x, payload["ema50"].values, connect="finite")
        self.ema200.setData(x, payload["ema200"].values, connect="finite")
        st, d = payload["supertrend"].values, payload["st_dir"].values
        self.st_up.setData(x, np.where(d == 1, st, np.nan), connect="finite")
        self.st_dn.setData(x, np.where(d == -1, st, np.nan), connect="finite")
        show = payload.get("show", {})
        emas = show.get("emas", True)
        self.ema20.setVisible(emas)
        self.ema50.setVisible(emas)
        self.ema200.setVisible(emas)
        if show.get("osc", "squeeze") == "rsi" or "sqz" not in payload:
            self.rsi.setVisible(True)
            self.rsi.setData(x, payload["rsi"].values, connect="finite")
            self.sqz_hist.setOpts(height=[0])
            self.sqz_dots.setData([], [])
            self.osc.setYRange(0, 100)
        else:
            self.rsi.setVisible(False)
            q = payload["sqz"]
            val = np.nan_to_num(q["val"])
            cols = {2: "#00e676", 1: "#2e7d32", -1: "#e53935", -2: "#7f1d1d", 0: "#888"}
            brushes = [pg.mkBrush(cols.get(int(b), "#888")) for b in q["bar"]]
            self.sqz_hist.setOpts(x=x, height=val, width=0.6, brushes=brushes, pens=[pg.mkPen(None)] * n)
            dcol = {0: "#000000", 1: "#9e9e9e", 2: "#2962ff"}
            self.sqz_dots.setData(x, np.zeros(n), brush=[pg.mkBrush(dcol.get(int(d), "#2962ff")) for d in q["dot"]])
            ylim = float(np.nanmax(np.abs(val))) * 1.2 if np.isfinite(val).any() else 1
            self.osc.setYRange(-max(ylim, 1e-9), max(ylim, 1e-9))
        off = float(np.nanmean(h[-100:] - l[-100:])) * 0.8
        lm = [(i, p - off) for i, s, p, *_ in payload["markers"] if s == "long"]
        sm = [(i, p + off) for i, s, p, *_ in payload["markers"] if s == "short"]
        self.long_m.setData([a for a, _ in lm], [b for _, b in lm])
        self.short_m.setData([a for a, _ in sm], [b for _, b in sm])
        self._draw_overlays(payload, x, df, show)
        self._draw_positions(payload.get("positions", []))

        local = [t.to_pydatetime().astimezone() for t in df.index]
        self.axis.times = self.axis2.times = local
        fmt = "%Y-%m-%d" if payload.get("timeframe") in ("1d",) else "%d %b %H:%M"
        self.axis.fmt = self.axis2.fmt = fmt

        last, prev = c[-1], c[-2] if n > 1 else c[-1]
        color = UP if last >= prev else DN
        self.price_lbl.setText(f"{last:.6g}")
        self.price_lbl.setStyleSheet(f"font-weight:bold; font-size:14px; color:{color};")
        if first:
            self.price.getViewBox().setXRange(max(0, n - 150), n + 8, padding=0)
        self._autoscale_y()

    def _draw_overlays(self, payload, x, df, show):
        for it in self.lux_items + self.sp2l_items + self.smc_items:
            self.price.removeItem(it)
        self.lux_items, self.sp2l_items, self.smc_items = [], [], []
        n = len(df)
        if show.get("smc") and payload.get("smc"):
            sm = payload["smc"]
            for (x0, top, bot, bull, kind) in sm["obs"]:
                rgba = (38, 166, 154, 40) if bull else (239, 83, 80, 40)
                r = pg.LinearRegionItem(values=(bot, top), orientation="horizontal", movable=False,
                                        brush=pg.mkBrush(rgba), pen=pg.mkPen(None))
                r.setZValue(-20)
                self.price.addItem(r, ignoreBounds=True)
                self.smc_items.append(r)
            for (x0, top, bot, bull) in payload["smc"]["fvgs"]:
                r = pg.LinearRegionItem(values=(bot, top), orientation="horizontal", movable=False,
                                        brush=pg.mkBrush((41, 98, 255, 30)), pen=pg.mkPen(None))
                r.setZValue(-25)
                self.price.addItem(r, ignoreBounds=True)
                self.smc_items.append(r)
            for (x1, x2, lvl, kind) in payload["smc"]["eqs"][-4:]:
                ln = pg.PlotDataItem([x1, x2], [lvl, lvl], pen=pg.mkPen("#ffd54f", width=1, style=Qt.PenStyle.DotLine))
                self.price.addItem(ln)
                self.smc_items.append(ln)
            for (bar, from_x, lvl, kind, bull, lv) in payload["smc"]["events"][-8:]:
                if from_x is None or bar >= n:
                    continue
                ln = pg.PlotDataItem([from_x, bar], [lvl, lvl], pen=pg.mkPen("#9e9e9e", width=1, style=Qt.PenStyle.DashLine))
                self.price.addItem(ln)
                self.smc_items.append(ln)
                t = pg.TextItem(kind, color="#d1d4dc", anchor=(0.5, 1.0))
                t.setPos(bar, lvl)
                self.price.addItem(t)
                self.smc_items.append(t)
        if show.get("lux") and payload.get("lux"):
            lx = payload["lux"]
            tr = lx["trail"]
            ok = ~np.isnan(tr)
            up = np.where(ok & (lx["trail_dir"] == 1), tr, np.nan)
            dn = np.where(ok & (lx["trail_dir"] == -1), tr, np.nan)
            for arr, col in ((up, UP), (dn, DN)):
                it = pg.PlotDataItem(x, arr, pen=pg.mkPen(col, width=1.2), connect="finite")
                self.price.addItem(it)
                self.lux_items.append(it)
            cd = lx["catcher_dir"]
            for sgn, col in ((1, "#26a69a"), (-1, "#ef5350")):
                arr = np.where(cd == sgn, lx["catcher"], np.nan)
                it = pg.PlotDataItem(x, arr, pen=pg.mkPen(col, width=1.6), connect="finite")
                self.price.addItem(it)
                self.lux_items.append(it)
            for lo, hi in (("dn_out", "dn_in"), ("up_in", "up_out")):
                for a_, b_ in ((lx[lo], lx[hi]),):
                    rg = pg.FillBetweenItem(pg.PlotDataItem(x, a_), pg.PlotDataItem(x, b_),
                                            brush=pg.mkBrush(120, 120, 255, 22))
                    self.price.addItem(rg)
                    self.lux_items.append(rg)
            for idx, sym, col in ((lx["buy"], "t1", UP), (lx["sell"], "t", DN)):
                sc = pg.ScatterPlotItem(x=idx, y=df["low" if sym == "t1" else "high"].values[idx] if len(idx) else [],
                                        symbol=sym, size=9, brush=pg.mkBrush(col), pen=pg.mkPen(None))
                self.price.addItem(sc)
                self.lux_items.append(sc)
            for idx, sym, col in ((lx["strong_buy"], "t1", "#00e676"), (lx["strong_sell"], "t", "#ff1744")):
                sc = pg.ScatterPlotItem(x=idx, y=df["low" if sym == "t1" else "high"].values[idx] if len(idx) else [],
                                        symbol=sym, size=14, brush=pg.mkBrush(col), pen=pg.mkPen("#ffffff", width=0.6))
                self.price.addItem(sc)
                self.lux_items.append(sc)
        if show.get("sp2l") and payload.get("sp2l"):
            for (x0, x1, origin, levels, bull, trig) in payload["sp2l"]:
                col = UP if bull else DN
                it = pg.PlotDataItem([x0, x1], [origin, origin], pen=pg.mkPen(col, width=1, style=Qt.PenStyle.DotLine))
                self.price.addItem(it)
                self.sp2l_items.append(it)
                for lvl in levels[-6:]:
                    it = pg.PlotDataItem([x0, x1], [lvl, lvl], pen=pg.mkPen(col, width=0.8, style=Qt.PenStyle.DashLine))
                    self.price.addItem(it)
                    self.sp2l_items.append(it)
                t = pg.TextItem("SP2L", color=col, anchor=(0, 1))
                t.setPos(x0, origin)
                self.price.addItem(t)
                self.sp2l_items.append(t)

    def _draw_positions(self, positions):
        for it in self._pos_items:
            self.price.removeItem(it)
        self._pos_items, self.levels = [], []

        def line(y, color, text, style=Qt.PenStyle.DashLine, width=1.3):
            it = pg.InfiniteLine(pos=y, angle=0, movable=False, pen=pg.mkPen(color, width=width, style=style),
                                 label=text, labelOpts={"position": 0.08, "color": color,
                                                        "fill": (19, 23, 34, 200)})
            self.price.addItem(it, ignoreBounds=True)
            self._pos_items.append(it)
            self.levels.append(y)

        for p in positions:
            pnl = p.upnl()
            line(p.entry, "#d1d4dc", f"{p.side.upper()} {p.remaining:.6g} @ {p.entry:.6g}  PnL {pnl:+.2f}",
                 Qt.PenStyle.SolidLine)
            line(p.sl, DN, f"SL {p.sl:.6g}")
            for k, tp in enumerate(p.tps):
                hit = k < p.tp_hit
                line(tp, "#4d7a6f" if hit else UP, f"TP{k + 1} {tp:.6g}{' ✓' if hit else ''}")
            if p.liq:
                line(p.liq, "#ff9800", f"Liq ~{p.liq:.6g}", Qt.PenStyle.DotLine, 1)


class ChartGrid(QWidget):
    """Grid of resizable charts (drag the splitter handles)."""
    symbolNeeded = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.charts: list[ChartWidget] = []
        self.cache: dict = {}
        self.symbols: list = []
        self.root = None
        self.maximized = None

    def configure(self, count: int, symbols: list):
        symbols = symbols or ["BTC/USDT"]
        old = [c.symbol for c in self.charts]
        if self.root is not None:
            self.root.setParent(None)
            self.root.deleteLater()
        self.symbols = list(symbols)
        self.charts = []
        for i in range(count):
            sym = old[i] if i < len(old) and old[i] in symbols else symbols[i % len(symbols)]
            ch = ChartWidget(symbols, sym)
            ch.symbolChanged.connect(self._on_symbol)
            ch.maximizeRequested.connect(self._toggle_max)
            self.charts.append(ch)
            if sym in self.cache:
                ch.update_data(self.cache[sym])
        self.root = self._arrange(self.charts)
        self.lay.addWidget(self.root)
        self.maximized = None

    def _arrange(self, charts):
        def split(orient, widgets):
            s = QSplitter(orient)
            for w in widgets:
                s.addWidget(w)
            s.setChildrenCollapsible(False)
            return s
        H, V = Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        n = len(charts)
        if n == 1:
            return split(H, charts)
        if n == 2:
            return split(H, charts)
        if n == 3:
            return split(H, [charts[0], split(V, charts[1:])])
        if n == 4:
            return split(V, [split(H, charts[:2]), split(H, charts[2:])])
        return split(V, [split(H, charts[:3]), split(H, charts[3:6])])

    def _toggle_max(self, chart):
        if self.maximized is chart:
            for c in self.charts:
                c.setVisible(True)
            self.maximized = None
        else:
            for c in self.charts:
                c.setVisible(c is chart)
            self.maximized = chart

    def _on_symbol(self, chart, sym):
        if sym in self.cache:
            chart.update_data(self.cache[sym])
        else:
            self.symbolNeeded.emit(sym)

    def update_symbol_tuning(self, info):
        """Tuning finished for a coin: nothing to redraw, the next candle uses the new settings."""
        return None

    def update_symbol(self, sym, payload):
        self.cache[sym] = payload
        for c in self.charts:
            if c.symbol == sym:
                c.update_data(payload)
