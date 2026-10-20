# v3.0 - more signals, indicator auto-tuning, demo fixes

## New indicators and strategies
- Smart Money Concepts [LuxAlgo] port (BOS/CHoCH, order blocks, FVG, equal highs/lows, premium/discount). Strategy "Smart Money Concepts"; overlays on the chart.
- SP2L (Spike + 2 Legs, Poursamadi): spike detection with FVG/doji filters, entry on HL/LH retest, stop beyond spike origin. Strategy "SP2L"; overlays on the chart.
- Signals & Overlays (LuxAlgo-style, clean-room re-implementation; the original is closed-source): confirmation + strong signals, Smart Trail, Trend Catcher, Trend Tracer, Neo Cloud, Reversal Zones.
- Squeeze Momentum [LazyBear]: 1:1 port of the TradingView script (histogram colours + dot states), shown as the bottom pane.
- Auto Pilot (new default): runs every strategy of the chosen style at once; several agreeing strategies raise confidence.

## Per-coin auto-tuning
- Each indicator's settings (lengths, multipliers, sensitivity, spike rules...) are tuned on that coin's own recent chart for the chosen timeframe and style.
- Walk-forward check: tuned on the older 65%, validated on the newest 35%. Failed strategies are muted in Auto Pilot.
- Tuning cache is stored; re-tune with the "Re-tune indicators now" button.

## Finding more positions
- Signal frequency: Quality / Balanced / More trades (shifts the confidence gate).
- Defaults: Auto Pilot strategy, 55% min confidence, 6 coins, 5 max open trades.
- Signals are still checked on candle close, but a signal from the last 1-3 candles is taken if price is still near its entry.
- Settings migrate once (v3) so existing installs get the new defaults without overwriting later choices.

## Fixes
- Robot logs why each signal is filtered or skipped, plus a periodic heartbeat (coins, tuning, counts).
- Strategy stops (order block, spike origin, Smart Trail) used when sane, clamped to 0.5-5 ATR.
- Demo open/close flow covered by a regression test.
- LinReg / WMA rewritten vectorized (much faster); indicator causality test added.

## Known limits
- Overlays and signals are re-implementations of the described logic, not bit-exact copies of the TradingView scripts.
- Backtests on synthetic demo data don't prove profitability; paper-trade before live money.
# v2.1 bug-fix release

## Live trading (critical)
- Failed close orders no longer mark the position closed; state is unchanged and the close is retried.
- Positions are reconciled with the exchange every tick (exchange stop hit, manual close, liquidation,
  partial reduce). Real stop fill price is used; no more fake TP profits after an exchange stop.
- Open positions are loaded per exchange (switching Binance -> Bybit no longer mixes positions).
- Live PnL is net of entry and exit fees (fees in base coin converted; spot qty reduced by coin fee).
- Isolated margin is requested; a trade is refused if the leverage can't be set.
- Exchange-side stop: new stop placed before the old is cancelled; stop cancelled on full close,
  restored if the close order fails.
- Partial TPs below the exchange minimum are skipped instead of being booked as filled.
- Testnet: Binance uses Demo Trading (enable_demo_trading), Bybit uses testnet; the live/testnet
  account uses its own venue's prices for charts, signals and SL/TP.

## Logic
- ML model now retrains every 25 new bars (counted by time; window length is fixed).
- ML holdout is purged by the label horizon and the edge uses a lower confidence bound.
- Demo data: price for manual open/close comes from the same series as the chart.
- Paper account: wicks between polls trigger SL/TP (stop first, gap fills at price).
- If candles fail to load, open positions are still checked with the ticker price.
- OHLCV pagination respects each exchange's per-request limit (KuCoin futures 200).
- AI Confluence strength rescaled so valid triggers aren't silently filtered at 60%.
- SL reason reports when the swing stop was clamped to the ATR range.
- Position size includes fees and is capped by free margin; risk is based on equity (futures).
- Daily loss limit uses start-of-day equity and includes partial exits of open trades.

## Backtest
- Gaps through the stop fill at the bar open; notional cap applied; intra-trade drawdown counted;
  min-confidence and leverage applied like the robot; time exits no longer double counted.

## App / security
- API secret/passphrase stored in the OS keychain (keyring), settings file written with 0600.
- Settings values are type-checked and clamped on load.
- Switching to a live account connects in the background (no UI freeze); on failure the old account is kept.
- Manual open and auto-trade are serialized (max positions can't be exceeded); signal memory is bounded.
- New regression tests: tests/test_fixes.py (fake exchange, no network).
