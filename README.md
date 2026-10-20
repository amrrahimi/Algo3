# AlgoPilot

Crypto auto-trading robot for Windows and macOS.

## Features
- Trading styles: Scalp (1m/3m/5m), Daily (15m/30m/1h), Long (4h/1d)
- Spot or futures (long + short, leverage), multiple coins at once
- 9 strategies incl. AI Confluence (11 indicators + ML filter), UT Bot, Squeeze, Supertrend, WaveTrend, Ichimoku
- Strategy ranking by backtest on your coins (fees + slippage included)
- Auto SL below/above swing levels, TP1-TP3 at real S/R levels, breakeven + trailing
- TradingView-style resizable charts (1-6 at once), signals and positions drawn on chart
- Semi-auto (notify, you click Open) or full auto (robot trades 24/7)
- Sound + tray notifications
- Demo account with virtual money on live prices
- Live trading: Binance, Bybit, KuCoin, Toobit via API keys
- History tab with CSV export

## Run from source
    pip install -r requirements.txt
    python main.py

## Testing on a real exchange with a TEST account (fake money)
1. **Binance**: log in at https://demo.binance.com, open API Management and create a key
   (Binance retired the old futures testnet; ccxt uses "Demo Trading"). Needs ccxt >= 4.5.
   **Bybit**: create an account + API key at https://testnet.bybit.com and claim test USDT.
2. In AlgoPilot: 🔑 Exchange API, pick the exchange, paste key + secret, tick
   **"Test account"**, press **Test connection** (it must say TESTNET / demo), Save.
3. Robot setup: Account = Live, start with trade mode = notify (semi-auto), low leverage.
   The account label in the toolbar shows **TESTNET**. Charts and signals then use the
   test venue's prices (they can differ from mainnet).
4. Mainnet keys do NOT work with the box ticked, and test keys do NOT work without it.

## Build
Push a tag like `v1.0.0`, or run the "Build AlgoPilot" workflow manually. Download
AlgoPilot-windows.zip / AlgoPilot-macos.zip from the run artifacts.

## Safety
- Trading involves real risk. Start on the Demo account.
- Use exchange API keys with trading permission only. Never enable withdrawals.
- Daily loss limit pauses trading for the rest of the UTC day.
- Futures: isolated margin, and an exchange-side stop-loss is placed as a safety net
  (the new stop is placed before the old one is cancelled).
- Positions are reconciled with the exchange every tick; trades are never booked from a failed order.
- API secret / passphrase are stored in the OS keychain (Windows Credential Manager / macOS Keychain).
- Past backtest results do not guarantee future profit.
