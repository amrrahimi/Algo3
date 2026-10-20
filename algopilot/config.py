"""Settings, paths and constants."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field

APP_NAME = "AlgoPilot"


def app_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def resource_path(rel: str) -> str:
    """Path to bundled resources (works in dev and in PyInstaller builds)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, rel)


MODES = {
    "scalp": {"label": "Scalp", "timeframes": ["1m", "3m", "5m"], "default": "5m"},
    "daily": {"label": "Daily (Day trading)", "timeframes": ["15m", "30m", "1h"], "default": "15m"},
    "long": {"label": "Long (Swing / Position)", "timeframes": ["4h", "1d"], "default": "4h"},
}

TF_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
              "4h": 14400, "1d": 86400}

EXCHANGES = ["binance", "bybit", "kucoin", "toobit"]
DATA_SOURCES = EXCHANGES + ["demo"]  # "demo" = synthetic offline data

DEFAULT_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT",
                 "ADA/USDT", "AVAX/USDT", "LINK/USDT", "TON/USDT", "SUI/USDT", "LTC/USDT",
                 "DOT/USDT", "TRX/USDT", "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT"]


# ---------------------------------------------------------------- secret storage
KEYRING_SERVICE = "AlgoPilot"
SECRET_FIELDS = ("secret", "password")


def _keyring():
    try:
        import keyring  # Windows Credential Manager / macOS Keychain
        keyring.get_keyring()
        return keyring
    except Exception:  # noqa: BLE001
        return None


def _secret_get(name: str) -> str | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(KEYRING_SERVICE, name)
    except Exception:  # noqa: BLE001
        return None


def _secret_set(name: str, value: str) -> bool:
    kr = _keyring()
    if kr is None:
        return False
    try:
        if value:
            kr.set_password(KEYRING_SERVICE, name, value)
        else:
            try:
                kr.delete_password(KEYRING_SERVICE, name)
            except Exception:  # noqa: BLE001
                pass
        return True
    except Exception:  # noqa: BLE001
        return False


def _coerce(default, value):
    """Cast a loaded JSON value to the type of the default; None if impossible."""
    try:
        if isinstance(default, bool):
            return value if isinstance(value, bool) else None
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return value if isinstance(value, str) else None
        if isinstance(default, list):
            return [x for x in value if isinstance(x, str)] if isinstance(value, list) else None
    except (TypeError, ValueError):
        return None
    return value


SETTINGS_VERSION = 3


@dataclass
class ExchangeCreds:
    exchange: str = "binance"
    api_key: str = ""
    secret: str = ""
    password: str = ""        # KuCoin passphrase
    testnet: bool = False


@dataclass
class Settings:
    mode: str = "daily"
    timeframe: str = "15m"
    market_type: str = "futures"          # spot | futures
    leverage: int = 5
    symbols: list = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                                                   "XRP/USDT", "DOGE/USDT"])
    strategy: str = "auto_pilot"
    trade_mode: str = "notify"            # notify (semi-auto) | auto
    account: str = "paper"                # paper | live
    data_exchange: str = "binance"
    risk_pct: float = 1.0                 # % of balance risked per trade
    max_positions: int = 5
    max_daily_loss_pct: float = 5.0
    min_confidence: float = 0.55
    use_ml: bool = True
    paper_start_balance: float = 10000.0
    poll_seconds: int = 10
    sound: bool = True
    chart_layout: int = 2
    exchange_side_sl: bool = True
    auto_tune: bool = True                # tune indicator settings on each coin's own chart
    signal_frequency: str = "balanced"    # quality | balanced | high (more trades)
    show_smc: bool = True                 # chart overlays
    show_lux: bool = True
    show_sp2l: bool = True
    show_emas: bool = True
    osc_pane: str = "squeeze"             # squeeze | rsi
    settings_version: int = 0
    creds: ExchangeCreds = field(default_factory=ExchangeCreds)

    # ---------- persistence ----------
    @staticmethod
    def path() -> str:
        return os.path.join(app_dir(), "settings.json")

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        try:
            with open(cls.path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            creds = data.pop("creds", {}) or {}
            for k, v in data.items():
                if hasattr(s, k) and k != "creds":
                    v = _coerce(getattr(s, k), v)
                    if v is not None:
                        setattr(s, k, v)
            if isinstance(creds, dict):
                for k, v in creds.items():
                    if hasattr(s.creds, k):
                        v = _coerce(getattr(s.creds, k), v)
                        if v is not None:
                            setattr(s.creds, k, v)
        except (OSError, ValueError, AttributeError):
            pass
        for f in SECRET_FIELDS:              # secrets live in the OS keychain when available
            if not getattr(s.creds, f):
                v = _secret_get(f"{s.creds.exchange}:{f}")
                if v:
                    setattr(s.creds, f, v)
        s.leverage = min(max(1, s.leverage), 125)
        s.risk_pct = min(max(0.01, s.risk_pct), 10.0)
        s.poll_seconds = max(3, s.poll_seconds)
        s.max_positions = max(1, s.max_positions)
        if s.market_type not in ("spot", "futures"):
            s.market_type = "futures"
        if s.account not in ("paper", "live"):
            s.account = "paper"
        if s.trade_mode not in ("notify", "auto"):
            s.trade_mode = "notify"
        if s.mode not in MODES:
            s.mode = "daily"
        if s.timeframe not in MODES[s.mode]["timeframes"]:
            s.timeframe = MODES[s.mode]["default"]
        if s.signal_frequency not in ("quality", "balanced", "high"):
            s.signal_frequency = "balanced"
        if s.osc_pane not in ("squeeze", "rsi"):
            s.osc_pane = "squeeze"
        if s.settings_version < SETTINGS_VERSION:
            # v3: new default strategy (Auto Pilot) and a slightly lower confidence gate so the
            # robot finds more trades. Only applied once; later user choices are kept.
            if s.strategy == "ai_confluence":
                s.strategy = "auto_pilot"
            if abs(s.min_confidence - 0.60) < 1e-9:
                s.min_confidence = 0.55
            if s.max_positions == 3:
                s.max_positions = 5
            s.settings_version = SETTINGS_VERSION
        return s

    def save(self) -> None:
        p = self.path()
        tmp = p + ".tmp"
        data = asdict(self)
        for fld in SECRET_FIELDS:
            if _secret_set(f"{self.creds.exchange}:{fld}", getattr(self.creds, fld)):
                data["creds"][fld] = ""          # stored in the OS keychain instead of the file
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, p)
        try:
            os.chmod(p, 0o600)   # keep API keys readable only by this user
        except OSError:
            pass

    def copy(self) -> "Settings":
        d = asdict(self)
        creds = ExchangeCreds(**d.pop("creds"))
        return Settings(**d, creds=creds)
