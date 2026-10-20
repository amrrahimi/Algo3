from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class Signal:
    symbol: str
    timeframe: str
    strategy: str
    side: str                 # "long" | "short"
    entry: float
    sl: float
    tps: list
    confidence: float
    reasons: list = field(default_factory=list)
    ml_prob: float | None = None
    bar_time: float = 0.0     # unix seconds of the signal candle
    created: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.sl)
        return abs(self.tps[-1] - self.entry) / risk if risk > 0 and self.tps else 0.0

    def to_dict(self) -> dict:
        return asdict(self)
