"""Shared dataset cadence and forecast-range definitions."""

from __future__ import annotations


RANGE_NAMES = ("short", "mid", "long")
RANGE_SETTINGS = {
    "hourly": {
        "short": (168, 24),
        "mid": (336, 48),
        "long": (504, 168),
    },
    "daily": {
        "short": (7, 1),
        "mid": (14, 2),
        "long": (30, 7),
    },
    "15min": {
        "short": (96, 4),
        "mid": (192, 8),
        "long": (672, 96),
    },
}
LOOKBACK_PERIOD_BY_FREQUENCY = {"hourly": 168, "daily": 7, "15min": 672}
DATASET_FREQUENCIES = {
    "electricity": "hourly",
    "traffic": "hourly",
    "solar": "hourly",
    "weather": "hourly",
    "etth1": "hourly",
    "etth2": "hourly",
    "ettm1": "15min",
    "ettm2": "15min",
    "exchange_rate": "daily",
}


def normalize_frequency(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "")
    aliases = {
        "h": "hourly",
        "1h": "hourly",
        "hour": "hourly",
        "hourly": "hourly",
        "d": "daily",
        "1d": "daily",
        "day": "daily",
        "daily": "daily",
        "15t": "15min",
        "15m": "15min",
        "15min": "15min",
        "15minute": "15min",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported dataset frequency {value!r}") from error


def forecast_range_name(lookback: int, horizon: int) -> str:
    """Return the cadence-independent range label for a standard L-H pair."""
    setting = (int(lookback), int(horizon))
    for ranges in RANGE_SETTINGS.values():
        for name, candidate in ranges.items():
            if candidate == setting:
                return name
    return "custom"
