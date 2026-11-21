"""Step 6: local time (America/New_York) and current NYC outdoor temperature (Open-Meteo)."""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

import config

log = logging.getLogger("context")


def outdoor_temp_f() -> float | None:
    """Current NYC outdoor temperature in °F from Open-Meteo, or None if the request fails."""
    try:
        r = requests.get(config.OPEN_METEO_URL, timeout=5, params={
            "latitude": config.NYC_LAT, "longitude": config.NYC_LON,
            "current": "temperature_2m", "temperature_unit": "fahrenheit", "timezone": config.TIMEZONE,
        })
        r.raise_for_status()
        return float(r.json()["current"]["temperature_2m"])
    except Exception as exc:
        log.warning("Open-Meteo failed: %s", exc)
        return None


def in_heat_season(dt: datetime) -> bool:
    """True from Oct 1 through May 31."""
    return dt.month >= 10 or dt.month <= 5


def get_context(local_time: str | None = None, outdoor_f: float | None = None) -> dict:
    """Return {"local_time": "HH:MM", "date", "heat_season", "outdoor_f"}.

    local_time / outdoor_f overrides let the eval replay clips "at night" or "on a warm day".
    """
    now = datetime.now(ZoneInfo(config.TIMEZONE))
    if local_time:
        h, m = (int(x) for x in local_time.split(":"))
        now = now.replace(hour=h, minute=m)
    return {
        "local_time": now.strftime("%H:%M"),
        "date": now.date().isoformat(),
        "heat_season": in_heat_season(now),
        "outdoor_f": outdoor_f if outdoor_f is not None else outdoor_temp_f(),
    }
