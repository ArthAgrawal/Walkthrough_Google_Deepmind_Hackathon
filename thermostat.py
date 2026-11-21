"""Step 5: if a usable frame shows a thermostat (P >= 0.7), NEON reads the number on the best frame."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from google.genai import types
from pydantic import BaseModel

import config
import models


class ThermostatReading(BaseModel):
    """What the thermostat display shows."""
    legible: bool
    current_temp: Optional[float]      # the room temperature reading, not the setpoint
    setpoint: Optional[float]
    unit: Literal["F", "C", "unknown"]


PROMPT = (
    "This photo shows a thermostat or room temperature display in an apartment. Read the CURRENT "
    "ROOM TEMPERATURE it shows (not the target/set temperature, if both are displayed). "
    "If you can't read the digits clearly, set legible=false and current_temp=null. Do not guess."
)


def best_thermostat_frame(frame_checks: list[dict]) -> dict | None:
    """Return the usable FrameCheck with the highest P(thermostat_visible) if >= threshold, else None."""
    cands = [f for f in frame_checks if f["usable"] and f["p"].get("thermostat_visible", 0) >= config.THERMOSTAT_MIN_P]
    return max(cands, key=lambda f: f["p"]["thermostat_visible"]) if cands else None


def read_thermostat(frame_checks: list[dict], work_dir: Path) -> dict | None:
    """Read the thermostat on the best frame. Returns {"t","frame_path","p_visible","indoor_f",...} or None."""
    best = best_thermostat_frame(frame_checks)
    if best is None:
        return None
    img = types.Part.from_bytes(data=(Path(work_dir) / best["frame_path"]).read_bytes(), mime_type="image/jpeg")
    resp = models.neon(
        [img, PROMPT],
        types.GenerateContentConfig(response_mime_type="application/json", response_schema=ThermostatReading),
        step="thermostat",
    )
    r = models.parse_json(resp)
    temp, unit = r.get("current_temp"), r.get("unit")
    if temp is not None and (unit == "C" or (unit == "unknown" and temp < 40)):
        temp = temp * 9 / 5 + 32          # a room reading under 40 is almost surely Celsius
    indoor_f = round(temp, 1) if (r.get("legible") and temp is not None) else None
    return {"t": best["t"], "frame_path": best["frame_path"], "p_visible": best["p"]["thermostat_visible"],
            "indoor_f": indoor_f, "raw": r}
