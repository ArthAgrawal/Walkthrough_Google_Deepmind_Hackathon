"""Step 2: HYDROGEN transcribes the audio in its ORIGINAL language, with timestamps."""
from __future__ import annotations

from pathlib import Path

from google.genai import types
from pydantic import BaseModel

import models


class Segment(BaseModel):
    """One timestamped stretch of speech."""
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    """Transcript in the original spoken language."""
    language: str          # ISO 639-1 code, e.g. "en", "es", "hi"
    segments: list[Segment]


PROMPT = (
    "Transcribe this audio from a tenant filming their apartment. Write the words exactly as "
    "spoken, in the ORIGINAL language and script. Do NOT translate. Give start/end times in "
    "seconds from the start of the audio for each sentence. Set `language` to the ISO 639-1 "
    "code of the main spoken language. If there is no speech, return an empty segments list."
)


def transcribe(audio_path: str | Path | None) -> dict:
    """Return {"language", "segments": [{"start","end","text"}], "text"}; empty if no audio."""
    if audio_path is None:
        return {"language": "en", "segments": [], "text": ""}
    audio = types.Part.from_bytes(data=Path(audio_path).read_bytes(), mime_type="audio/mp3")
    resp = models.hydrogen(
        [audio, PROMPT],
        types.GenerateContentConfig(response_mime_type="application/json", response_schema=Transcript),
        step="transcribe",
    )
    data = models.parse_json(resp)
    data["text"] = " ".join(s["text"].strip() for s in data.get("segments", []))
    return data


def format_for_prompt(transcript: dict) -> str:
    """Render a transcript as '[12.0s] text' lines for downstream prompts."""
    return "\n".join(f"[{s['start']:.1f}s] {s['text']}" for s in transcript.get("segments", []))
