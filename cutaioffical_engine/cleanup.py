"""CleanUp pipeline as a pure library.

Ported from CleanUp/transcribe.py with CLI / print / file-I/O removed. Behavior
of the AI step is identical: same Deepgram params, same OpenRouter model, same
SYSTEM_PROMPT and SCRIPT_SCHEMA. Do not change those — see prompts.py.

Public functions:
  run_pipeline(video_path)           -> dict with deepgram, transcript, annotated,
                                        script, script_text, clip
  run_cleanup(video_path)            -> same minus 'clip'
  run_cleanup_from_deepgram(dg, *, client=None, structure_fn=None)
                                     -> same minus 'clip' and using a cached dg
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from .prompts import SCRIPT_SCHEMA, SYSTEM_PROMPT
from .clip import align


DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
HTTP_TIMEOUT = httpx.Timeout(600.0, connect=30.0)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4.5"

PAUSE_MARKER_THRESHOLD = 0.5  # seconds; pauses ≥ this get an explicit marker for the AI


def extract_audio(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr.strip()}")


def call_deepgram(wav_bytes: bytes, api_key: str) -> dict:
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}
    params = {
        "model": "nova-3",
        "punctuate": "true",
        "smart_format": "false",
        "numerals": "false",
        "filler_words": "true",
    }
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        r = client.post(DEEPGRAM_URL, headers=headers, params=params, content=wav_bytes)
    if r.status_code >= 400:
        raise RuntimeError(f"Deepgram HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def annotated_transcript(dg_result: dict) -> str:
    """Plain transcript text with ⟨pause N.Ns⟩ markers inserted at silence gaps."""
    try:
        words = dg_result["results"]["channels"][0]["alternatives"][0]["words"]
    except (KeyError, IndexError, TypeError):
        words = []
    if not words:
        try:
            return dg_result["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (KeyError, IndexError, TypeError):
            return ""

    out = []
    prev_end = None
    for w in words:
        word_text = w.get("punctuated_word") or w.get("word") or ""
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start))
        if prev_end is not None:
            gap = start - prev_end
            if gap >= PAUSE_MARKER_THRESHOLD:
                out.append(f"⟨pause {gap:.1f}s⟩")
        out.append(word_text)
        prev_end = end
    return " ".join(out)


def structure_script(transcript: str, client: OpenAI) -> dict:
    user_msg = (
        "Raw transcript follows. Edit per the system rules and return the JSON object.\n\n"
        f"---\n{transcript}\n---"
    )
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=8000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "clean_script",
                "strict": True,
                "schema": SCRIPT_SCHEMA,
            },
        },
        extra_headers={
            "HTTP-Referer": "https://github.com/local/cleanup",
            "X-Title": "CleanUp",
        },
    )
    return _parse_json(response.choices[0].message.content)


def _parse_json(content: str) -> dict:
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(s[start:end + 1])


def _raw_transcript(dg_result: dict) -> str:
    try:
        return dg_result["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError, TypeError):
        return ""


def _make_openai_client() -> OpenAI:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set in env/.env")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def _deepgram_from_video(video_path: Path) -> dict:
    load_dotenv()
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY not set in env/.env")
    with tempfile.TemporaryDirectory(prefix="cutaioffical-") as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        extract_audio(video_path, wav_path)
        return call_deepgram(wav_path.read_bytes(), key)


def run_cleanup_from_deepgram(
    deepgram: dict,
    *,
    client: Optional[OpenAI] = None,
    structure_fn: Optional[Callable[[str, OpenAI], dict]] = None,
) -> dict:
    """AI cleanup against an already-fetched Deepgram response.

    `client` and `structure_fn` are injection seams for tests that want to
    avoid network calls. In production both are None and the function uses
    OpenRouter + the locked prompt.
    """
    transcript = _raw_transcript(deepgram)
    if not transcript.strip():
        raise RuntimeError("empty transcript")
    annotated = annotated_transcript(deepgram)

    fn = structure_fn or structure_script
    used_client = client
    if structure_fn is None and used_client is None:
        used_client = _make_openai_client()
    script = fn(annotated, used_client)
    script_text = " ".join(script.get("kept_spans", []))
    return {
        "deepgram": deepgram,
        "transcript": transcript,
        "annotated": annotated,
        "script": script,
        "script_text": script_text,
    }


def run_cleanup(video_path: str | Path) -> dict:
    """ffmpeg -> Deepgram -> AI cleanup. Returns the cleanup dict (no 'clip')."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    dg = _deepgram_from_video(video_path)
    return run_cleanup_from_deepgram(dg)


def run_pipeline(video_path: str | Path) -> dict:
    """Full pipeline: cleanup + clip alignment."""
    video_path = Path(video_path)
    result = run_cleanup(video_path)
    result["clip"] = align(result["script"], result["deepgram"], video_name=video_path.stem)
    return result
