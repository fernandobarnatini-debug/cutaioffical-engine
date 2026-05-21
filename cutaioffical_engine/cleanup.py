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
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from .prompts import SCRIPT_SCHEMA, SYSTEM_PROMPT
from .clip import align
from .refine import refine_ranges

log = logging.getLogger(__name__)


DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
# 90s total + 15s connect is the right ceiling for Deepgram nova-3 against
# speech-typical audio. The previous 600s default let a single hung socket
# pin the worker for 10 minutes with no diagnostic trace.
HTTP_TIMEOUT = httpx.Timeout(90.0, connect=15.0)
# Sonnet call ceiling. Sonnet 4.5 normally returns in 10–25s for our prompts;
# 120s leaves enough slack for a slow day at Anthropic without letting a
# stuck request hang the pipeline.
LLM_TIMEOUT_SECONDS = 120.0

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4.5"

PAUSE_MARKER_THRESHOLD = 0.3  # seconds; pauses ≥ this get an explicit marker for the AI.
# 300ms is the physical boundary between natural inter-word gaps (30–150ms)
# and intentional breath / sentence-break pauses (200ms+). Below 300ms is
# normal speech rhythm; at/above 300ms is a deliberate stop. Lowered from
# 0.5s once Sonnet was taught to split kept_spans at pause markers — see
# SPAN BREAKS AT PAUSES in prompts.py.


def extract_audio(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(wav_path),
    ]
    t0 = time.time()
    try:
        # 90s ceiling — audio extraction is fast even on long sources; the
        # timeout exists purely to keep a hung ffmpeg from holding the worker.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg audio extract timed out after {exc.timeout}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr.strip()}")
    log.info("extract_audio done in %.1fs", time.time() - t0)


def call_deepgram(wav_bytes: bytes, api_key: str) -> dict:
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}
    params = {
        "model": "nova-3",
        "punctuate": "true",
        "smart_format": "false",
        "numerals": "false",
        "filler_words": "true",
    }
    t0 = time.time()
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        r = client.post(DEEPGRAM_URL, headers=headers, params=params, content=wav_bytes)
    if r.status_code >= 400:
        raise RuntimeError(f"Deepgram HTTP {r.status_code}: {r.text[:500]}")
    log.info("deepgram done in %.1fs", time.time() - t0)
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
    t0 = time.time()
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
        # Hard ceiling on the LLM call — Sonnet defaults to ~10min via the
        # OpenAI SDK which is far too long to wait on a stuck request.
        timeout=LLM_TIMEOUT_SECONDS,
    )
    log.info("sonnet done in %.1fs", time.time() - t0)
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
    """Full pipeline: cleanup + clip alignment + Block-3 word-boundary refinement.

    Internal-silence splitting is disabled to match the CLI behavior signed off
    on 2026-05-21 with `--keep-internal-silence`. Natural breaths inside a
    kept_span are preserved as audible time in the final cut rather than being
    sliced out. Cross-span silence still drops via the span-break + concat
    mechanism in the prompt and render; only WITHIN-clip pauses are kept.

    To restore aggressive internal-silence splitting (original behavior),
    call refine_ranges() directly with default args.
    """
    video_path = Path(video_path)
    result = run_cleanup(video_path)
    t0 = time.time()
    result["clip"] = align(result["script"], result["deepgram"], video_name=video_path.stem)
    log.info("align done in %.1fs", time.time() - t0)

    # Block 3: wav2vec2 + CTC forced alignment snaps each range's edges to
    # sub-50ms-accurate word boundaries. We pass split_internal_silence=False
    # so internal pauses (natural breaths, mid-sentence beats) are PRESERVED
    # in the rendered cut — this matches the locally-tested CLI behavior.
    t0 = time.time()
    dg_words = result["deepgram"]["results"]["channels"][0]["alternatives"][0]["words"]
    result["clip"] = refine_ranges(
        result["clip"], video_path, dg_words,
        split_internal_silence=False,
    )
    log.info("refine done in %.1fs", time.time() - t0)
    return result
