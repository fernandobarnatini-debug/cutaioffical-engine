"""Block 3 — precise word-boundary refinement via wav2vec2 + CTC forced alignment.

Takes the output of align() (clip_json with imprecise Deepgram word timestamps
on each range) and re-snaps every range's start/end to sub-50ms-accurate
positions by forced-aligning the source audio against the Deepgram word list.

The forced-alignment is over the FULL Deepgram word sequence (one pass per
clip). Each range then looks up its head/tail words via start_word_idx /
end_word_idx (produced by clip.make_range) and replaces start/end with the
refined values.

Tuning-free: wav2vec2-base-960h is a fixed model. No thresholds, no per-video
parameters. Boundaries are determined by the model's phoneme onsets/offsets.

Public API:
    refine_ranges(clip_json, audio_path, dg_words) -> dict

Pipeline integration: called in cleanup.run_pipeline after align(). See
prompts.py (Block 1 / pause-aware span breaks) for the editorial side that
runs before this.
"""
from __future__ import annotations

import copy
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taF

log = logging.getLogger(__name__)

# wav2vec2-base-960h: 50Hz output, char-level CTC over uppercase A-Z + '
# The model is ~360MB on first download; cached by HuggingFace under
# ~/.cache/huggingface afterward. On Fly, the cache persists for the lifetime
# of a machine; first job after a deploy pays the download cost once.
_MODEL_ID = "facebook/wav2vec2-base-960h"
_SAMPLE_RATE = 16000

# Lazy-loaded singletons. Loading at import time would pull ~360MB just to
# import the module — bad for cold-start latency on the worker.
_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        _processor = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
        _model = Wav2Vec2ForCTC.from_pretrained(_MODEL_ID)
        _model.eval()
    return _model, _processor


def _normalize_token(w: str) -> str:
    """Match the engine's token normalization: uppercase, keep A-Z and apostrophes only.

    Identical to clip.py's normalize_token semantics — wav2vec2's vocab is
    uppercase A-Z plus apostrophe, so any other character must be stripped
    before alignment.
    """
    return re.sub(r"[^A-Z']", "", w.upper())


def _extract_audio(video_path: Path) -> np.ndarray:
    """ffmpeg-decode the video to mono 16kHz float32 PCM.

    Mirrors the audio-extraction style used by cleanup.extract_audio but
    returns the samples in-memory instead of writing a wav file the caller
    has to manage.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-ac", "1",
                "-ar", str(_SAMPLE_RATE),
                "-acodec", "pcm_s16le",
                out,
            ],
            check=True,
            timeout=90,
        )
        audio, sr = sf.read(out, dtype="float32")
        if sr != _SAMPLE_RATE:
            raise RuntimeError(f"unexpected sample rate after ffmpeg: {sr}")
        return audio
    finally:
        Path(out).unlink(missing_ok=True)


def _align_words(audio: np.ndarray, words: list[str]) -> list[tuple[float, float] | None]:
    """Forced-align a sequence of already-normalized words against audio.

    Returns one (start_s, end_s) tuple per input word, or None for words whose
    normalized form contained no in-vocab characters.
    """
    model, processor = _load_model()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id
    delim_id = vocab[processor.tokenizer.word_delimiter_token]

    target_ids: list[int] = []
    # (first_token_idx, last_token_idx) into target_ids, inclusive; (-1, -1) for
    # words whose normalized form contains no in-vocab characters.
    word_token_spans: list[tuple[int, int]] = []
    for i, w in enumerate(words):
        if i > 0 and target_ids:
            target_ids.append(delim_id)
        start = len(target_ids)
        for ch in w:
            tid = vocab.get(ch)
            if tid is not None:
                target_ids.append(tid)
        end = len(target_ids) - 1
        if end < start:
            word_token_spans.append((-1, -1))
            # Roll back the delimiter we just added so we don't end up with two
            # in a row when the empty word sits between two non-empty ones.
            if target_ids and target_ids[-1] == delim_id:
                target_ids.pop()
        else:
            word_token_spans.append((start, end))

    if not target_ids:
        return [None] * len(words)

    input_values = processor(
        audio, sampling_rate=_SAMPLE_RATE, return_tensors="pt"
    ).input_values
    with torch.inference_mode():
        logits = model(input_values).logits  # (1, T, V)
        emission = torch.log_softmax(logits, dim=-1)

    targets = torch.tensor([target_ids], dtype=torch.int32)
    aligned, scores = taF.forced_align(emission, targets, blank=blank_id)
    aligned = aligned[0]
    scores = scores[0].exp()
    token_spans = taF.merge_tokens(aligned, scores)

    if len(token_spans) != len(target_ids):
        raise RuntimeError(
            f"alignment span mismatch: got {len(token_spans)} spans for {len(target_ids)} tokens"
        )

    num_samples = audio.shape[0]
    num_frames = emission.shape[1]
    samples_per_frame = num_samples / num_frames

    out: list[tuple[float, float] | None] = []
    for span in word_token_spans:
        a, b = span
        if a < 0:
            out.append(None)
            continue
        s_frame = token_spans[a].start
        e_frame = token_spans[b].end  # exclusive frame
        s = s_frame * samples_per_frame / _SAMPLE_RATE
        e = e_frame * samples_per_frame / _SAMPLE_RATE
        out.append((float(s), float(e)))
    return out


def refine_ranges(clip_json: dict, audio_path: str | Path, dg_words: list[dict]) -> dict:
    """Snap each range in clip_json.segments[].ranges to exact word boundaries.

    Strategy: forced-align the full Deepgram word list against the source audio
    once, then for each range look up the refined start/end of its head and
    tail Deepgram words via start_word_idx / end_word_idx (which clip.make_range
    already produces). The original clip_json is not mutated — a deep copy is
    returned.

    If alignment fails for the head or tail word (e.g. normalized form had no
    in-vocab characters), the original imprecise timestamps for that range are
    preserved rather than corrupted with None.
    """
    audio = _extract_audio(Path(audio_path))
    norm_words = [_normalize_token(w.get("word", "")) for w in dg_words]
    timings = _align_words(audio, norm_words)

    out = copy.deepcopy(clip_json)
    for seg in out.get("segments", []):
        for rng in seg.get("ranges", []):
            i0 = rng.get("start_word_idx")
            i1 = rng.get("end_word_idx")
            if i0 is None or i1 is None:
                # Defensive: a range without word indices can't be refined.
                # Older clip_json shapes may lack these — leave untouched.
                continue
            if i0 < 0 or i1 >= len(timings):
                continue
            head = timings[i0]
            tail = timings[i1]
            if head is None or tail is None:
                continue
            rng["start"] = round(float(head[0]), 3)
            rng["end"] = round(float(tail[1]), 3)
    return out
