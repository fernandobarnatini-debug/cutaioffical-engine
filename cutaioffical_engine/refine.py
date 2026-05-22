"""Block 3 — precise word-boundary refinement + within-range deadspace removal.

Two passes per clip:

1) wav2vec2 + CTC forced alignment over the full Deepgram word sequence. Gives
   us sub-50ms-accurate per-word (start, end) timings.
2) For each kept range, walk consecutive word pairs and split the range at any
   gap ≥ INTERNAL_SILENCE_SPLIT_S between wav2vec2-precise word boundaries.
   Each sub-range is then assigned wav2vec2 boundaries on both edges.

The split uses wav2vec2 word gaps (NOT Deepgram word gaps, NOT ffmpeg
silencedetect):

- Deepgram tokenizes consecutive words with end_of_prev = start_of_next even
  when there's real acoustic silence between, so word-gap-based splitting on
  Deepgram boundaries fires almost never (we measured this).
- ffmpeg silencedetect at speech thresholds catches silence during the
  natural decay of word endings, which then gets misattributed as "before the
  word's end" because Deepgram's reported word end is loose (±50-100ms). The
  splitter ends up cutting INTO the last word of each sub-range.
- wav2vec2's per-word boundaries are precise, AND the gap between two
  consecutive word boundaries IS the actual inter-word silence. Splitting
  here is safe — the split point is the same boundary used to mark the
  word's end, so we can't accidentally cut into it.

Tuning-free where it matters: wav2vec2-base-960h is a fixed model. The
0.2s threshold is the lower bound of noticeable speech pause.

Public API:
    refine_ranges(clip_json, audio_path, dg_words) -> dict

Pipeline integration: called in cleanup.run_pipeline after align(). See
prompts.py (Block 1 / pause-aware span breaks) for the editorial side.
"""
from __future__ import annotations

import copy
import logging
import os
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

# PyTorch defaults to 1-2 CPU threads when OMP_NUM_THREADS isn't set, which
# leaves 6+ of the 8 vCPUs idle on Fly's shared-cpu-8x machines. Setting this
# explicitly at import time lets wav2vec2 inference use the whole machine —
# measured impact: ~58s refine drops to ~15-25s on the same hardware.
# Safe default of 8; if a smaller machine is ever used, torch will accept any
# value ≤ the actual CPU count without error.
torch.set_num_threads(8)

# Internal-silence split threshold — wav2vec2-measured inter-word gap above
# which we split a kept range. 200ms is the lower bound of noticeable speech
# pause; below this is natural inter-word flow. Same threshold semantics as
# Block 1's pause-aware span breaks, but measured against wav2vec2 precise
# boundaries instead of the annotated transcript's pause markers.
INTERNAL_SILENCE_SPLIT_S = 0.2

# Lazy-loaded singletons. Loading at import time would pull ~360MB just to
# import the module — bad for cold-start latency on the worker.
_model = None
_processor = None
_ort_session = None

# Default-on env gate for the ONNX INT8 wav2vec2 path. Set
# CUTAIOFFICAL_USE_ONNX_WAV2VEC2=0 to revert every worker to the slow PyTorch
# path without a redeploy (kill switch for production rollback).
_USE_ONNX = os.getenv("CUTAIOFFICAL_USE_ONNX_WAV2VEC2", "1") != "0"

# INT8 ONNX file lives in the user's cache directory (~/.cache by default,
# overridable via CUTAIOFFICAL_CACHE_DIR for containerized environments where
# $HOME may not be writable). Lazy-downloaded from a GitHub release asset on
# first use — the file is ~117MB, which exceeds GitHub's 100MB blob limit, so
# we can't bundle it inside the wheel. The download is one-time per worker
# lifetime; subsequent jobs hit the cached file directly. Mirrors how
# HuggingFace fetches model weights to ~/.cache/huggingface on first use.
_ONNX_VERSION = "v0.2.0-wav2vec2-onnx"
_ONNX_FILENAME = "wav2vec2_base_960h.int8.onnx"
_ONNX_EXPECTED_SHA256 = (
    "0000000000000000000000000000000000000000000000000000000000000000"  # filled at release time
)
_ONNX_URL = (
    f"https://github.com/fernandobarnatini-debug/cutaioffical-engine/"
    f"releases/download/{_ONNX_VERSION}/{_ONNX_FILENAME}"
)
_CACHE_ROOT = Path(
    os.getenv("CUTAIOFFICAL_CACHE_DIR")
    or (Path.home() / ".cache" / "cutaioffical_engine")
)
_ONNX_PATH = _CACHE_ROOT / _ONNX_FILENAME


def _load_model():
    """Load the PyTorch wav2vec2 model + processor.

    Always called: the processor handles audio normalization and the
    tokenizer/vocab, which both paths need. The model itself is only used as
    a fallback when the ONNX path is disabled or fails.
    """
    global _model, _processor
    if _model is None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        _processor = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
        _model = Wav2Vec2ForCTC.from_pretrained(_MODEL_ID)
        _model.eval()
    return _model, _processor


def _load_processor_only():
    """Load only the processor (tokenizer + audio normalizer).

    Used by the ONNX path so we don't pay the ~360MB PyTorch model load on
    every worker boot when the model itself never runs.
    """
    global _processor
    if _processor is None:
        from transformers import Wav2Vec2Processor

        _processor = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
    return _processor


def _ensure_onnx_file() -> bool:
    """Make sure the INT8 ONNX file is present on disk; download if not.

    Returns True if the file is ready, False if the download failed (in which
    case the caller falls back to the PyTorch path).

    Atomic write: download to a .partial path and rename on success. Avoids
    leaving a half-written file that ORT would later refuse to load.
    """
    if _ONNX_PATH.exists() and _ONNX_PATH.stat().st_size > 1024 * 1024:
        return True
    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = _ONNX_PATH.with_suffix(".partial")
        log.info("refine: downloading ONNX INT8 model from %s", _ONNX_URL)
        import urllib.request

        with urllib.request.urlopen(_ONNX_URL, timeout=300) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(_ONNX_PATH)
        log.info(
            "refine: cached ONNX model at %s (%.1f MB)",
            _ONNX_PATH, _ONNX_PATH.stat().st_size / 1024 / 1024,
        )
        return True
    except Exception as e:
        log.warning("refine: ONNX model download failed (%s); will use PyTorch path", e)
        return False


def _load_onnx_session():
    """Load the ONNX Runtime session (lazy, singleton).

    intra_op_num_threads=8 matches torch.set_num_threads(8) above and Fly's
    shared-cpu-8x configuration. If the worker ever runs on a smaller box,
    ORT silently caps at the actual core count.
    """
    global _ort_session
    if _ort_session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        _ort_session = ort.InferenceSession(
            _ONNX_PATH.as_posix(), opts, providers=["CPUExecutionProvider"]
        )
        log.info("refine: loaded ONNX INT8 wav2vec2 session from %s", _ONNX_PATH.name)
    return _ort_session


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
    use_onnx = _USE_ONNX and _ensure_onnx_file()
    if use_onnx:
        processor = _load_processor_only()
    else:
        _, processor = _load_model()
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

    # Forward pass — the expensive step that dominates Block 3's wall-clock.
    # Try ONNX INT8 first when enabled; on any failure, fall back to the
    # PyTorch FP32 path so a busted .onnx file or a runtime error degrades
    # gracefully instead of breaking every job.
    logits = None
    if use_onnx:
        try:
            session = _load_onnx_session()
            logits_np = session.run(None, {"input_values": input_values.numpy()})[0]
            logits = torch.from_numpy(logits_np)
        except Exception as e:
            log.warning("refine: ONNX forward failed (%s); falling back to PyTorch", e)
            logits = None
    if logits is None:
        model, _ = _load_model()
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


def _split_indices_on_wav2vec2_gaps(
    i0: int, i1: int, timings: list[tuple[float, float] | None]
) -> list[int]:
    """Return word indices AFTER which to split (i.e. the last word of each
    pre-split sub-range), based on wav2vec2 inter-word gaps ≥ threshold.

    Empty list = no internal silence found, keep the range as-is.
    """
    splits: list[int] = []
    for k in range(i0, i1):
        t_k = timings[k] if k < len(timings) else None
        t_k1 = timings[k + 1] if k + 1 < len(timings) else None
        if t_k is None or t_k1 is None:
            continue
        gap = t_k1[0] - t_k[1]
        if gap >= INTERNAL_SILENCE_SPLIT_S:
            splits.append(k)
    return splits


def _build_sub_range(
    dg_words: list[dict],
    timings: list[tuple[float, float] | None],
    i_start: int,
    i_end: int,
) -> dict:
    """Build a range dict for words[i_start..i_end] using wav2vec2 timings
    for start/end AND for the surrounding silence durations.

    pre_silence_ms = wav2vec2 gap from previous word (0 if first in source)
    post_silence_ms = wav2vec2 gap to next word (0 if last in source)

    Using wav2vec2 gaps here (not Deepgram) so render-time padding logic sees
    the actual acoustic silence available, not Deepgram's 0ms-everywhere lie.
    """
    t_start = timings[i_start]
    t_end = timings[i_end]
    start_s = float(t_start[0]) if t_start is not None else float(dg_words[i_start]["start"])
    end_s = float(t_end[1]) if t_end is not None else float(dg_words[i_end]["end"])

    pre_silence_ms = 0.0
    if i_start > 0:
        t_prev = timings[i_start - 1]
        if t_prev is not None and t_start is not None:
            pre_silence_ms = max(0.0, (t_start[0] - t_prev[1]) * 1000.0)
        else:
            pre_silence_ms = max(
                0.0, (dg_words[i_start]["start"] - dg_words[i_start - 1]["end"]) * 1000.0
            )

    post_silence_ms = 0.0
    if i_end + 1 < len(dg_words):
        t_next = timings[i_end + 1] if i_end + 1 < len(timings) else None
        if t_next is not None and t_end is not None:
            post_silence_ms = max(0.0, (t_next[0] - t_end[1]) * 1000.0)
        else:
            post_silence_ms = max(
                0.0, (dg_words[i_end + 1]["start"] - dg_words[i_end]["end"]) * 1000.0
            )

    return {
        "start": round(start_s, 3),
        "end": round(end_s, 3),
        "start_word_idx": i_start,
        "end_word_idx": i_end,
        "pre_silence_ms": round(pre_silence_ms, 1),
        "post_silence_ms": round(post_silence_ms, 1),
    }


def refine_ranges(
    clip_json: dict,
    audio_path: str | Path,
    dg_words: list[dict],
    split_internal_silence: bool = True,
    internal_silence_threshold_s: float | None = None,
    preloaded_wav_bytes: bytes | None = None,
) -> dict:
    """Two-step range refinement:

    1) wav2vec2 + CTC forced alignment over the full Deepgram word sequence,
       producing per-word precise (start, end) timings.
    2) For each kept range, walk consecutive word pairs and split the range
       at any wav2vec2-measured gap ≥ threshold. Each sub-range gets
       wav2vec2 boundaries on both edges, and pre/post_silence_ms
       computed from wav2vec2 inter-word gaps (so render-time padding has
       correct headroom for the post-split sub-ranges).

    Internal-silence behavior is controlled by two parameters:

      split_internal_silence=True (default) + internal_silence_threshold_s=None
        → Production behavior. Split at any gap ≥ INTERNAL_SILENCE_SPLIT_S
          (0.2s). Aggressively kills internal silence but can chop natural
          breaths and sound staccato.

      split_internal_silence=True + internal_silence_threshold_s=X (positive float)
        → Surgical mode. Preserve gaps < X as audible time (natural breaths),
          split anything ≥ X (dramatic mid-sentence pauses, hesitations).
          Example: 0.7 keeps breaths up to 700ms but drops anything longer.

      split_internal_silence=False
        → Preserve all internal silence (legacy "keep all" mode). Equivalent
          to passing a huge threshold. Natural cadence intact but dramatic
          mid-sentence pauses play in full.

    If both parameters are set in a way that disagrees, the threshold wins
    (an explicit threshold overrides the boolean).

    The original clip_json is not mutated — a deep copy is returned.

    If wav2vec2 alignment fails for a range's head or tail word, the original
    Deepgram-based timestamps for that range are preserved.
    """
    # Resolve the effective threshold. The explicit threshold wins over the
    # boolean so the surgical and legacy modes can coexist on the same caller.
    if internal_silence_threshold_s is not None:
        threshold = float(internal_silence_threshold_s)
    elif split_internal_silence:
        threshold = INTERNAL_SILENCE_SPLIT_S
    else:
        threshold = float("inf")

    # If cleanup.run_pipeline already extracted the audio for Deepgram, it
    # passes the WAV bytes through so we don't re-spawn ffmpeg over the
    # source video (saves ~7s per job on typical 1080p clips).
    if preloaded_wav_bytes is not None:
        import io
        audio, sr = sf.read(io.BytesIO(preloaded_wav_bytes), dtype="float32")
        if sr != _SAMPLE_RATE:
            raise RuntimeError(
                f"preloaded wav sample rate {sr} != expected {_SAMPLE_RATE}"
            )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    else:
        audio = _extract_audio(Path(audio_path))
    norm_words = [_normalize_token(w.get("word", "")) for w in dg_words]
    timings = _align_words(audio, norm_words)

    split_count = 0
    out = copy.deepcopy(clip_json)
    for seg in out.get("segments", []):
        new_ranges: list[dict] = []
        for rng in seg.get("ranges", []):
            i0 = rng.get("start_word_idx")
            i1 = rng.get("end_word_idx")
            if i0 is None or i1 is None or i0 < 0 or i1 >= len(timings):
                # Defensive: a range without proper word indices can't be
                # split or refined. Keep as-is.
                new_ranges.append(rng)
                continue

            # Find internal wav2vec2 gaps ≥ threshold to split on. With
            # threshold=inf, this returns an empty list — same shape as a
            # range with no internal silence.
            if threshold == float("inf"):
                split_after: list[int] = []
            else:
                split_after = _split_indices_on_wav2vec2_gaps_with_threshold(
                    i0, i1, timings, threshold
                )
                split_count += len(split_after)

            # Build sub-ranges. If no splits, this produces one sub-range
            # spanning [i0, i1] — same shape as the original range but with
            # wav2vec2-precise edge boundaries.
            sub_start = i0
            boundaries = split_after + [i1]
            for sub_end in boundaries:
                sub_ranges_built = _build_sub_range(dg_words, timings, sub_start, sub_end)
                new_ranges.append(sub_ranges_built)
                sub_start = sub_end + 1
        seg["ranges"] = new_ranges

    if threshold == float("inf"):
        log.info("refine: internal-silence splitting disabled; preserved all breaths in-place")
    else:
        log.info(
            "refine: split %d internal silence(s) at threshold=%.2fs",
            split_count, threshold,
        )
    return out


def _split_indices_on_wav2vec2_gaps_with_threshold(
    i0: int, i1: int, timings: list[tuple[float, float] | None], threshold_s: float
) -> list[int]:
    """Return word indices after which to split, based on wav2vec2 inter-word
    gaps ≥ threshold_s. The standalone helper exists so callers can pass an
    explicit threshold without monkey-patching the module-level constant.
    """
    splits: list[int] = []
    for k in range(i0, i1):
        t_k = timings[k] if k < len(timings) else None
        t_k1 = timings[k + 1] if k + 1 < len(timings) else None
        if t_k is None or t_k1 is None:
            continue
        gap = t_k1[0] - t_k[1]
        if gap >= threshold_s:
            splits.append(k)
    return splits
