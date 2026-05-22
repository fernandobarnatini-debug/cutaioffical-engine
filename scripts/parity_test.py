"""Parity smoke test: PyTorch FP32 wav2vec2 vs ONNX INT8 on real speech.

Loads a local <stem>_cut.mp4 + <stem>_script.txt pair (the words spoken in that
cut audio), runs forced alignment under both backends, and reports:

  - emission cosine similarity (target > 0.999)
  - per-word boundary deltas: median / p95 / max
  - wall-clock speedup factor (target ≥ 2×)

Usage:
    python scripts/parity_test.py IMG_0078
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taF
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_ID = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "cutaioffical_engine" / "data" / "wav2vec2_base_960h.int8.onnx"


def _normalize_token(w: str) -> str:
    return re.sub(r"[^A-Z']", "", w.upper())


def _extract_audio(video_path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le", out],
            check=True, timeout=120,
        )
        audio, sr = sf.read(out, dtype="float32")
        assert sr == SAMPLE_RATE
        return audio
    finally:
        Path(out).unlink(missing_ok=True)


def _build_targets(words, processor):
    vocab = processor.tokenizer.get_vocab()
    delim_id = vocab[processor.tokenizer.word_delimiter_token]
    target_ids = []
    spans = []
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
            spans.append((-1, -1))
            if target_ids and target_ids[-1] == delim_id:
                target_ids.pop()
        else:
            spans.append((start, end))
    return target_ids, spans


def _emission_to_word_times(emission, target_ids, spans, blank_id, num_samples):
    targets = torch.tensor([target_ids], dtype=torch.int32)
    aligned, scores = taF.forced_align(emission, targets, blank=blank_id)
    token_spans = taF.merge_tokens(aligned[0], scores[0].exp())
    if len(token_spans) != len(target_ids):
        raise RuntimeError(f"alignment span mismatch: {len(token_spans)} vs {len(target_ids)}")
    spf = num_samples / emission.shape[1]
    out = []
    for a, b in spans:
        if a < 0:
            out.append(None)
        else:
            s = token_spans[a].start * spf / SAMPLE_RATE
            e = token_spans[b].end * spf / SAMPLE_RATE
            out.append((float(s), float(e)))
    return out


def run_torch(audio, words, processor, model):
    target_ids, spans = _build_targets(words, processor)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs).logits
        emission = torch.log_softmax(logits, dim=-1)
    elapsed = time.perf_counter() - t0
    blank_id = processor.tokenizer.pad_token_id
    times = _emission_to_word_times(emission, target_ids, spans, blank_id, audio.shape[0])
    return times, emission, elapsed


def run_onnx(audio, words, processor, session):
    target_ids, spans = _build_targets(words, processor)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
    x = inputs.numpy()
    t0 = time.perf_counter()
    (logits_np,) = session.run(None, {"input_values": x})
    elapsed = time.perf_counter() - t0
    logits = torch.from_numpy(logits_np)
    emission = torch.log_softmax(logits, dim=-1)
    blank_id = processor.tokenizer.pad_token_id
    times = _emission_to_word_times(emission, target_ids, spans, blank_id, audio.shape[0])
    return times, emission, elapsed


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <stem>  (looks for <stem>_cut.mp4 + <stem>_script.txt)", file=sys.stderr)
        return 1
    stem = sys.argv[1]
    video = ROOT / f"{stem}_cut.mp4"
    script_p = ROOT / f"{stem}_script.txt"
    if not video.exists() or not script_p.exists():
        print(f"missing inputs: {video} and/or {script_p}", file=sys.stderr)
        return 1

    print(f"Loading audio: {video.name}")
    audio = _extract_audio(video)
    print(f"  {audio.shape[0]/SAMPLE_RATE:.2f}s of mono 16kHz audio")

    raw = script_p.read_text().split()
    words = [_normalize_token(w) for w in raw]
    words = [w for w in words if w]
    print(f"  {len(words)} normalized words")

    import torch as _t
    _t.set_num_threads(8)

    print(f"Loading PyTorch {MODEL_ID}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()

    print(f"Loading ONNX session: {ONNX_PATH.relative_to(ROOT)}")
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    session = ort.InferenceSession(ONNX_PATH.as_posix(), opts, providers=["CPUExecutionProvider"])

    # Warm both backends so we time steady-state, not first-call cost.
    print("Warming up...")
    _ = run_torch(audio[:SAMPLE_RATE], ["HELLO"], processor, model)
    _ = run_onnx(audio[:SAMPLE_RATE], ["HELLO"], processor, session)

    print("\n--- PyTorch FP32 forward pass ---")
    pt_times, pt_emission, pt_elapsed = run_torch(audio, words, processor, model)
    print(f"  forward pass: {pt_elapsed:.2f}s")

    print("\n--- ONNX INT8 forward pass ---")
    ox_times, ox_emission, ox_elapsed = run_onnx(audio, words, processor, session)
    print(f"  forward pass: {ox_elapsed:.2f}s")

    print(f"\nSpeedup: {pt_elapsed / ox_elapsed:.2f}×")

    # Emission cosine similarity (flatten across time/vocab).
    pt_flat = pt_emission.numpy().reshape(-1)
    ox_flat = ox_emission.numpy().reshape(-1)
    # Match frame counts if they differ by 1 due to convolution edge effects.
    n = min(pt_flat.size, ox_flat.size)
    cos = float(np.dot(pt_flat[:n], ox_flat[:n]) / (np.linalg.norm(pt_flat[:n]) * np.linalg.norm(ox_flat[:n])))
    print(f"Emission cosine similarity: {cos:.6f}  (target > 0.999)")

    # Per-word boundary deltas.
    deltas_ms = []
    miss = 0
    for pt, ox in zip(pt_times, ox_times):
        if pt is None or ox is None:
            miss += 1
            continue
        deltas_ms.append(abs(pt[0] - ox[0]) * 1000.0)
        deltas_ms.append(abs(pt[1] - ox[1]) * 1000.0)
    deltas = np.array(deltas_ms)
    print(f"\nBoundary deltas across {len(deltas)} boundaries ({miss} words skipped):")
    print(f"  median: {np.median(deltas):.2f} ms")
    print(f"  p95:    {np.percentile(deltas, 95):.2f} ms")
    print(f"  max:    {np.max(deltas):.2f} ms")

    print("\nVerdict:")
    ok = True
    if cos < 0.999:
        print(f"  FAIL emission cosine {cos:.4f} < 0.999")
        ok = False
    if np.median(deltas) >= 5:
        print(f"  FAIL median delta {np.median(deltas):.1f} ms >= 5 ms")
        ok = False
    if np.percentile(deltas, 95) >= 25:
        print(f"  FAIL p95 delta {np.percentile(deltas, 95):.1f} ms >= 25 ms")
        ok = False
    if np.max(deltas) >= 50:
        print(f"  WARN max delta {np.max(deltas):.1f} ms >= 50 ms")
    if pt_elapsed / ox_elapsed < 2.0:
        print(f"  WARN speedup {pt_elapsed/ox_elapsed:.2f}× < 2.0×")
    if ok:
        print("  PASS — quantization preserves alignment quality")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
