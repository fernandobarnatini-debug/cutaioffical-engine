"""End-to-end check: drive refine._align_words exactly as the worker would,
under CUTAIOFFICAL_USE_ONNX_WAV2VEC2=1 then =0, and compare per-word timings.

Usage: python scripts/end_to_end_check.py <stem>
  expects <stem>_cut.mp4 + <stem>_script.txt locally.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())


def extract_audio(video: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", out],
            check=True, timeout=120,
        )
        audio, sr = sf.read(out, dtype="float32")
        assert sr == 16000
        return audio
    finally:
        Path(out).unlink(missing_ok=True)


def normalize(w: str) -> str:
    return re.sub(r"[^A-Z']", "", w.upper())


def run_once(audio, words, use_onnx: bool):
    os.environ["CUTAIOFFICAL_USE_ONNX_WAV2VEC2"] = "1" if use_onnx else "0"
    # Fresh import so the _USE_ONNX module-level constant re-evaluates.
    import cutaioffical_engine.refine as refine
    importlib.reload(refine)
    # Reset singletons across reloads so we time clean state.
    refine._model = None
    refine._processor = None
    refine._ort_session = None

    # Warm-up: pay the model-load + session-load + first-forward jit cost once.
    refine._align_words(audio[:16000], ["HELLO"])

    t0 = time.perf_counter()
    times = refine._align_words(audio, words)
    elapsed = time.perf_counter() - t0
    return times, elapsed, refine._USE_ONNX


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: end_to_end_check.py <stem>")
        return 1
    stem = sys.argv[1]
    video = ROOT / f"{stem}_cut.mp4"
    script = ROOT / f"{stem}_script.txt"
    audio = extract_audio(video)
    raw = script.read_text().split()
    words = [normalize(w) for w in raw if normalize(w)]
    print(f"audio: {audio.shape[0]/16000:.2f}s   words: {len(words)}")

    print("\nrun with USE_ONNX=1 ...")
    t_onnx, dt_onnx, on1 = run_once(audio, words, True)
    print(f"  refine._USE_ONNX={on1}, _align_words elapsed: {dt_onnx:.2f}s")

    print("\nrun with USE_ONNX=0 ...")
    t_pt, dt_pt, on2 = run_once(audio, words, False)
    print(f"  refine._USE_ONNX={on2}, _align_words elapsed: {dt_pt:.2f}s")

    print(f"\nSpeedup (PyTorch / ONNX): {dt_pt/dt_onnx:.2f}×")

    deltas = []
    for a, b in zip(t_onnx, t_pt):
        if a is None or b is None:
            continue
        deltas.append(abs(a[0] - b[0]) * 1000)
        deltas.append(abs(a[1] - b[1]) * 1000)
    d = np.array(deltas)
    print(f"\nPer-boundary deltas ({len(d)} edges):")
    print(f"  median {np.median(d):.2f} ms | p95 {np.percentile(d, 95):.2f} ms | max {np.max(d):.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
