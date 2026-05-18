"""Audio waveform peaks generation.

Uses ffmpeg to extract raw mono samples from any container ffmpeg can decode
(MP4, MOV, MKV, etc.), then computes max-amplitude buckets for waveform
visualization.

Output is a list of floats in [0.0, 1.0] — one peak per bucket. The frontend
mirrors each peak from a horizontal center axis to render the classic
audio-editor waveform shape.

Server-side generation avoids the browser Web Audio API decode landmine
(Safari rejects MOV audio tracks via decodeAudioData; ffmpeg handles
everything).

Public API:
    generate_peaks(video_path, target_buckets=4000) -> list[float]
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path


def generate_peaks(
    video_path: str | Path,
    target_buckets: int = 4000,
    sample_rate: int = 8000,
) -> list[float]:
    """Extract `target_buckets` peak floats from the source audio.

    Pipeline:
      1. ffmpeg decodes audio to mono, sample_rate Hz, 32-bit float, raw
         little-endian via stdout.
      2. We bucket the absolute samples into target_buckets max-amplitude
         peaks.

    sample_rate=8000 is plenty for waveform visualization (well above the
    Nyquist for speech, dramatically smaller than 48kHz).

    Args:
        video_path: any container ffmpeg can decode (MP4, MOV, MKV, etc.).
        target_buckets: how many peaks to emit. The frontend draws 1 bar per
            ~2 pixels, so 4000 buckets covers a ~8000px-wide timeline at full
            zoom without aliasing.
        sample_rate: target audio sample rate after ffmpeg's aresample. 8000
            Hz is the floor for intelligible speech and keeps decode fast.

    Returns:
        list[float] of length target_buckets, each in [0.0, 1.0] (max
        absolute amplitude in that bucket, peaks may exceed 1.0 on overdriven
        sources — clamped to 1.0).

    Raises:
        RuntimeError: ffmpeg decode failed.
        ValueError: source produced zero audio samples.
    """
    video_path = Path(video_path)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vn",                  # ignore video
        "-ac", "1",             # mono
        "-ar", str(sample_rate),
        "-f", "f32le",          # 32-bit little-endian floats
        "-",                    # stdout
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        tail = (result.stderr or b"").decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"ffmpeg peaks decode failed (exit {result.returncode}):\n{tail}")

    raw = result.stdout
    if not raw:
        raise ValueError("ffmpeg produced no audio samples (no audio track?)")

    # Unpack as float32; each sample is 4 bytes. Reject misaligned output
    # rather than silently dropping a partial trailing sample.
    if len(raw) % 4 != 0:
        raise RuntimeError(
            f"ffmpeg returned non-4-byte-aligned audio stream ({len(raw)} bytes)"
        )
    sample_count = len(raw) // 4
    if sample_count == 0:
        raise ValueError("ffmpeg produced no audio samples (no audio track?)")
    samples = struct.unpack(f"<{sample_count}f", raw)

    samples_per_bucket = max(1, sample_count // target_buckets)
    peaks: list[float] = []
    for i in range(target_buckets):
        start = i * samples_per_bucket
        end = min(start + samples_per_bucket, sample_count)
        if end <= start:
            peaks.append(0.0)
            continue
        max_amp = 0.0
        for j in range(start, end):
            v = abs(samples[j])
            if v > max_amp:
                max_amp = v
        peaks.append(min(1.0, max_amp))
    return peaks
