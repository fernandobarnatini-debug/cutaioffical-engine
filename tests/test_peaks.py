"""Tests for peaks.py (audio waveform generation).

Validates:
1. generate_peaks() returns exactly target_buckets floats in [0.0, 1.0]
2. A silent source produces all-zero peaks
3. A non-silent source produces some non-zero peaks
4. Missing source raises RuntimeError
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutaioffical_engine.peaks import generate_peaks


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_audio_source(out_path: Path, duration_sec: int, silent: bool = False) -> None:
    """Generate a synthetic video with audio (sine or silence)."""
    audio_filter = "anullsrc=r=44100" if silent else "sine=frequency=440"
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=320x240:rate=30",
        "-f", "lavfi",
        "-i", f"{audio_filter}:duration={duration_sec}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"test source generation failed: {result.stderr[-500:]}")


class TestGeneratePeaks(unittest.TestCase):
    def test_raises_on_missing_source(self) -> None:
        if not _has_ffmpeg():
            self.skipTest("ffmpeg not available")
        with self.assertRaises(RuntimeError):
            generate_peaks("/nonexistent/path.mov", target_buckets=100)

    def test_returns_target_bucket_count(self) -> None:
        if not _has_ffmpeg():
            self.skipTest("ffmpeg not available")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "audio.mp4"
            _make_audio_source(src, duration_sec=5)
            peaks = generate_peaks(src, target_buckets=400)
            self.assertEqual(len(peaks), 400)
            for p in peaks:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_silent_source_produces_zero_peaks(self) -> None:
        if not _has_ffmpeg():
            self.skipTest("ffmpeg not available")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "silence.mp4"
            _make_audio_source(src, duration_sec=3, silent=True)
            peaks = generate_peaks(src, target_buckets=200)
            self.assertEqual(len(peaks), 200)
            self.assertTrue(all(p < 0.001 for p in peaks), f"silent source had non-zero peaks; max={max(peaks)}")

    def test_sine_source_produces_nonzero_peaks(self) -> None:
        if not _has_ffmpeg():
            self.skipTest("ffmpeg not available")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sine.mp4"
            _make_audio_source(src, duration_sec=3, silent=False)
            peaks = generate_peaks(src, target_buckets=200)
            self.assertEqual(len(peaks), 200)
            non_zero = [p for p in peaks if p > 0.05]
            self.assertGreater(len(non_zero), 100, "sine wave should produce mostly non-trivial peaks")

    def test_video_with_no_audio_track_raises(self) -> None:
        if not _has_ffmpeg():
            self.skipTest("ffmpeg not available")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "video_only.mp4"
            # Generate a video with NO audio track.
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", "testsrc=duration=2:size=160x120:rate=15",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-an",
                    str(src),
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                self.skipTest(f"audioless source generation failed: {result.stderr[-200:]}")
            # ffmpeg's -i with no audio track + -ac 1 fails outright; expect either ValueError or RuntimeError.
            with self.assertRaises((ValueError, RuntimeError)):
                generate_peaks(src, target_buckets=100)


if __name__ == "__main__":
    unittest.main()
