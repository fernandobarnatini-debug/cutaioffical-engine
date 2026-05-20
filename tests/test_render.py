"""Tests for render.py (Block 4 — CUT).

Validates:
1. _flatten_ranges handles tolerant input shapes (missing pad fields default to 0)
2. _padded_ranges enforces floor + target + hard-cap rules
3. render() raises ValueError when no ranges are valid
4. render() raises RuntimeError when ffmpeg fails (bad input path)
5. Smoke: render() against a tiny ffmpeg-generated source produces a valid MP4
   whose duration ≈ sum of padded ranges (within 0.5s tolerance for encoding)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutaioffical_engine.render import (
    HEAD_TARGET_MS,
    PAD_FLOOR_MS,
    TAIL_TARGET_MS,
    _flatten_ranges,
    _padded_ranges,
    render,
)


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_test_source(out_path: Path, duration_sec: int = 15) -> None:
    """Generate a synthetic test video + audio with ffmpeg.

    testsrc=720x480 + sine=440Hz, 15 seconds. Small file, valid container.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=320x240:rate=30",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_sec}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"test source generation failed: {result.stderr[-500:]}")


def _probe_duration(path: Path) -> float:
    """Use ffprobe to get the duration of a rendered MP4."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


class TestFlattenRanges(unittest.TestCase):
    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(_flatten_ranges({}), [])
        self.assertEqual(_flatten_ranges({"segments": []}), [])
        self.assertEqual(_flatten_ranges({"segments": None}), [])

    def test_missing_pad_fields_default_to_zero(self) -> None:
        clip = {"segments": [{"ranges": [{"start": 1.0, "end": 2.0}]}]}
        flat = _flatten_ranges(clip)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["pre_silence_ms"], 0.0)
        self.assertEqual(flat[0]["post_silence_ms"], 0.0)

    def test_invalid_range_skipped(self) -> None:
        clip = {
            "segments": [
                {"ranges": [
                    {"start": 1.0, "end": 2.0},
                    {"start": "bad", "end": 3.0},
                    {"start": 4.0, "end": 4.0},  # end == start, invalid
                    {"start": 5.0, "end": 6.0},
                ]}
            ]
        }
        flat = _flatten_ranges(clip)
        self.assertEqual(len(flat), 2)
        self.assertEqual(flat[0]["start"], 1.0)
        self.assertEqual(flat[1]["start"], 5.0)

    def test_ranges_sorted_by_start(self) -> None:
        clip = {
            "segments": [
                {"ranges": [{"start": 10.0, "end": 11.0}]},
                {"ranges": [{"start": 2.0, "end": 3.0}]},
            ]
        }
        flat = _flatten_ranges(clip)
        self.assertEqual([r["start"] for r in flat], [2.0, 10.0])


class TestPaddedRanges(unittest.TestCase):
    def test_floor_applies_at_isolated_range(self) -> None:
        # One range, no neighbors: pads should be FLOOR (no per-word silence)
        flat = [{"start": 5.0, "end": 6.0, "pre_silence_ms": 0, "post_silence_ms": 0}]
        out = _padded_ranges(flat)
        self.assertEqual(len(out), 1)
        s, e = out[0]
        # Floor of 25ms on both sides: start should be 5.0 - 0.025 = 4.975
        self.assertAlmostEqual(s, 5.0 - PAD_FLOOR_MS / 1000.0, places=4)
        self.assertAlmostEqual(e, 6.0 + PAD_FLOOR_MS / 1000.0, places=4)

    def test_target_applies_with_sufficient_silence(self) -> None:
        # Long-clip (≥1.5s) range with abundant silence on both sides —
        # should hit HEAD_TARGET_MS (0 by default) and TAIL_TARGET_MS (30).
        flat = [
            {"start": 5.0, "end": 7.0, "pre_silence_ms": 500, "post_silence_ms": 500},
        ]
        out = _padded_ranges(flat)
        s, e = out[0]
        self.assertAlmostEqual(s, 5.0 - HEAD_TARGET_MS / 1000.0, places=4)
        self.assertAlmostEqual(e, 7.0 + TAIL_TARGET_MS / 1000.0, places=4)

    def test_short_clip_gets_attack_window(self) -> None:
        # Short-clip (<1.5s) standalone utterance — gets larger head + tail
        # padding (SHORT_CLIP_HEAD_MS / SHORT_CLIP_TAIL_MS) than sentence-flow
        # cuts, to compensate for wav2vec2's tight phoneme-core snapping that
        # misses the natural VOT prep and voiced-consonant resonance decay.
        # This is what makes structural-repetition items ("One.", "Two.",
        # "Not one") play at natural counting cadence instead of rushed.
        from cutaioffical_engine.render import SHORT_CLIP_HEAD_MS, SHORT_CLIP_TAIL_MS
        flat = [
            {"start": 5.0, "end": 5.4, "pre_silence_ms": 500, "post_silence_ms": 500},
        ]
        out = _padded_ranges(flat)
        s, e = out[0]
        self.assertAlmostEqual(s, 5.0 - SHORT_CLIP_HEAD_MS / 1000.0, places=4)
        self.assertAlmostEqual(e, 5.4 + SHORT_CLIP_TAIL_MS / 1000.0, places=4)

    def test_hard_cap_prevents_overlap(self) -> None:
        # Two ranges, gap of 50ms between them. Hard cap = 25ms per side.
        flat = [
            {"start": 1.0, "end": 2.0, "pre_silence_ms": 500, "post_silence_ms": 500},
            {"start": 2.05, "end": 3.0, "pre_silence_ms": 500, "post_silence_ms": 500},
        ]
        out = _padded_ranges(flat)
        # First range's tail_pad capped at half the 50ms gap = 25ms
        s1, e1 = out[0]
        s2, e2 = out[1]
        self.assertLessEqual(e1, s2)  # no overlap


class TestRender(unittest.TestCase):
    def test_raises_on_no_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                render("/nonexistent.mov", {"segments": []}, Path(tmp) / "out.mp4")

    def test_raises_on_ffmpeg_failure(self) -> None:
        if not _has_ffmpeg():
            self.skipTest("ffmpeg not available")
        clip = {"segments": [{"ranges": [{"start": 0.5, "end": 1.0, "pre_silence_ms": 50, "post_silence_ms": 50}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                render("/nonexistent/path.mov", clip, Path(tmp) / "out.mp4")
            self.assertIn("ffmpeg", str(ctx.exception).lower())

    def test_smoke_render_against_synthetic_source(self) -> None:
        if not _has_ffmpeg() or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe not available")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "source.mp4"
            _make_test_source(src, duration_sec=15)

            # Two kept ranges with light padding hints. Total kept content = 5s.
            clip = {
                "segments": [
                    {"span_index": 0, "text": "first", "ranges": [
                        {"start": 2.0, "end": 4.0, "pre_silence_ms": 100, "post_silence_ms": 100},
                    ]},
                    {"span_index": 1, "text": "second", "ranges": [
                        {"start": 8.0, "end": 11.0, "pre_silence_ms": 100, "post_silence_ms": 100},
                    ]},
                ]
            }
            out = tmp_path / "cut.mp4"
            render(src, clip, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)  # non-empty output

            # Expected duration: kept content (2 + 3 = 5s) + total padding.
            # Both ranges hit pad target on both sides: 60 + 120 ms each.
            # Two ranges → 2 * (0.06 + 0.12) = 0.36s padding total.
            expected = 5.0 + 0.36
            actual = _probe_duration(out)
            self.assertAlmostEqual(actual, expected, delta=0.5)


if __name__ == "__main__":
    unittest.main()
