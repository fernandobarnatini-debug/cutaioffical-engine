"""Tests for overlap.py (Block 4 — overlap variant).

Validates:
1. _plan_overlap uses padded video bounds and reduces silence budget by the
   padding render() already consumed.
2. min_overlap_ms gate collapses sub-threshold extensions to 0.
3. neighbor_ratio caps each extension by the neighbor's padded video duration.
4. First/last extensions clamp to zero on the open side, audio clamps to source
   edges.
5. Collision rule — post wins. Tail of i-1 truncates head of i.
6. Filter_complex string contains the expected nodes (video concat, asplit,
   per-branch atrim+adelay, amix).
7. Smoke: overlap_render() against a synthetic source produces a playable MP4
   with iOS-Safari-safe codec flags.
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

from cutaioffical_engine.overlap import (
    _build_filter_complex,
    _plan_overlap,
    overlap_render,
)
from cutaioffical_engine.render import _flatten_ranges, _padded_ranges_with_meta


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_test_source(out_path: Path, duration_sec: int = 15) -> None:
    cmd = [
        "ffmpeg", "-y",
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


def _ffprobe_json(path: Path) -> dict:
    import json
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def _plan_from_clip(clip: dict, source_duration: float, **kwargs) -> list[dict]:
    flat = _flatten_ranges(clip)
    padded = _padded_ranges_with_meta(flat)
    return _plan_overlap(padded, source_duration=source_duration, **kwargs)


class TestPlanOverlap(unittest.TestCase):
    """Math-level tests — no ffmpeg required."""

    def _kwargs(self, **overrides) -> dict:
        kw = dict(max_overlap_ms=500, min_overlap_ms=50, neighbor_ratio=0.5)
        kw.update(overrides)
        return kw

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(_plan_overlap([], source_duration=10.0, **self._kwargs()), [])

    def test_silence_budget_reduced_by_head_pad(self) -> None:
        # Range with 600ms pre_silence, padding will consume some of it.
        # remaining_pre = 600 - head_pad_applied. Whatever's left should drive
        # the audio extension, capped at max_overlap_ms (500).
        clip = {
            "segments": [{"ranges": [
                {"start": 5.0, "end": 7.0, "pre_silence_ms": 600, "post_silence_ms": 600},
                {"start": 10.0, "end": 12.0, "pre_silence_ms": 600, "post_silence_ms": 600},
            ]}]
        }
        flat = _flatten_ranges(clip)
        padded = _padded_ranges_with_meta(flat)
        plan = _plan_overlap(padded, source_duration=20.0, **self._kwargs())

        # First range: back_ext should be 0 (open side).
        self.assertEqual(plan[0]["back_ext"], 0.0)
        # Last range: fwd_ext should be 0 (open side).
        self.assertEqual(plan[-1]["fwd_ext"], 0.0)

        # Second range's back_ext = remaining_pre = (600 - head_pad_ms) / 1000,
        # capped by max_overlap_ms (500ms = 0.5s) AND by neighbor_ratio * prev
        # padded duration. Just assert it's > 0 and <= 0.5.
        self.assertGreater(plan[1]["back_ext"], 0.0)
        self.assertLessEqual(plan[1]["back_ext"], 0.5 + 1e-6)

    def test_min_overlap_gate_collapses_tiny_extensions(self) -> None:
        # 40ms pre/post silence is under the 50ms min_overlap_ms gate. Even
        # after padding (keoni_tight head=0/tail=40 for >=1.5s ranges) the
        # remaining budget would be tiny → must collapse to 0 to avoid
        # imperceptible bleed nodes in the filter graph.
        clip = {
            "segments": [{"ranges": [
                {"start": 1.0, "end": 3.0, "pre_silence_ms": 40, "post_silence_ms": 40},
                {"start": 5.0, "end": 7.0, "pre_silence_ms": 40, "post_silence_ms": 40},
            ]}]
        }
        plan = _plan_from_clip(clip, source_duration=10.0, **self._kwargs())
        for p in plan:
            self.assertEqual(p["back_ext"], 0.0)
            self.assertEqual(p["fwd_ext"], 0.0)

    def test_neighbor_ratio_caps_extension(self) -> None:
        # Plenty of silence (1500ms each side) and a short neighbor (200ms).
        # back_ext should be capped at neighbor_ratio * prev_padded_dur,
        # not the full max_overlap_ms.
        clip = {
            "segments": [{"ranges": [
                {"start": 2.0, "end": 2.2, "pre_silence_ms": 1500, "post_silence_ms": 1500},
                {"start": 4.0, "end": 6.0, "pre_silence_ms": 1500, "post_silence_ms": 1500},
            ]}]
        }
        kw = self._kwargs(neighbor_ratio=0.5)
        plan = _plan_from_clip(clip, source_duration=10.0, **kw)
        # plan[1].back_ext capped by 0.5 * prev_padded_dur. Prev padded dur is
        # ~0.2s + head/tail pad. Cap should be roughly 0.1s + a hair, well
        # under the 0.5s max_overlap_ms.
        self.assertLess(plan[1]["back_ext"], 0.3)

    def test_first_last_open_sides_zero(self) -> None:
        clip = {
            "segments": [{"ranges": [
                {"start": 1.0, "end": 3.0, "pre_silence_ms": 800, "post_silence_ms": 800},
                {"start": 5.0, "end": 7.0, "pre_silence_ms": 800, "post_silence_ms": 800},
                {"start": 9.0, "end": 11.0, "pre_silence_ms": 800, "post_silence_ms": 800},
            ]}]
        }
        plan = _plan_from_clip(clip, source_duration=15.0, **self._kwargs())
        self.assertEqual(plan[0]["back_ext"], 0.0)
        self.assertEqual(plan[-1]["fwd_ext"], 0.0)

    def test_audio_clamped_to_source_edges(self) -> None:
        # Last range pushes audio_end past source_duration. Should clamp.
        clip = {
            "segments": [{"ranges": [
                {"start": 1.0, "end": 3.0, "pre_silence_ms": 800, "post_silence_ms": 800},
                {"start": 5.0, "end": 9.5, "pre_silence_ms": 800, "post_silence_ms": 800},
            ]}]
        }
        plan = _plan_from_clip(clip, source_duration=10.0, **self._kwargs())
        self.assertLessEqual(plan[-1]["audio_end"], 10.0 + 1e-6)

    def test_collision_rule_post_wins(self) -> None:
        # Two close ranges with big silence allowances. Configure so the prev
        # tail would extend INTO the next head's extension window. Collision
        # rule should push the next head's audio_start = prev audio_end.
        clip = {
            "segments": [{"ranges": [
                {"start": 1.0, "end": 2.0, "pre_silence_ms": 800, "post_silence_ms": 800},
                {"start": 2.6, "end": 3.6, "pre_silence_ms": 800, "post_silence_ms": 800},
            ]}]
        }
        plan = _plan_from_clip(clip, source_duration=10.0, **self._kwargs())
        # audio_start[1] must equal audio_end[0] (or be capped at padded_start
        # of range 1 if collision would otherwise push it past).
        if plan[1]["audio_start"] != plan[0]["audio_end"]:
            # Capped case — audio_start can't exceed padded_start.
            from cutaioffical_engine.render import _flatten_ranges as _f, _padded_ranges_with_meta as _pwm
            padded = _pwm(_f(clip))
            self.assertEqual(plan[1]["audio_start"], padded[1]["padded_start"])
        else:
            self.assertEqual(plan[1]["audio_start"], plan[0]["audio_end"])


class TestFilterComplex(unittest.TestCase):
    def test_filter_complex_has_expected_nodes(self) -> None:
        plan = [
            {"video_start": 1.0, "video_end": 2.0, "audio_start": 0.8,
             "audio_end": 2.2, "delay_ms": 0, "back_ext": 0.2, "fwd_ext": 0.2},
            {"video_start": 5.0, "video_end": 6.0, "audio_start": 4.7,
             "audio_end": 6.3, "delay_ms": 1000, "back_ext": 0.3, "fwd_ext": 0.3},
        ]
        fg = _build_filter_complex(plan)
        self.assertIn("[0:v]trim=1:2", fg)
        self.assertIn("[0:v]trim=5:6", fg)
        self.assertIn("concat=n=2:v=1:a=0[vout]", fg)
        self.assertIn("aresample=async=1:first_pts=0,asplit=2[s0][s1]", fg)
        self.assertIn("atrim=0.8:2.2", fg)
        self.assertIn("atrim=4.7:6.3", fg)
        self.assertIn("adelay=0:all=1", fg)
        self.assertIn("adelay=1000:all=1", fg)
        self.assertIn("amix=inputs=2:normalize=0:dropout_transition=0[aout]", fg)

    def test_empty_plan_raises(self) -> None:
        with self.assertRaises(ValueError):
            _build_filter_complex([])


@unittest.skipUnless(_has_ffmpeg(), "ffmpeg/ffprobe not on PATH")
class TestOverlapRenderSmoke(unittest.TestCase):
    """End-to-end render against a synthetic source. Requires ffmpeg + ffprobe."""

    def test_synthetic_source_renders_playable_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            out = Path(td) / "out.mp4"
            _make_test_source(src, duration_sec=15)

            clip = {
                "segments": [{"ranges": [
                    {"start": 1.0, "end": 3.0, "pre_silence_ms": 800, "post_silence_ms": 800},
                    {"start": 6.0, "end": 8.0, "pre_silence_ms": 800, "post_silence_ms": 800},
                    {"start": 11.0, "end": 13.0, "pre_silence_ms": 800, "post_silence_ms": 800},
                ]}]
            }
            overlap_render(src, clip, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1024)

            info = _ffprobe_json(out)
            # Codec check — iOS-Safari-safe flags survived the encode.
            v = next(s for s in info["streams"] if s["codec_type"] == "video")
            a = next(s for s in info["streams"] if s["codec_type"] == "audio")
            self.assertEqual(v["codec_name"], "h264")
            self.assertEqual(v["pix_fmt"], "yuv420p")
            self.assertEqual(a["codec_name"], "aac")

            # Duration should be roughly sum of padded video durations (~6s).
            duration = float(info["format"]["duration"])
            self.assertGreater(duration, 5.0)
            self.assertLess(duration, 7.5)


if __name__ == "__main__":
    unittest.main()
