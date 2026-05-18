"""Block 4 — CUT render.

Takes a source video + the clip_json output of align() and produces a single
MP4 by ffmpeg-trimming each kept range and concatenating them.

Per-range padding ports the proven logic from ~/Clip/pipeline.py: head/tail
targets with a per-word silence soft ceiling and a half-inter-range-gap hard
cap, plus a floor to prevent word-start clipping at cut boundaries.

Public API:
    render(video_path, clip_json, output_path) -> None
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# Padding constants — same as the working ~/Clip/pipeline.py.
HEAD_TARGET_MS = 60
TAIL_TARGET_MS = 120
PAD_FLOOR_MS = 25


def _flatten_ranges(clip_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull all ranges across all segments in timeline order.

    Tolerant to missing pre_silence_ms / post_silence_ms (default 0).
    """
    segments = clip_json.get("segments")
    if not isinstance(segments, list) or not segments:
        return []
    flat: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        ranges = seg.get("ranges")
        if not isinstance(ranges, list):
            continue
        for r in ranges:
            if not isinstance(r, dict):
                continue
            start = r.get("start")
            end = r.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            if end <= start:
                continue
            flat.append({
                "start": float(start),
                "end": float(end),
                "pre_silence_ms": float(r.get("pre_silence_ms") or 0),
                "post_silence_ms": float(r.get("post_silence_ms") or 0),
            })
    # Sort by start time defensively — ranges should already be ordered but
    # belt-and-suspenders.
    flat.sort(key=lambda x: x["start"])
    return flat


def _padded_ranges(flat: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Expand each range by head/tail padding, applying soft ceiling + hard cap.

    Returns a list of (start_sec, end_sec) tuples ready for ffmpeg.
    """
    out: list[tuple[float, float]] = []
    for i, r in enumerate(flat):
        head_hard_cap = float("inf")
        tail_hard_cap = float("inf")
        if i > 0:
            inter_ms = max(0.0, (r["start"] - flat[i - 1]["end"]) * 1000.0)
            head_hard_cap = inter_ms / 2.0
        if i + 1 < len(flat):
            inter_ms = max(0.0, (flat[i + 1]["start"] - r["end"]) * 1000.0)
            tail_hard_cap = inter_ms / 2.0

        head_pad = max(PAD_FLOOR_MS, min(HEAD_TARGET_MS, r["pre_silence_ms"]))
        head_pad = max(0.0, min(head_pad, head_hard_cap))
        tail_pad = max(PAD_FLOOR_MS, min(TAIL_TARGET_MS, r["post_silence_ms"]))
        tail_pad = max(0.0, min(tail_pad, tail_hard_cap))

        out.append((
            max(0.0, r["start"] - head_pad / 1000.0),
            r["end"] + tail_pad / 1000.0,
        ))
    return out


def _build_filter_complex(ranges: list[tuple[float, float]]) -> str:
    parts: list[str] = []
    labels: list[str] = []
    for i, (s, e) in enumerate(ranges):
        parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    filtergraph = ";".join(parts)
    filtergraph += ";" + "".join(labels)
    filtergraph += f"concat=n={len(ranges)}:v=1:a=1[vout][aout]"
    return filtergraph


def render(
    video_path: str | Path,
    clip_json: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Render the kept ranges from clip_json into a single MP4 at output_path.

    Raises:
        ValueError: clip_json has no valid ranges to render.
        RuntimeError: ffmpeg invocation failed (stderr tail included).
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    flat = _flatten_ranges(clip_json)
    if not flat:
        raise ValueError("no ranges to render")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranges = _padded_ranges(flat)
    filtergraph = _build_filter_complex(ranges)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-filter_complex", filtergraph,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "")[-1500:]
        raise RuntimeError(f"ffmpeg render failed (exit {result.returncode}):\n{tail}")
