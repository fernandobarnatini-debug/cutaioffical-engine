"""Block 4 (overlap variant) — CUT render with J/L-cut audio bleed.

Same padded video timeline as render.py — every clip boundary is identical to
a non-overlap render. The audio stream is allowed to extend *further* into the
remaining silence on either side of each clip, so the trailing breath of one
speaker bleeds into the start of the next clip (and vice versa) just like
dragging audio handles past a video cut in CapCut.

Public API:
    overlap_render(video_path, clip_json, output_path) -> None

Behavior comes from the validated ~/overlap/ prototype:
    - desired_back/fwd  = remaining_silence (post-padding) capped at max_overlap_ms
    - neighbor cap      = neighbor padded-video duration × neighbor_ratio
    - min_overlap gate  = sub-50ms residuals collapse to 0
    - collision rule    = post wins (prev tail truncates next head)

Output flags mirror render.py exactly (libx264 high@4.2, yuv420p, +faststart,
aac 192k) so iOS Safari / QuickTime playback is unaffected.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .render import _flatten_ranges, _padded_ranges_with_meta

# Audio-bleed knobs. Defaults match the validated overlap v2 prototype.
DEFAULT_MAX_OVERLAP_MS = 500
DEFAULT_MIN_OVERLAP_MS = 50
DEFAULT_NEIGHBOR_RATIO = 0.5


def _probe_duration(video_path: Path) -> float:
    """Return source duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {result.stderr}")
    return float(result.stdout.strip())


def _plan_overlap(
    padded: list[dict[str, Any]],
    source_duration: float,
    max_overlap_ms: int,
    min_overlap_ms: int,
    neighbor_ratio: float,
) -> list[dict[str, Any]]:
    """Build the audio extension plan on top of already-padded video ranges.

    Returns per-range dicts with video_start/end (= padded), audio_start/end,
    delay_ms (placement on the output timeline), and the head/tail extension
    deltas for debugging.
    """
    n = len(padded)
    if n == 0:
        return []

    max_s = max_overlap_ms / 1000.0
    min_s = min_overlap_ms / 1000.0

    # Step 1: how much silence remains on each side after render padding
    # already consumed pre/post_silence by head_pad/tail_pad_ms.
    remaining_pre = [
        max(0.0, p["pre_silence_ms"] - p["head_pad_ms"]) / 1000.0 for p in padded
    ]
    remaining_post = [
        max(0.0, p["post_silence_ms"] - p["tail_pad_ms"]) / 1000.0 for p in padded
    ]

    desired_back = [min(rp, max_s) for rp in remaining_pre]
    desired_fwd = [min(rp, max_s) for rp in remaining_post]

    # Step 2: neighbor cap by PADDED video duration. The clip we're bleeding
    # into is the padded one — that's what's actually being played at the cut.
    for i in range(n):
        if i == 0:
            desired_back[i] = 0.0
        else:
            prev_dur = padded[i - 1]["padded_end"] - padded[i - 1]["padded_start"]
            desired_back[i] = min(desired_back[i], prev_dur * neighbor_ratio)
        if i == n - 1:
            desired_fwd[i] = 0.0
        else:
            next_dur = padded[i + 1]["padded_end"] - padded[i + 1]["padded_start"]
            desired_fwd[i] = min(desired_fwd[i], next_dur * neighbor_ratio)

    # Step 3: min-overlap gate AFTER neighbor cap so tiny residuals collapse.
    for i in range(n):
        if desired_back[i] < min_s:
            desired_back[i] = 0.0
        if desired_fwd[i] < min_s:
            desired_fwd[i] = 0.0

    # Step 4: tentative audio bounds, anchored on the padded video bounds.
    audio_start = [p["padded_start"] - b for p, b in zip(padded, desired_back)]
    audio_end = [p["padded_end"] + f for p, f in zip(padded, desired_fwd)]

    # Step 5: source-edge clamp on the first/last.
    audio_start[0] = max(0.0, audio_start[0])
    audio_end[-1] = min(source_duration, audio_end[-1])

    # Step 6: collision rule — post wins. Tail of i-1 truncates head of i.
    # Don't allow the forced push to advance past this range's padded video
    # start (that would chop into the speaker's actual onset).
    for i in range(1, n):
        if audio_start[i] < audio_end[i - 1]:
            audio_start[i] = audio_end[i - 1]
            if audio_start[i] > padded[i]["padded_start"]:
                audio_start[i] = padded[i]["padded_start"]

    # Step 7: output-timeline placement. Each clip's video occupies
    # padded_end - padded_start on the output; audio starts at (current t_out)
    # minus how far back we extended.
    out: list[dict[str, Any]] = []
    t_out = 0.0
    for i, p in enumerate(padded):
        back_ext = p["padded_start"] - audio_start[i]
        fwd_ext = audio_end[i] - p["padded_end"]
        delay_s = max(0.0, t_out - back_ext)
        delay_ms = int(round(delay_s * 1000.0))
        out.append({
            "video_start": p["padded_start"],
            "video_end": p["padded_end"],
            "audio_start": audio_start[i],
            "audio_end": audio_end[i],
            "delay_ms": delay_ms,
            "back_ext": back_ext,
            "fwd_ext": fwd_ext,
        })
        t_out += p["padded_end"] - p["padded_start"]
    return out


def _fmt(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _build_filter_complex(plan: list[dict[str, Any]]) -> str:
    n = len(plan)
    if n == 0:
        raise ValueError("no ranges to render")

    parts: list[str] = []

    # Video: trim padded ranges + concat (matches render.py output exactly).
    for i, r in enumerate(plan):
        parts.append(
            f"[0:v]trim={_fmt(r['video_start'])}:{_fmt(r['video_end'])},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
    video_inputs = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{video_inputs}concat=n={n}:v=1:a=0[vout]")

    # Audio: aresample once for .mov compat (same fix as render.py), then
    # asplit to feed N parallel atrim branches.
    split_outs = "".join(f"[s{i}]" for i in range(n))
    parts.append(f"[0:a]aresample=async=1:first_pts=0,asplit={n}{split_outs}")

    # Each branch: atrim the extended audio range, reset PTS, delay to the
    # branch's output-timeline position.
    for i, r in enumerate(plan):
        parts.append(
            f"[s{i}]atrim={_fmt(r['audio_start'])}:{_fmt(r['audio_end'])},"
            f"asetpts=PTS-STARTPTS,"
            f"adelay={r['delay_ms']}:all=1[a{i}]"
        )

    # Mix all branches — normalize=0 keeps levels (no per-stream attenuation),
    # dropout_transition=0 prevents fade artifacts at silent gaps.
    audio_inputs = "".join(f"[a{i}]" for i in range(n))
    parts.append(
        f"{audio_inputs}amix=inputs={n}:normalize=0:dropout_transition=0[aout]"
    )

    return ";".join(parts)


def compute_audio_ranges(
    video_path: str | Path,
    clip_json: dict[str, Any],
    *,
    max_overlap_ms: int = DEFAULT_MAX_OVERLAP_MS,
    min_overlap_ms: int = DEFAULT_MIN_OVERLAP_MS,
    neighbor_ratio: float = DEFAULT_NEIGHBOR_RATIO,
) -> list[dict[str, float]]:
    """Return the per-clip audio bounds the overlap render WOULD produce.

    Same plan overlap_render() runs internally — exposed so the worker can
    write it back onto clip_json["audio_ranges"] for the editor to draw the
    waveform overhang past each clip's video boundary.

    Each entry: {
        "video_start": float, "video_end": float,     # padded video bounds
        "audio_start": float, "audio_end": float,     # extended audio bounds
        "back_ext": float,    "fwd_ext": float,       # debug: how far we bled
    }

    Returns empty list if clip_json has no valid ranges.
    """
    video_path = Path(video_path)
    flat = _flatten_ranges(clip_json)
    if not flat:
        return []
    padded = _padded_ranges_with_meta(flat)
    source_duration = _probe_duration(video_path)
    plan = _plan_overlap(
        padded,
        source_duration=source_duration,
        max_overlap_ms=max_overlap_ms,
        min_overlap_ms=min_overlap_ms,
        neighbor_ratio=neighbor_ratio,
    )
    return [
        {
            "video_start": p["video_start"],
            "video_end": p["video_end"],
            "audio_start": p["audio_start"],
            "audio_end": p["audio_end"],
            "back_ext": p["back_ext"],
            "fwd_ext": p["fwd_ext"],
        }
        for p in plan
    ]


def overlap_render(
    video_path: str | Path,
    clip_json: dict[str, Any],
    output_path: str | Path,
    *,
    max_overlap_ms: int = DEFAULT_MAX_OVERLAP_MS,
    min_overlap_ms: int = DEFAULT_MIN_OVERLAP_MS,
    neighbor_ratio: float = DEFAULT_NEIGHBOR_RATIO,
) -> None:
    """Render the kept ranges from clip_json into a single MP4 at output_path,
    with audio extended into the surrounding silence on each side of every cut.

    Video timeline matches render.py exactly. Only the audio differs.

    Raises:
        ValueError: clip_json has no valid ranges.
        RuntimeError: ffmpeg or ffprobe invocation failed.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    flat = _flatten_ranges(clip_json)
    if not flat:
        raise ValueError("no ranges to render")

    padded = _padded_ranges_with_meta(flat)
    source_duration = _probe_duration(video_path)
    plan = _plan_overlap(
        padded,
        source_duration=source_duration,
        max_overlap_ms=max_overlap_ms,
        min_overlap_ms=min_overlap_ms,
        neighbor_ratio=neighbor_ratio,
    )
    filtergraph = _build_filter_complex(plan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-filter_complex", filtergraph,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        "-profile:v", "high",
        "-level:v", "4.2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg overlap render timed out after {exc.timeout}s") from exc
    if result.returncode != 0:
        tail = (result.stderr or "")[-1500:]
        raise RuntimeError(
            f"ffmpeg overlap render failed (exit {result.returncode}):\n{tail}"
        )
