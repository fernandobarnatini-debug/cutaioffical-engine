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

import os
import subprocess
from pathlib import Path
from typing import Any

# Render style — flips between two padding profiles. Flip and redeploy the
# worker to switch. The "default" profile is the original calibration; the
# "keoni_tight" profile chops right at the syllable end to match the affiliate
# CapCut workflow (zero noticeable dead space between cuts) at the cost of
# occasionally clipping voiced consonants.
#
# "default"     — original padding calibrated for English voiced-consonant
#                 decay (120ms tail / 150ms short-clip tail).
# "keoni_tight" — affiliate-flow padding (40ms tail / 60ms short-clip tail).
RENDER_STYLE = "keoni_tight"

# Padding constants — calibrated against wav2vec2-precise word boundaries.
#   HEAD_TARGET_MS — pre-onset compensation. 0 for sentence-flow because the
#                    surrounding rhythm context masks tight starts.
#   TAIL_TARGET_MS — post-offset compensation for voiced-consonant decay.
#                    120ms = natural English decay; 40ms = chop at syllable end.
#   PAD_FLOOR_MS   — minimum padding; 0 because wav2vec2 boundaries are precise.
#   SHORT_CLIP_*   — wider window for short standalone utterances where there
#                    is no surrounding rhythm to mask tight boundaries.
SHORT_CLIP_DURATION_S = 1.5

# When the wav2vec2 refine pass is skipped (CUTAIOFFICAL_SKIP_REFINE=1), edges
# are only as precise as Deepgram (~±50-100ms). The "keoni_tight" 0/40 padding
# would clip into word starts and chop tails. Use a hybrid profile that's
# tighter than the pre-Block-3 60/120/25 default but loose enough to absorb
# Deepgram's worst case. Same env var that gates refine flips the padding
# here, so a single `fly secrets unset` reverts both behaviors atomically.
_SKIP_REFINE = os.getenv("CUTAIOFFICAL_SKIP_REFINE", "").strip().lower() in (
    "1", "true", "yes", "on"
)

if _SKIP_REFINE:
    # Hybrid padding for the no-wav2vec2 path. 30ms head + 50ms tail covers
    # Deepgram's typical imprecision; 15ms floor prevents the edges of even
    # the shortest sub-ranges from chopping. Wider short-clip window because
    # short standalone beats are the most exposed to boundary imprecision.
    HEAD_TARGET_MS = 30
    TAIL_TARGET_MS = 50
    PAD_FLOOR_MS = 15
    SHORT_CLIP_HEAD_MS = 60
    SHORT_CLIP_TAIL_MS = 80
elif RENDER_STYLE == "keoni_tight":
    # Affiliate-flow: cut at the end of the last syllable. Pairs with the
    # Keoni prompt's tighter span-break threshold to deliver zero noticeable
    # gap between cuts. Risk: voiced consonants ("s", "z", final vowels) may
    # truncate on some words — accepted tradeoff for flow per Keoni.
    HEAD_TARGET_MS = 0
    TAIL_TARGET_MS = 40
    PAD_FLOOR_MS = 0
    SHORT_CLIP_HEAD_MS = 80
    SHORT_CLIP_TAIL_MS = 60
else:
    # Original calibration. 120ms tail = English voiced-consonant decay
    # window. 120/150 short-clip = natural deliberate counting cadence
    # verified by A/B listening test on the Henley count-reveal source.
    HEAD_TARGET_MS = 0
    TAIL_TARGET_MS = 120
    PAD_FLOOR_MS = 0
    SHORT_CLIP_HEAD_MS = 120
    SHORT_CLIP_TAIL_MS = 150


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

        # Short standalone utterances get larger head + tail windows than
        # sentence-flow cuts. wav2vec2 snaps tight to the phoneme core and
        # misses the natural VOT preparation + voiced-consonant resonance
        # decay — only matters for short clips where there's no surrounding
        # rhythm to hide the tightness.
        duration_s = r["end"] - r["start"]
        is_short = duration_s < SHORT_CLIP_DURATION_S
        head_target = SHORT_CLIP_HEAD_MS if is_short else HEAD_TARGET_MS
        tail_target = SHORT_CLIP_TAIL_MS if is_short else TAIL_TARGET_MS

        head_pad = max(PAD_FLOOR_MS, min(head_target, r["pre_silence_ms"]))
        head_pad = max(0.0, min(head_pad, head_hard_cap))
        tail_pad = max(PAD_FLOOR_MS, min(tail_target, r["post_silence_ms"]))
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


def compute_padded_ranges(clip_json: dict[str, Any]) -> list[tuple[float, float]]:
    """Public: return the (start, end) tuples that render() would actually
    feed to ffmpeg, including per-range padding. Frontends can use this to
    compute cut-timeline positions that match the rendered MP4 exactly.

    Returns empty list if clip_json has no valid ranges.
    """
    flat = _flatten_ranges(clip_json)
    if not flat:
        return []
    return _padded_ranges(flat)


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
        # ultrafast gives 40–60% faster encode for ~25% larger files. For a
        # delivery preview that ships through Supabase Storage in seconds either
        # way, the tradeoff is correct. CRF 20 still pins quality.
        "-preset", "ultrafast",
        "-crf", "20",
        # Force a known iOS-Safari-compatible H.264 profile + level. Without
        # these flags libx264 picks profile/level based on input resolution +
        # ultrafast's feature flags — in practice that often lands at High
        # 5.0+ for 1080p iPhone source, which iOS Safari silently refuses to
        # decode in HTML5 <video> (the user sees "tap Play, nothing happens").
        # High @ 4.2 supports up to 1080p60 and is universally playable on
        # every iOS device since iOS 11.
        "-profile:v", "high",
        "-level:v", "4.2",
        # Force 8-bit yuv420p output. iPhone HDR / Dolby Vision sources land as
        # 10-bit (yuv420p10le) → libx264 silently picks the High 10 profile,
        # which no browser can decode in HTML5 <video> (Chrome, Safari, Firefox
        # all reject it with NotSupportedError). High 8-bit is universally
        # playable.
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        # +faststart moves the moov atom (track metadata) to the front of the
        # MP4 so HTML5 <video> can start decoding before the full file has
        # downloaded. Without this, iOS Safari and many mobile browsers
        # silently refuse to play the cut until the whole file is buffered —
        # which on mobile data with a multi-MB cut can mean tens of seconds
        # of "tap Play and nothing happens." Adds a tiny post-encode pass
        # to relocate the atom; trivial cost.
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        # Hard ceiling so a bad input / stuck encoder can't hang the worker.
        # 4 min covers a generous render of a multi-minute source on a
        # shared-cpu Fly machine. Anything longer is pathological.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg render timed out after {exc.timeout}s") from exc
    if result.returncode != 0:
        tail = (result.stderr or "")[-1500:]
        raise RuntimeError(f"ffmpeg render failed (exit {result.returncode}):\n{tail}")
