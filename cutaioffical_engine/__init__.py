"""cutaioffical-engine — unified pipeline: video -> clean script -> aligned clips.

Public API:
  run_pipeline(video_path)             full pipeline; returns dict with 'clip'
  run_cleanup(video_path)              ffmpeg -> Deepgram -> AI cleanup; no 'clip'
  run_cleanup_from_deepgram(dg)        AI cleanup from cached Deepgram JSON
  align(script, deepgram, video_name)  re-export of Clip's aligner
  render(video, clip_json, output)     Block 4 — ffmpeg trim+concat the cut
  overlap_render(video, clip_json, output)
                                       Block 4 (overlap variant) — same padded
                                       video, audio extended into surrounding
                                       silence (J/L cuts)
  generate_peaks(video, buckets)       audio waveform peaks for the timeline
"""
from __future__ import annotations

from .cleanup import (
    run_pipeline,
    run_cleanup,
    run_cleanup_from_deepgram,
)
from .clip import align
from .render import compute_padded_ranges, render
from .overlap import overlap_render
from .peaks import generate_peaks

__all__ = [
    "run_pipeline",
    "run_cleanup",
    "run_cleanup_from_deepgram",
    "align",
    "render",
    "overlap_render",
    "compute_padded_ranges",
    "generate_peaks",
]
