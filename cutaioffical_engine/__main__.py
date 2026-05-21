"""`python -m cutaioffical_engine <input> [--outdir PATH] [--transcript-only]`.

Reproduces CleanUp's CLI behavior file-for-file, then additionally writes
<stem>_clip.json from Clip's aligner when the full pipeline runs.

Output files (in --outdir, default cwd):
  <stem>_deepgram.json     only when input is a video (skipped for .json input)
  <stem>_transcript.txt    raw Deepgram transcript
  <stem>_annotated.txt     transcript with ⟨pause⟩ markers (skipped in transcript-only)
  <stem>_script.json       AI cleanup result (skipped in transcript-only)
  <stem>_script.txt        joined kept_spans (skipped in transcript-only)
  <stem>_clip.json         Clip alignment output (skipped in transcript-only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAIError

from .cleanup import (
    MODEL,
    _raw_transcript,
    annotated_transcript,
    call_deepgram,
    extract_audio,
    structure_script,
)
from .clip import align
from .refine import refine_ranges
from .render import render
from openai import OpenAI
from .cleanup import OPENROUTER_BASE_URL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cutaioffical_engine",
        description="Transcribe a video with Deepgram (or load a Deepgram JSON), clean it with AI, and align spans back to timestamps.",
    )
    parser.add_argument("input", type=Path, help="Path to a video file OR a Deepgram-format .json")
    parser.add_argument("--outdir", type=Path, default=Path.cwd(),
                        help="Directory for output files (default: current dir)")
    parser.add_argument("--transcript-only", action="store_true",
                        help="Skip the AI step (and clip alignment)")
    parser.add_argument("--render", action="store_true",
                        help="After alignment, ffmpeg-render the cut to <stem>_cut.mp4 (requires a video input)")
    parser.add_argument("--keep-internal-silence", action="store_true",
                        help="Skip refine.py's in-clip silence splitter entirely. Preserves ALL internal silences (breaths AND dramatic mid-sentence pauses).")
    parser.add_argument("--max-internal-silence", type=float, default=None, metavar="SECONDS",
                        help="Surgical mode: preserve internal silences SHORTER than this threshold (natural breaths), split anything longer. Example: 0.7 keeps breaths up to 700ms but trims dramatic pauses. Overrides --keep-internal-silence when both are set.")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_required = not args.transcript_only
    if openrouter_required and not openrouter_key:
        print("error: OPENROUTER_API_KEY not set (or pass --transcript-only)", file=sys.stderr)
        return 1

    is_json_input = args.input.suffix.lower() == ".json"

    if is_json_input:
        print(f"Loading Deepgram JSON: {args.input.name}")
        try:
            dg_result = json.loads(args.input.read_text())
        except json.JSONDecodeError as e:
            print(f"error: bad JSON: {e}", file=sys.stderr)
            return 1
    else:
        if not deepgram_key:
            print("error: DEEPGRAM_API_KEY not set in env/.env", file=sys.stderr)
            return 1
        with tempfile.TemporaryDirectory(prefix="transcribe-") as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            print(f"Extracting audio: {args.input.name} -> 16kHz mono wav")
            try:
                extract_audio(args.input, wav_path)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"  ok ({wav_path.stat().st_size / 1024.0:.1f} KB)")

            print("Calling Deepgram (nova-3)...")
            try:
                dg_result = call_deepgram(wav_path.read_bytes(), deepgram_key)
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1

    transcript = _raw_transcript(dg_result)

    stem = args.input.stem
    if not is_json_input:
        (args.outdir / f"{stem}_deepgram.json").write_text(json.dumps(dg_result, indent=2, ensure_ascii=False))
    (args.outdir / f"{stem}_transcript.txt").write_text(transcript + "\n")
    print(f"  ok ({len(transcript.split())} words)")

    if args.transcript_only:
        print()
        print(transcript)
        return 0

    if not transcript.strip():
        print("error: empty transcript", file=sys.stderr)
        return 1

    annotated = annotated_transcript(dg_result)
    (args.outdir / f"{stem}_annotated.txt").write_text(annotated + "\n")

    print(f"Calling AI ({MODEL}) to clean script...")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key)
    try:
        script = structure_script(annotated, client)
    except (OpenAIError, json.JSONDecodeError) as e:
        print(f"error: AI step failed: {e}", file=sys.stderr)
        return 1

    final_script = " ".join(script.get("kept_spans", []))
    (args.outdir / f"{stem}_script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False))
    (args.outdir / f"{stem}_script.txt").write_text(final_script + "\n")

    clip_result = align(script, dg_result, video_name=stem)

    # Block 3 refinement: wav2vec2 + CTC forced alignment snaps each range's
    # start/end to sub-50ms-accurate word boundaries. Production runs this in
    # run_pipeline() — without it, renders use Deepgram's loose ±50–100ms
    # word ends and the render padding constants over/under-compensate.
    # Skipped when input is a Deepgram .json (no source audio to align against).
    if not is_json_input:
        if args.max_internal_silence is not None:
            print(f"Refining word boundaries (wav2vec2, split internal silence ≥ {args.max_internal_silence:.2f}s)...")
        elif args.keep_internal_silence:
            print("Refining word boundaries (wav2vec2, internal-silence split DISABLED)...")
        else:
            print("Refining word boundaries (wav2vec2)...")
        try:
            clip_result = refine_ranges(
                clip_result,
                args.input,
                dg_result["results"]["channels"][0]["alternatives"][0]["words"],
                split_internal_silence=not args.keep_internal_silence,
                internal_silence_threshold_s=args.max_internal_silence,
            )
        except (RuntimeError, KeyError) as e:
            print(f"warning: refine step skipped: {e}", file=sys.stderr)

    with open(args.outdir / f"{stem}_clip.json", "w") as f:
        json.dump(clip_result, f, indent=2)

    if args.render:
        if is_json_input:
            print("error: --render needs a video input, not a Deepgram JSON", file=sys.stderr)
            return 1
        cut_path = args.outdir / f"{stem}_cut.mp4"
        print(f"Rendering cut -> {cut_path.name}...")
        try:
            render(args.input, clip_result, cut_path)
        except (RuntimeError, ValueError) as e:
            print(f"error: render failed: {e}", file=sys.stderr)
            return 1
        print(f"  ok ({cut_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print()
    print(final_script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
