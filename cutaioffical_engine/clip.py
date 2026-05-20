"""Clip — Block 2 of the cutAI pipeline.

Maps CleanUp's clean text spans back to start/end timestamps in the source
audio by walking Deepgram's word-level timeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


TOKEN_RE = re.compile(r"[^a-z0-9']+")


def normalize_token(s: str) -> str:
    return TOKEN_RE.sub("", s.lower())


def tokenize(text: str) -> list[str]:
    return [t for t in (normalize_token(w) for w in text.split()) if t]


# MIN_CHUNK_TOKENS = 1 admits single-token spans into the render plan. This
# is required for structural-repetition hooks ("One.", "Two.", "Three of
# these...") where Sonnet correctly emits each count item as its own kept
# span; the previous floor of 2 silently dropped them. Validated against
# real prod failure case 95d0a5c0: with 1, 6/6 spans render; with 2, 4/6
# render and the count is gone.
MIN_CHUNK_TOKENS = 1


def find_longest_substring(
    span: list[str],
    src: list[str],
    sp_lo: int,
    sp_hi: int,
    src_lo: int,
    src_hi: int,
) -> tuple[int, int, int]:
    """Return (length, src_start, sp_start) of the longest contiguous match
    between span[sp_lo:sp_hi] and src[src_lo:src_hi]. On length ties, prefer
    the LATER source position (later takes are cleaner). length=0 if none.
    """
    best_len = 0
    best_src = -1
    best_sp = -1
    for sp_start in range(sp_lo, sp_hi):
        for src_start in range(src_lo, src_hi):
            k = 0
            while (
                sp_start + k < sp_hi
                and src_start + k < src_hi
                and span[sp_start + k] == src[src_start + k]
            ):
                k += 1
            if k > best_len or (k > 0 and k == best_len and src_start > best_src):
                best_len, best_src, best_sp = k, src_start, sp_start
    return best_len, best_src, best_sp


def longest_match_start(
    span: list[str], src: list[str], src_lo: int, src_hi: int, min_size: int
) -> int:
    """Source position where span's LONGEST substring match begins. Used to
    derive the upper bound for the previous span — represents where the next
    span's territory actually starts. Filters out short coincidental matches
    (e.g. shared phrases like 'it's not because' between adjacent spans).
    Returns src_hi if no match of >= min_size found.
    """
    length, src_start, _ = find_longest_substring(
        span, src, 0, len(span), src_lo, src_hi
    )
    if length < min_size:
        return src_hi
    return src_start


def find_longest_tail_match(
    span: list[str],
    src: list[str],
    sp_lo: int,
    sp_hi: int,
    src_lo: int,
    src_hi: int,
) -> tuple[int, int]:
    """Find the largest N such that span[sp_hi-N : sp_hi] appears contiguously
    somewhere in src[src_lo : src_hi]. Among multiple source positions, pick
    the LATEST one (max src_start). Returns (length, src_start) where
    sp_start = sp_hi - length. Returns (0, -1) if no tail of length >= 1 matches.
    """
    span_len = sp_hi - sp_lo
    for n in range(span_len, 0, -1):
        sp_start = sp_hi - n
        latest_src = -1
        last_src = src_hi - n
        for src_start in range(src_lo, last_src + 1):
            ok = True
            for k in range(n):
                if span[sp_start + k] != src[src_start + k]:
                    ok = False
                    break
            if ok and src_start > latest_src:
                latest_src = src_start
        if latest_src >= 0:
            return n, latest_src
    return 0, -1


def align_span(
    span: list[str],
    src: list[str],
    sp_lo: int,
    sp_hi: int,
    src_lo: int,
    src_hi: int,
    min_size: int,
) -> list[tuple[int, int, int, int]]:
    """Recursively cover span[sp_lo:sp_hi] within src[src_lo:src_hi].

    Reverse-anchor strategy: find the LARGEST tail of the current span subset
    that appears contiguously in source, anchored at the LATEST source
    position. End-of-sentence rarely fumbles, so anchoring there reliably
    lands on the clean final take. Then recurse on the left half.

    Falls back to longest-substring-anywhere when no tail of length
    >= min_size matches (handles edge cases where the span's tail itself
    can't be found as a clean run).

    Returns list of (src_start, src_end_exclusive, sp_start, sp_end_exclusive).
    """
    if sp_lo >= sp_hi or src_lo >= src_hi:
        return []

    tail_len, tail_src = find_longest_tail_match(
        span, src, sp_lo, sp_hi, src_lo, src_hi
    )
    if tail_len >= min_size:
        sp_start = sp_hi - tail_len
        src_start = tail_src
        src_end = src_start + tail_len
        sp_end = sp_hi
        left = align_span(span, src, sp_lo, sp_start, src_lo, src_start, min_size)
        return left + [(src_start, src_end, sp_start, sp_end)]

    # Fallback: no usable tail match. Try longest-substring-anywhere.
    length, src_start, sp_start = find_longest_substring(
        span, src, sp_lo, sp_hi, src_lo, src_hi
    )
    if length < min_size:
        return []
    src_end = src_start + length
    sp_end = sp_start + length
    left = align_span(span, src, sp_lo, sp_start, src_lo, src_start, min_size)
    right = align_span(span, src, sp_end, sp_hi, src_end, src_hi, min_size)
    return left + [(src_start, src_end, sp_start, sp_end)] + right


def make_range(words: list[dict], s: int, e: int, duration: float | None = None) -> dict:
    """Build a range descriptor. pre_silence_ms = gap from prior source word's
    end to this range's first word's start (0 if no prior word). post_silence_ms
    = gap from this range's last word's end to next source word's start (or to
    the source duration if this range ends on the last source word)."""
    if s > 0:
        pre = (words[s]["start"] - words[s - 1]["end"]) * 1000.0
    else:
        pre = 0.0
    if e + 1 < len(words):
        post = (words[e + 1]["start"] - words[e]["end"]) * 1000.0
    elif duration is not None:
        post = (duration - words[e]["end"]) * 1000.0
    else:
        post = 0.0
    return {
        "start": round(words[s]["start"], 3),
        "end": round(words[e]["end"], 3),
        "start_word_idx": s,
        "end_word_idx": e,
        "pre_silence_ms": round(max(0.0, pre), 1),
        "post_silence_ms": round(max(0.0, post), 1),
    }


def extract_dg_words(deepgram: dict) -> list[dict]:
    return deepgram["results"]["channels"][0]["alternatives"][0]["words"]


def align(script: dict, deepgram: dict, video_name: str = "") -> dict:
    """Align CleanUp's kept_spans to Deepgram word indices.

    Strategy (Plan B): for each kept span, recursively find the longest
    contiguous substring match anywhere in source[cursor:upper]. Emit that as
    one chunk. Recurse on the left half (span tokens before the match, source
    before the match) and the right half. This handles three cases naturally:

    - Full contiguous match → one chunk, longest length wins.
    - Multiple full matches (e.g. retake) → tiebreak prefers the LATER source
      position (cleaner take).
    - Frankenstein span (e.g. CleanUp merged words across takes) → algorithm
      finds the longest clean substring, drops the unmatched prefix/suffix if
      they can't be located cleanly, recurses on the rest.

    Chunks below MIN_CHUNK_TOKENS are dropped to avoid scattered-word junk.
    """
    dg_words = extract_dg_words(deepgram)
    dg_tokens = [normalize_token(w["word"]) for w in dg_words]
    duration = deepgram.get("metadata", {}).get("duration")

    kept_spans: list[str] = script.get("kept_spans", [])
    kept_tokens = [tokenize(s) for s in kept_spans]

    segments: list[dict] = []
    notes: list[dict] = []
    cursor = 0

    for i, span in enumerate(kept_spans):
        tokens = kept_tokens[i]
        if not tokens:
            notes.append({"span_index": i, "kind": "empty", "detail": "span had no comparable tokens"})
            continue

        # Upper bound: the earliest source position where ANY later span has a
        # viable substring match. Prevents the current span from drifting past
        # the next span's territory when multiple full matches exist.
        upper = len(dg_tokens)
        for j in range(i + 1, len(kept_spans)):
            if not kept_tokens[j]:
                continue
            pos = longest_match_start(
                kept_tokens[j], dg_tokens, cursor, len(dg_tokens), MIN_CHUNK_TOKENS
            )
            if pos < len(dg_tokens):
                upper = pos
                break

        chunks = align_span(
            tokens, dg_tokens, 0, len(tokens), cursor, upper, MIN_CHUNK_TOKENS
        )
        if not chunks:
            notes.append(
                {
                    "span_index": i,
                    "kind": "no_match",
                    "detail": f"no chunk of >= {MIN_CHUNK_TOKENS} tokens in source",
                }
            )
            continue

        covered = sum(sp_e - sp_s for _, _, sp_s, sp_e in chunks)
        ranges = [make_range(dg_words, s, e - 1, duration) for s, e, _, _ in chunks]
        segments.append({"span_index": i, "text": span, "ranges": ranges})
        cursor = chunks[-1][1]

        if len(chunks) > 1:
            notes.append(
                {
                    "span_index": i,
                    "kind": "stitched",
                    "detail": f"split into {len(chunks)} chunks, covered {covered}/{len(tokens)} tokens",
                }
            )
        elif covered < len(tokens):
            notes.append(
                {
                    "span_index": i,
                    "kind": "partial",
                    "detail": f"covered {covered}/{len(tokens)} tokens",
                }
            )

    out = {
        "video": video_name,
        "duration": duration,
        "segments": segments,
        "notes": notes,
    }
    return out


def derive_video_name(script_path: str) -> str:
    base = os.path.basename(script_path)
    return base[: -len("_script.json")] if base.endswith("_script.json") else base


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Clip — align CleanUp spans to Deepgram timestamps.")
    parser.add_argument("script_json", help="Path to <name>_script.json from CleanUp")
    parser.add_argument("deepgram_json", help="Path to <name>_deepgram.json from CleanUp")
    parser.add_argument(
        "-o", "--out", help="Output JSON path (default: ~/Clip/out/<name>_clip.json)"
    )
    args = parser.parse_args(argv)

    with open(args.script_json) as f:
        script = json.load(f)
    with open(args.deepgram_json) as f:
        deepgram = json.load(f)

    video_name = derive_video_name(args.script_json)
    result = align(script, deepgram, video_name=video_name)

    if args.out:
        out_path = args.out
    else:
        out_dir = os.path.expanduser("~/Clip/out")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{video_name}_clip.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {out_path}")
    print(
        f"  spans: {len(script.get('kept_spans', []))}  "
        f"segments: {len(result['segments'])}  "
        f"ranges: {sum(len(s['ranges']) for s in result['segments'])}  "
        f"notes: {len(result['notes'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
