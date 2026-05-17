"""Invariant tests for Clip (ported from ~/Clip/test_clip.py).

Runs align() against every fixture pair under tests/fixtures/ and checks the
seven invariants from the plan. Prints a per-fixture summary and a final
pass/fail.
"""
from __future__ import annotations

import glob
import json
import os
import sys

# Make the repo root importable when run as `python tests/test_clip.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutaioffical_engine.clip import (
    align,
    derive_video_name,
    tokenize,
    normalize_token,
    extract_dg_words,
    MIN_CHUNK_TOKENS,
)


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def discover_fixtures() -> list[tuple[str, str]]:
    pairs = []
    for script_path in sorted(glob.glob(os.path.join(FIXTURE_DIR, "*_script.json"))):
        base = script_path[: -len("_script.json")]
        dg_path = base + "_deepgram.json"
        if os.path.exists(dg_path):
            pairs.append((script_path, dg_path))
    return pairs


def check_invariants(script: dict, deepgram: dict, result: dict) -> list[str]:
    """Returns a list of failure messages (empty list = all invariants pass)."""
    fails = []
    kept_spans = script.get("kept_spans", [])
    dg_words = extract_dg_words(deepgram)
    duration = deepgram.get("metadata", {}).get("duration", float("inf"))

    # Invariant 1: every kept_span produced >=1 range, OR a note explaining why.
    seg_by_index = {s["span_index"]: s for s in result["segments"]}
    note_indices = {n["span_index"] for n in result["notes"]}
    for i in range(len(kept_spans)):
        if i in seg_by_index:
            if not seg_by_index[i]["ranges"]:
                fails.append(f"span {i}: segment present but ranges empty")
        elif i not in note_indices:
            fails.append(f"span {i}: missing from output (no segment, no note)")

    # Invariant 2: no two ranges share any word index.
    seen = {}
    for seg in result["segments"]:
        for r in seg["ranges"]:
            for w in range(r["start_word_idx"], r["end_word_idx"] + 1):
                if w in seen:
                    fails.append(
                        f"word idx {w} appears in span {seen[w]} and span {seg['span_index']}"
                    )
                else:
                    seen[w] = seg["span_index"]

    # Invariant 3: ranges in monotonically increasing source-time order.
    flat = []
    for seg in result["segments"]:
        for r in seg["ranges"]:
            flat.append((seg["span_index"], r))
    last_end_idx = -1
    for span_idx, r in flat:
        if r["start_word_idx"] <= last_end_idx:
            fails.append(
                f"span {span_idx}: range starts at word {r['start_word_idx']} "
                f"but previous ended at {last_end_idx}"
            )
        last_end_idx = max(last_end_idx, r["end_word_idx"])

    # Invariant 4: contiguous matches are text-equivalent to the span.
    for seg in result["segments"]:
        if len(seg["ranges"]) != 1:
            continue
        r = seg["ranges"][0]
        source_tokens = [
            normalize_token(dg_words[i]["word"])
            for i in range(r["start_word_idx"], r["end_word_idx"] + 1)
        ]
        source_tokens = [t for t in source_tokens if t]
        span_tokens = tokenize(seg["text"])
        if source_tokens != span_tokens:
            if len(source_tokens) == len(span_tokens):
                fails.append(
                    f"span {seg['span_index']}: contiguous match text mismatch"
                )

    # Invariant 5: timestamps in valid bounds.
    for seg in result["segments"]:
        for r in seg["ranges"]:
            if not (0 <= r["start"] < r["end"] <= duration + 0.5):
                fails.append(
                    f"span {seg['span_index']}: bad timestamps "
                    f"{r['start']}-{r['end']} (duration {duration})"
                )

    # Invariant 6: pre/post silence non-negative.
    for seg in result["segments"]:
        for r in seg["ranges"]:
            if r.get("pre_silence_ms", 0) < 0:
                fails.append(
                    f"span {seg['span_index']}: negative pre_silence_ms {r['pre_silence_ms']}"
                )
            if r.get("post_silence_ms", 0) < 0:
                fails.append(
                    f"span {seg['span_index']}: negative post_silence_ms {r['post_silence_ms']}"
                )

    # Invariant 7: stitched segments anchor the last range on the LARGEST tail
    # at the LATEST source position.
    src_tokens = [normalize_token(w["word"]) for w in dg_words]
    for seg in result["segments"]:
        if len(seg["ranges"]) < 2:
            continue
        span_tokens = tokenize(seg["text"])
        last_r = seg["ranges"][-1]
        best_n = 0
        best_src_start = -1
        for n in range(len(span_tokens), MIN_CHUNK_TOKENS - 1, -1):
            tail = span_tokens[-n:]
            latest = -1
            for j in range(0, len(src_tokens) - n + 1):
                if src_tokens[j : j + n] == tail:
                    latest = j
            if latest >= 0:
                best_n = n
                best_src_start = latest
                break
        if best_n > 0:
            if last_r["start_word_idx"] != best_src_start:
                fails.append(
                    f"span {seg['span_index']}: last range starts at word "
                    f"{last_r['start_word_idx']} but largest tail anchor (N={best_n}) "
                    f"starts at word {best_src_start}"
                )

    return fails


def main() -> int:
    pairs = discover_fixtures()
    if not pairs:
        print(f"No fixtures found in {FIXTURE_DIR}.")
        return 1

    print(f"Running invariants on {len(pairs)} fixtures.\n")
    print(f"{'fixture':<50} {'spans':>5} {'segs':>5} {'rngs':>5} {'stch':>5} {'note':>5}  status")
    print("-" * 95)

    total_fails = 0
    for script_path, dg_path in pairs:
        with open(script_path) as f:
            script = json.load(f)
        with open(dg_path) as f:
            deepgram = json.load(f)

        name = derive_video_name(script_path)
        try:
            result = align(script, deepgram, video_name=name)
        except Exception as e:  # pragma: no cover
            print(f"{name:<50} ERROR: {e}")
            total_fails += 1
            continue

        fails = check_invariants(script, deepgram, result)
        n_spans = len(script.get("kept_spans", []))
        n_segs = len(result["segments"])
        n_rngs = sum(len(s["ranges"]) for s in result["segments"])
        n_stitched = sum(1 for s in result["segments"] if len(s["ranges"]) > 1)
        n_notes = len(result["notes"])

        status = "PASS" if not fails else f"FAIL ({len(fails)})"
        print(
            f"{name:<50} {n_spans:>5} {n_segs:>5} {n_rngs:>5} {n_stitched:>5} {n_notes:>5}  {status}"
        )
        if fails:
            total_fails += 1
            for f in fails[:5]:
                print(f"    - {f}")
            if len(fails) > 5:
                print(f"    ... and {len(fails) - 5} more")

    print()
    if total_fails:
        print(f"FAIL: {total_fails}/{len(pairs)} fixtures had invariant violations.")
        return 1
    print(f"PASS: all {len(pairs)} fixtures clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
