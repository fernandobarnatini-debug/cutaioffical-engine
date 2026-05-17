"""Smoke test for cutaioffical_engine.run_cleanup_from_deepgram.

Loads a known fixture, injects a fake `structure_fn` so the AI step is offline,
and asserts the returned dict has the expected shape and values.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cutaioffical_engine import run_cleanup_from_deepgram


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FIXTURE_NAME = "IMG_2707"

EXPECTED_KEYS = {"deepgram", "transcript", "annotated", "script", "script_text"}


def main() -> int:
    dg_path = os.path.join(FIXTURE_DIR, f"{FIXTURE_NAME}_deepgram.json")
    script_path = os.path.join(FIXTURE_DIR, f"{FIXTURE_NAME}_script.json")
    with open(dg_path) as f:
        deepgram = json.load(f)
    with open(script_path) as f:
        canned_script = json.load(f)

    captured = {}

    def fake_structure(transcript: str, client) -> dict:
        captured["transcript"] = transcript
        captured["client"] = client
        return canned_script

    result = run_cleanup_from_deepgram(deepgram, structure_fn=fake_structure)

    fails: list[str] = []

    missing = EXPECTED_KEYS - set(result.keys())
    if missing:
        fails.append(f"missing keys: {sorted(missing)}")

    if result.get("deepgram") is not deepgram:
        fails.append("deepgram key should be the same dict passed in")

    if not isinstance(result.get("transcript"), str) or not result["transcript"]:
        fails.append("transcript should be non-empty string")

    if "⟨pause" not in result.get("annotated", "") and "pause" not in result.get("annotated", ""):
        # Most real fixtures have at least one pause; warn but don't fail if not.
        pass

    if result.get("script") is not canned_script:
        fails.append("script key should be exactly the structure_fn return value")

    expected_text = " ".join(canned_script.get("kept_spans", []))
    if result.get("script_text") != expected_text:
        fails.append(
            f"script_text mismatch: {result.get('script_text')!r} != {expected_text!r}"
        )

    if "transcript" not in captured:
        fails.append("structure_fn was not invoked")
    elif captured["transcript"] != result.get("annotated"):
        fails.append("structure_fn received transcript that differs from result.annotated")

    if fails:
        print("FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1

    print(f"PASS: run_cleanup_from_deepgram returned expected shape "
          f"({len(result['transcript'].split())} transcript words, "
          f"{len(result['script_text'].split())} script words).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
