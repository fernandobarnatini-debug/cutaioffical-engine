# cutaioffical-engine

Unified Python package merging the **CleanUp** (video → clean script) and **Clip** (clean script → aligned source timestamps) pipelines into one importable engine.

Designed to be `pip install -e`'d by a FastAPI worker (separate repo) and called as a pure library. Also ships a CLI so the original local-iteration workflow keeps working.

---

## Pipeline

```
 video  ──ffmpeg──▶  16kHz mono wav  ──Deepgram nova-3──▶  word-level transcript
                                                                  │
                                                                  ▼
                                          annotated (pause markers inserted)
                                                                  │
                                                                  ▼
                          OpenRouter Sonnet 4.5 (subtractive editor prompt)
                                                                  │
                                                                  ▼
                          script  =  { kept_spans, removed_segments, notes }
                                                                  │
                                                                  ▼
                                       Clip aligner (token recursion)
                                                                  │
                                                                  ▼
                clip  =  { segments: [ { text, ranges: [ start/end + silence ] } ], notes }
```

The prompt, schema, Deepgram params, model params, and align algorithm are **locked** — do not modify. See `cutaioffical_engine/prompts.py` and the docstrings in `cleanup.py` / `clip.py`.

---

## Install

```bash
pip install -e .
```

Or, from a sibling repo:

```bash
pip install -e ../cutaioffical-engine
```

Requires `ffmpeg` on PATH for the video → wav step.

Set `DEEPGRAM_API_KEY` and `OPENROUTER_API_KEY` in your environment or a `.env` file (see `.env.example`).

---

## Public API

```python
from cutaioffical_engine import (
    run_pipeline,
    run_cleanup,
    run_cleanup_from_deepgram,
    align,
)

# Full pipeline (video → clip alignment)
result = run_pipeline("IMG_2707.MOV")
result["deepgram"]      # full Deepgram response dict
result["transcript"]    # raw text
result["annotated"]     # text with ⟨pause N.Ns⟩ markers
result["script"]        # { kept_spans, removed_segments, notes }
result["script_text"]   # " ".join(script["kept_spans"])
result["clip"]          # { video, duration, segments, notes }

# Cleanup only (no alignment)
result = run_cleanup("IMG_2707.MOV")

# Cleanup from a cached Deepgram response (no ffmpeg, no Deepgram call)
import json
dg = json.load(open("IMG_2707_deepgram.json"))
result = run_cleanup_from_deepgram(dg)

# Re-export of Clip's aligner
clip = align(script_dict, deepgram_dict, video_name="IMG_2707")
```

All functions are pure: no file I/O, no stdout printing, raise on failure.

---

## CLI

```bash
python -m cutaioffical_engine <input> [--outdir PATH] [--transcript-only]
```

`<input>` is either a video file or a Deepgram-format `.json` (skips ffmpeg + Deepgram and starts from the cached transcript).

Outputs (in `--outdir`, default cwd):

| File | When |
|---|---|
| `<stem>_deepgram.json`  | video input only |
| `<stem>_transcript.txt` | always |
| `<stem>_annotated.txt`  | not `--transcript-only` |
| `<stem>_script.json`    | not `--transcript-only` |
| `<stem>_script.txt`     | not `--transcript-only` |
| `<stem>_clip.json`      | not `--transcript-only` |

A top-level `cli.py` shim is provided so `python cli.py <input>` works without installing.

---

## Tests

Pure-stdlib scripts (no pytest required):

```bash
python tests/test_clip.py            # invariant tests against bundled fixtures
python tests/test_cleanup_module.py  # offline smoke test of run_cleanup_from_deepgram
```

Fixtures live under `tests/fixtures/` as `<name>_script.json` + `<name>_deepgram.json` pairs.
