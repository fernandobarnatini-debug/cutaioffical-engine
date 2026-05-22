"""Export facebook/wav2vec2-base-960h to ONNX, then dynamic-quantize to INT8.

Produces a single file at cutaioffical_engine/data/wav2vec2_base_960h.int8.onnx
that the engine's refine.py loads via onnxruntime.InferenceSession at runtime.

Two-step process:
  1) torch.onnx.export(Wav2Vec2ForCTC) -> wav2vec2_base_960h.fp32.onnx
     with dynamic axes on both batch (axis 0) and time (axis 1) so any audio
     length works at inference. opset_version=14 covers all ops wav2vec2 uses
     (gelu, layer_norm, attention) and is what HF/optimum default to.
  2) onnxruntime.quantization.quantize_dynamic on the fp32 file → int8 file.
     Dynamic quantization quantizes weights at export time and activations at
     runtime; no calibration data needed. Per-tensor symmetric on weights,
     QUInt8 dtype (the ORT default that matches CPU MatMul kernels best).

Run from the engine repo root:
    python scripts/export_wav2vec2_onnx.py

The FP32 intermediate is left on disk (~360MB) for debugging but isn't shipped;
.gitignore excludes it. Only the INT8 file (~95MB) is committed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_ID = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000

ROOT = Path(__file__).resolve().parent.parent
# Build artifacts land in the engine's data/ during the run, then the final
# INT8 file is copied into the user's cache dir (the same path refine.py
# checks first before falling back to the GitHub-release download). FP32
# intermediates stay under cutaioffical_engine/data/ and are deleted at the
# end of main() — they're never shipped.
OUT_DIR = ROOT / "cutaioffical_engine" / "data"
CACHE_DIR = Path.home() / ".cache" / "cutaioffical_engine"
FP32_PATH = OUT_DIR / "wav2vec2_base_960h.fp32.onnx"
INT8_PATH = OUT_DIR / "wav2vec2_base_960h.int8.onnx"
CACHE_PATH = CACHE_DIR / "wav2vec2_base_960h.int8.onnx"


def export_fp32() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID}...")
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()

    # 2s of audio is plenty for tracing — dynamic axes let runtime accept any T.
    dummy = torch.randn(1, SAMPLE_RATE * 2, dtype=torch.float32)

    # Wrap forward so the exported graph emits logits directly (not a HF
    # ModelOutput namedtuple, which ONNX can't represent).
    class CTCWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values).logits

    wrapper = CTCWrapper(model)

    # Clean any stale external-data artifact from a previous dynamo-path run.
    for p in (FP32_PATH, FP32_PATH.with_suffix(".onnx.data")):
        p.unlink(missing_ok=True)

    print(f"Exporting FP32 ONNX -> {FP32_PATH}")
    # dynamo=False forces the legacy TorchScript-based exporter, which embeds
    # all weights as initializers in a single self-contained file. The newer
    # dynamo path emits external-data + fuses some weights into runtime Mul
    # nodes, which onnxruntime's quantize_dynamic then rejects with
    # "Expected ... to be an initializer".
    torch.onnx.export(
        wrapper,
        (dummy,),
        FP32_PATH.as_posix(),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "samples"},
            "logits": {0: "batch", 1: "frames"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  ok ({FP32_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


def quantize_int8() -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Pre-process: ORT-recommended shape inference + symbolic shape fold pass.
    # Skipping this leaves dynamic shapes that quantize_dynamic can't reason
    # about, and yields a noticeably worse INT8 model.
    pre_path = FP32_PATH.with_suffix(".pre.onnx")
    print(f"Pre-processing (shape inference) -> {pre_path.name}")
    # skip_symbolic_shape=True: the symbolic-shape inferencer chokes on the
    # complex einsum/conv graph wav2vec2 produces, but ONNX's plain shape
    # inferencer is enough for the quantizer to find MatMul tensors.
    quant_pre_process(
        input_model_path=FP32_PATH.as_posix(),
        output_model_path=pre_path.as_posix(),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
    )

    # Quantize MatMul ONLY. wav2vec2 has a Conv1d feature extractor up front
    # that's quantization-sensitive (small filters, narrow dynamic range);
    # quantizing those Convs collapses emission cosine similarity from
    # ~0.999 to ~0.96 and pushes alignment error past 20ms. Keeping Convs in
    # FP32 and quantizing only the transformer's MatMul ops gives us the
    # speedup (the transformer dominates FLOPs) without the accuracy loss.
    print(f"Dynamic-quantizing (MatMul only, per-channel) -> {INT8_PATH}")
    # per_channel=True: one scale per output channel instead of one per tensor.
    # Costs a couple MB of weights but tightens emission cosine sim from
    # ~0.998 to ~0.9995 — well past the brief's 0.999 threshold — and trims
    # outlier word deltas (50-60ms tail drops back under 40ms).
    quantize_dynamic(
        model_input=pre_path.as_posix(),
        model_output=INT8_PATH.as_posix(),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
    )
    print(f"  ok ({INT8_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


def smoke_test() -> None:
    """Sanity: run ORT on random audio, confirm shape & finite values."""
    import numpy as np
    import onnxruntime as ort

    print("Smoke test: running ONNX INT8 session on random 2s audio...")
    sess = ort.InferenceSession(INT8_PATH.as_posix(), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, SAMPLE_RATE * 2).astype(np.float32)
    (logits,) = sess.run(None, {"input_values": x})
    print(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
    if not np.isfinite(logits).all():
        sys.exit("ERROR: non-finite values in ONNX logits")
    print("  ok")


def main() -> int:
    export_fp32()
    quantize_int8()
    smoke_test()

    # Reclaim the ~720MB of FP32 intermediates the moment they're no longer
    # needed. The INT8 file is the only artifact that ships; leaving the
    # FP32 graphs on disk has burned us before (ENOSPC mid-validation).
    print("Cleaning up FP32 intermediates...")
    for p in (FP32_PATH, FP32_PATH.with_suffix(".pre.onnx"),
              FP32_PATH.with_suffix(".onnx.data")):
        if p.exists():
            print(f"  rm {p.relative_to(ROOT)} ({p.stat().st_size / 1024 / 1024:.0f} MB)")
            p.unlink()

    # Copy the INT8 file into the cache dir refine.py reads from. Uploading
    # this same file to a GitHub release matching _ONNX_VERSION in refine.py
    # is the public distribution step.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(INT8_PATH, CACHE_PATH)

    print()
    print(f"INT8 model staged at: {INT8_PATH.relative_to(ROOT)}")
    print(f"Copied to cache:      {CACHE_PATH}")
    print(f"Upload to:            github.com/.../releases (tag from refine._ONNX_VERSION)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
