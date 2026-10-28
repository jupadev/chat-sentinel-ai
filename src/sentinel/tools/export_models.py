"""Download the encoders at their pinned revisions and export them to ONNX (fp32 and dynamic int8).

Runs once, in the tools image (needs the `export` dependency group). Layout per model:
    <models_dir>/<name>/torch       original checkpoint (benchmarks only)
    <models_dir>/<name>/onnx        fp32 ONNX
    <models_dir>/<name>/onnx-int8   dynamically quantized ONNX
"""

import shutil
import sys

from huggingface_hub import list_repo_files, snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from sentinel.config import get_config
from sentinel.encoders import model_dir

_FILES = ["config.json", "tokenizer*", "special_tokens_map.json", "*.model"]


def _weights(repo: str, revision: str) -> str:
    """Prefer safetensors; some repos (robertuito-hate-speech) only ship pytorch_model.bin."""
    files = list_repo_files(repo, revision=revision)
    return "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"


def export(force: bool = False) -> None:
    config = get_config().encoders
    for name, spec in config.models.items():
        torch_dir = model_dir(config.models_dir, name, "torch")
        onnx_dir = model_dir(config.models_dir, name, "onnx")
        int8_dir = model_dir(config.models_dir, name, "onnx-int8")
        if not force and (int8_dir / "model_quantized.onnx").exists():
            print(f"{name}: already exported, skipping (use --force to redo)")
            continue

        print(f"{name}: downloading {spec.repo}@{spec.revision[:8]}")
        patterns = [*_FILES, _weights(spec.repo, spec.revision)]
        snapshot_download(spec.repo, revision=spec.revision, local_dir=torch_dir, allow_patterns=patterns)

        print(f"{name}: exporting to ONNX")
        model = ORTModelForSequenceClassification.from_pretrained(torch_dir, export=True)
        model.save_pretrained(onnx_dir)
        tokenizer = AutoTokenizer.from_pretrained(torch_dir)
        tokenizer.save_pretrained(onnx_dir)

        print(f"{name}: quantizing to int8")
        # Only MatMul weights: that is where the compute is. Leaving the embedding table (250k rows in
        # XLM-R) in fp32 keeps accuracy and keeps the quantizer's peak memory low; optimum's
        # ORTQuantizer was OOM-killed on XLM-R base with ~5.7 GB available.
        int8_dir.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(
            onnx_dir / "model.onnx",
            int8_dir / "model_quantized.onnx",
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
        for f in onnx_dir.iterdir():
            if f.suffix != ".onnx" and f.is_file():
                shutil.copy(f, int8_dir / f.name)
        print(f"{name}: done")


if __name__ == "__main__":
    export(force="--force" in sys.argv)
