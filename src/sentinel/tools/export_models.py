"""Download the encoders at their pinned revisions and export them to ONNX (fp32 and dynamic int8).

Runs once, in the tools image (needs the `export` dependency group), on a dev machine: copy the
resulting onnx-int8 folders to the server. Usage: export_models [--force] [model ...]. Layout per model:
    <models_dir>/<name>/torch       original checkpoint (benchmarks only)
    <models_dir>/<name>/onnx        fp32 ONNX
    <models_dir>/<name>/onnx-int8   dynamically quantized ONNX
"""

import shutil
import subprocess
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


def _download_and_export(name: str) -> None:
    config = get_config().encoders
    spec = config.models[name]
    torch_dir = model_dir(config.models_dir, name, "torch")
    onnx_dir = model_dir(config.models_dir, name, "onnx")

    print(f"{name}: downloading {spec.repo}@{spec.revision[:8]}", flush=True)
    patterns = [*_FILES, _weights(spec.repo, spec.revision)]
    snapshot_download(spec.repo, revision=spec.revision, local_dir=torch_dir, allow_patterns=patterns)

    print(f"{name}: exporting to ONNX", flush=True)
    ORTModelForSequenceClassification.from_pretrained(torch_dir, export=True).save_pretrained(onnx_dir)
    AutoTokenizer.from_pretrained(torch_dir).save_pretrained(onnx_dir)


def _quantize(name: str) -> None:
    config = get_config().encoders
    onnx_dir = model_dir(config.models_dir, name, "onnx")
    int8_dir = model_dir(config.models_dir, name, "onnx-int8")

    print(f"{name}: quantizing to int8", flush=True)
    # All supported ops, embeddings included: the XLM-R embedding table (250k rows) alone is ~770 MB in fp32.
    # reduce_range (7-bit weights) avoids the U8S8 saturation of x86 CPUs without VNNI (AVX2/AVX512, both
    # the dev machine and the VPS); without it the toxicity score drifted up to 0.99 from torch.
    # The .onnx file is portable, but this saturation is a runtime property of the CPU.
    int8_dir.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        onnx_dir / "model.onnx",
        int8_dir / "model_quantized.onnx",
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    for f in onnx_dir.iterdir():
        if f.suffix != ".onnx" and f.is_file():
            shutil.copy(f, int8_dir / f.name)


STEPS = {"export": _download_and_export, "quantize": _quantize}


def main(argv: list[str]) -> None:
    if argv[:1] == ["--step"]:
        STEPS[argv[1]](argv[2])
        return

    force = "--force" in argv
    only = [a for a in argv if not a.startswith("--")]
    config = get_config().encoders
    for name in only or list(config.models):
        if not force and (model_dir(config.models_dir, name, "onnx-int8") / "model_quantized.onnx").exists():
            print(f"{name}: already exported, skipping (use --force to redo)")
            continue
        # Each step in its own process: memory held by the torch export is not returned before the
        # quantizer starts, and together they exceed 6 GB on XLM-R base. Apart, each fits in ~4 GB.
        for step in STEPS:
            subprocess.run([sys.executable, "-m", "sentinel.tools.export_models", "--step", step, name], check=True)
        print(f"{name}: done")


if __name__ == "__main__":
    main(sys.argv[1:])
