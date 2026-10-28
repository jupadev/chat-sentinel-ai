import json
from pathlib import Path
from typing import Protocol

import numpy as np

from sentinel.config import EncoderModel, EncodersConfig, Runtime
from sentinel.preprocessing import preprocess_tweet_es


class Runner(Protocol):
    def logits(self, texts: list[str]) -> np.ndarray: ...


def model_dir(models_dir: Path, name: str, runtime: Runtime) -> Path:
    return models_dir / name / runtime


class OnnxRunner:
    def __init__(self, path: Path, max_length: int, threads: int):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_files = sorted(path.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx file in {path}; run scripts/download_models.sh")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(onnx_files[0]), options, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}

        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        pad_token = json.loads((path / "tokenizer_config.json").read_text()).get("pad_token", "<pad>")
        if isinstance(pad_token, dict):
            pad_token = pad_token["content"]
        self._tokenizer.enable_truncation(max_length)
        self._tokenizer.enable_padding(pad_id=self._tokenizer.token_to_id(pad_token), pad_token=pad_token)

    def logits(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        return self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]


class TorchRunner:
    """Reference runner for benchmarks. Needs the `export` dependency group (tools image)."""

    def __init__(self, path: Path, max_length: int, threads: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self._max_length = max_length

    def logits(self, texts: list[str]) -> np.ndarray:
        batch = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
        )
        with self._torch.inference_mode():
            return self._model(**batch).logits.numpy()


class Encoder:
    def __init__(self, name: str, spec: EncoderModel, runtime: Runtime, models_dir: Path, threads: int):
        self.name = name
        self.spec = spec
        path = model_dir(models_dir, name, runtime)
        id2label = json.loads((path / "config.json").read_text())["id2label"]
        self.labels = [id2label[str(i)] for i in range(len(id2label))]
        unknown = set(spec.score_labels) - set(self.labels)
        if unknown:
            raise ValueError(f"{name}: score_labels {unknown} not in model labels {self.labels}")
        runner_cls = TorchRunner if runtime == "torch" else OnnxRunner
        self._runner: Runner = runner_cls(path, spec.max_length, threads)

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        if self.spec.preprocess == "tweet_es":
            texts = [preprocess_tweet_es(t) for t in texts]
        logits = self._runner.logits(texts).astype(np.float64)
        if self.spec.activation == "sigmoid":
            probs = 1 / (1 + np.exp(-logits))
        else:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
        return [dict(zip(self.labels, row.tolist())) for row in probs]

    def score(self, probs: dict[str, float]) -> float:
        return max(probs[label] for label in self.spec.score_labels)


class EncoderSet:
    def __init__(self, config: EncodersConfig, runtime: Runtime | None = None):
        runtime = runtime or config.runtime
        self.encoders = [
            Encoder(name, spec, runtime, config.models_dir, config.threads) for name, spec in config.models.items()
        ]

    def score_batch(self, texts: list[str]) -> list[tuple[float, dict[str, float]]]:
        """Return (risk_score, per-model scores) per text."""
        per_model = {e.name: [e.score(p) for p in e.predict(texts)] for e in self.encoders} if texts else {}
        results = []
        for i in range(len(texts)):
            scores = {name: round(values[i], 4) for name, values in per_model.items()}
            results.append((combine(scores, {e.name: e.spec.weight for e in self.encoders}), scores))
        return results


def combine(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Noisy-OR: one strong signal is enough, several weak ones add up."""
    keep = 1.0
    for name, score in scores.items():
        keep *= 1 - weights[name] * score
    return round(1 - keep, 4)
