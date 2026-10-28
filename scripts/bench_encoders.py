"""Compare encoder runtimes (torch vs ONNX fp32 vs ONNX int8) on the test set.

    docker compose --profile tools run --rm tools python scripts/bench_encoders.py

Reports per-batch latency, throughput, process memory growth and how far each runtime's scores drift
from torch. Runs in the tools image because it needs torch.
"""

import gc
import json
import statistics
import time
from pathlib import Path

from sentinel.config import get_config
from sentinel.encoders import Encoder

RUNTIMES = ["torch", "onnx", "onnx-int8"]
REPEATS = 5


def rss_mb() -> float:
    """Current resident memory (Linux)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def main() -> None:
    config = get_config().encoders
    texts = [json.loads(line)["text"] for line in Path("data/test_messages.jsonl").read_text().splitlines()]
    batches = [texts[i : i + 32] for i in range(0, len(texts), 32)]

    lines = [
        "# Encoder benchmark",
        "",
        f"{len(texts)} messages, batches of 32, {config.threads} threads, best of {REPEATS} runs per batch.",
        "",
        "| model | runtime | load s | batch p50 ms | batch max ms | msg/s | RSS +MB | max abs Δscore vs torch |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, spec in config.models.items():
        reference: list[float] | None = None
        for runtime in RUNTIMES:
            before = rss_mb()
            started = time.perf_counter()
            encoder = Encoder(name, spec, runtime, config.models_dir, config.threads)
            load_s = time.perf_counter() - started
            encoder.predict(batches[0][:4])  # warm-up

            timings, scores = [], []
            for batch in batches:
                best = float("inf")
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    probs = encoder.predict(batch)
                    best = min(best, time.perf_counter() - t0)
                timings.append(best * 1000)
                scores += [encoder.score(p) for p in probs]
            if runtime == "torch":
                reference = scores
            drift = max(abs(a - b) for a, b in zip(scores, reference)) if reference else 0.0
            throughput = len(texts) / (sum(timings) / 1000)
            lines.append(
                f"| {name} | {runtime} | {load_s:.1f} | {statistics.median(timings):.0f} | {max(timings):.0f} | "
                f"{throughput:.0f} | {rss_mb() - before:.0f} | {drift:.3f} |"
            )
            print(lines[-1], flush=True)
            del encoder
            gc.collect()

    report = "\n".join(lines) + "\n"
    out = Path("reports/bench-encoders.md")
    out.write_text(report)
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
