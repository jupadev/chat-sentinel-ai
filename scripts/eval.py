"""Run the full moderation pipeline over data/test_messages.jsonl and report quality and latency.

    python scripts/eval.py [--runtime onnx-int8] [--no-llm | --llm-model qwen3:4b-instruct-2507-q4_K_M]

Runs in-process (no API, Redis or database), batching like the worker does. The LLM step needs Ollama.
"""

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sentinel.config import get_config
from sentinel.encoders import EncoderSet
from sentinel.llm import OllamaJudge
from sentinel.models import ChatMessage, Decision
from sentinel.pipeline import Moderator

VERDICTS = ["ok", "flag", "hide"]


class MemoryHistory:
    def __init__(self, messages: list[ChatMessage]):
        self._messages = messages

    def recent_texts_by_author(self, message: ChatMessage, since: datetime) -> list[str]:
        return [
            m.text
            for m in self._messages
            if m.author_id == message.author_id and m.key != message.key and since <= m.created_at <= message.created_at
        ]

    def conversation_context(self, message: ChatMessage, limit: int) -> list[ChatMessage]:
        earlier = [
            m
            for m in self._messages
            if m.conversation_id == message.conversation_id and m.created_at < message.created_at
        ]
        return earlier[-limit:] if limit else []


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def load(path: Path) -> tuple[list[ChatMessage], dict[str, dict]]:
    base = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    messages, labels = [], {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        messages.append(
            ChatMessage(row["id"], "eval", row["id"], row["conversation_id"], row["author_id"], row["text"],
                        base + timedelta(minutes=i))
        )  # fmt: skip
        labels[row["id"]] = row
    return messages, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test_messages.jsonl")
    parser.add_argument("--runtime", choices=["torch", "onnx", "onnx-int8"])
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-model", help="override llm.model, to compare LLMs on the same data")
    parser.add_argument("--out")
    args = parser.parse_args()

    config = get_config()
    if args.llm_model:
        config.llm = config.llm.model_copy(update={"model": args.llm_model})
    runtime = args.runtime or config.encoders.runtime
    messages, labels = load(Path(args.data))
    history = MemoryHistory(messages)

    started = time.perf_counter()
    encoders = EncoderSet(config.encoders, runtime)
    load_s = time.perf_counter() - started
    judge = OllamaJudge(config.llm) if config.llm.enabled and not args.no_llm else None
    if judge:
        judge.warm_up()
    moderator = Moderator(config, encoders, judge.judge if judge else None)

    # Warm-up pass so the first batch does not pay for ONNX graph initialization.
    encoders.score_batch([m.text for m in messages[:4]])

    decisions: dict[str, Decision] = {}
    end_to_end: list[float] = []
    encoder_batches: list[float] = []
    llm_calls: list[float] = []
    total_started = time.perf_counter()
    size = config.batch.max_size
    for start in range(0, len(messages), size):
        batch = messages[start : start + size]
        batch_started = time.perf_counter()
        screened = moderator.screen(batch, history)
        screen_ms = (time.perf_counter() - batch_started) * 1000
        encoder_batches.append(screened[0].latency_ms["encoders_batch"])
        llm_elapsed = 0.0
        for message, decision in zip(batch, screened):
            if decision.doubtful:
                decision = moderator.adjudicate(message, decision, history)
                llm_ms = decision.latency_ms.get("llm", 0.0)
                llm_calls.append(llm_ms)
                llm_elapsed += llm_ms
                end_to_end.append(screen_ms + llm_elapsed)
            else:
                end_to_end.append(screen_ms)
            decisions[message.key] = decision
    total_s = time.perf_counter() - total_started

    report = render(args, runtime, config, messages, labels, decisions, load_s, total_s, end_to_end,
                    encoder_batches, llm_calls)  # fmt: skip
    print(report)
    suffix = config.llm.model.replace(":", "_") if judge else "nollm"
    out = Path(args.out or f"reports/eval-{runtime}-{suffix}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nreport written to {out}")


def render(args, runtime, config, messages, labels, decisions, load_s, total_s, end_to_end, encoder_batches,
           llm_calls) -> str:  # fmt: skip
    t = config.thresholds
    expected = {k: labels[k]["expected"] for k in decisions}
    predicted = {k: d.verdict for k, d in decisions.items()}
    lines = [
        f"# Evaluation — {runtime}{' (no LLM)' if args.no_llm else ''}",
        "",
        f"- Messages: {len(messages)} · batch size {config.batch.max_size}",
        f"- Thresholds: llm_min={t.llm_min} llm_max={t.llm_max} hide={t.hide} · LLM: "
        + ("disabled" if args.no_llm else config.llm.model),
        f"- Encoder load: {load_s:.1f}s · total run: {total_s:.1f}s · throughput {len(messages) / total_s:.1f} msg/s",
        "",
        "## Precision / recall per verdict",
        "",
        "| verdict | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for v in VERDICTS:
        tp = sum(1 for k in decisions if predicted[k] == v and expected[k] == v)
        p_den = sum(1 for k in decisions if predicted[k] == v)
        r_den = sum(1 for k in decisions if expected[k] == v)
        p = tp / p_den if p_den else 0.0
        r = tp / r_den if r_den else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        lines.append(f"| {v} | {p:.2f} | {r:.2f} | {f1:.2f} | {r_den} |")

    # "Needs a human" view: flag and hide both reach the review queue.
    tp = sum(1 for k in decisions if predicted[k] != "ok" and expected[k] != "ok")
    fp = sum(1 for k in decisions if predicted[k] != "ok" and expected[k] == "ok")
    fn = sum(1 for k in decisions if predicted[k] == "ok" and expected[k] != "ok")
    lines += [
        "",
        f"**Actionable (flag or hide) vs ok:** precision {tp / (tp + fp) if tp + fp else 0:.2f} · "
        f"recall {tp / (tp + fn) if tp + fn else 0:.2f} · false positives {fp} · false negatives {fn}",
        "",
        "## Confusion matrix (rows = expected, columns = predicted)",
        "",
        "| | " + " | ".join(VERDICTS) + " |",
        "|---|" + "---|" * len(VERDICTS),
    ]
    matrix = Counter((expected[k], predicted[k]) for k in decisions)
    for e in VERDICTS:
        lines.append(f"| **{e}** | " + " | ".join(str(matrix[(e, p)]) for p in VERDICTS) + " |")

    by_category: dict[str, list[bool]] = defaultdict(list)
    for k in decisions:
        by_category[labels[k]["category"]].append(predicted[k] == expected[k])
    lines += ["", "## Accuracy per message category", "", "| category | correct | total |", "|---|---|---|"]
    for category, results in sorted(by_category.items()):
        lines.append(f"| {category} | {sum(results)} | {len(results)} |")

    stages = Counter(d.stage for d in decisions.values())
    lines += ["", "## Where decisions were made", ""]
    lines += [f"- {stage}: {count}" for stage, count in sorted(stages.items())]

    lines += ["", "## Latency (ms)", "", "| stage | p50 | p95 | n |", "|---|---|---|---|"]
    rules = [d.latency_ms["rules"] for d in decisions.values()]
    for name, values in (
        ("rules (per message)", rules),
        (f"encoders (per batch of ≤{config.batch.max_size})", encoder_batches),
        ("llm (per call)", llm_calls),
        ("end to end, excluding queue wait", end_to_end),
    ):
        lines.append(f"| {name} | {percentile(values, 0.5):.1f} | {percentile(values, 0.95):.1f} | {len(values)} |")

    def table(title: str, keys: list[str]) -> list[str]:
        out = ["", f"## {title} ({len(keys)})", ""]
        if not keys:
            return out + ["None."]
        out += ["| id | expected | got | stage | risk | reason | text |", "|---|---|---|---|---|---|---|"]
        for k in keys:
            d = decisions[k]
            text = labels[k]["text"].replace("|", "\\|")
            reason = d.reason.replace("|", "\\|")
            risk = f"{d.risk_score:.2f}" if d.risk_score is not None else "-"
            out.append(f"| {k} | {expected[k]} | {d.verdict} | {d.stage} | {risk} | {reason} | {text} |")
        return out

    lines += table("False positives (expected ok)", [k for k in decisions if expected[k] == "ok" and predicted[k] != "ok"])
    lines += table("False negatives (expected flag/hide, got ok)",
                   [k for k in decisions if expected[k] != "ok" and predicted[k] == "ok"])  # fmt: skip
    lines += table("Severity mismatches (flag vs hide)",
                   [k for k in decisions if "ok" not in (expected[k], predicted[k]) and expected[k] != predicted[k]])  # fmt: skip

    # Threshold sweep over the encoder risk alone, for messages no acting rule decided.
    unruled = [k for k, d in decisions.items() if d.stage != "rules" and d.risk_score is not None]
    lines += [
        "",
        "## Encoder risk sweep (messages not decided by a rule)",
        "",
        "How many would reach the LLM or be flagged at each `llm_min`, and how many harmful ones fall below it",
        "and are never seen by the LLM.",
        "",
        "| llm_min | ok above (LLM load / FP risk) | harmful below (missed) |",
        "|---|---|---|",
    ]
    for cut in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        above_ok = sum(1 for k in unruled if expected[k] == "ok" and decisions[k].risk_score >= cut)
        below_bad = sum(1 for k in unruled if expected[k] != "ok" and decisions[k].risk_score < cut)
        lines.append(f"| {cut:.2f} | {above_ok} | {below_bad} |")

    lines += ["", "## Risk score per message", "", "| id | expected | risk | scores |", "|---|---|---|---|"]
    for k in unruled:
        lines.append(f"| {k} | {expected[k]} | {decisions[k].risk_score:.3f} | {decisions[k].scores} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
