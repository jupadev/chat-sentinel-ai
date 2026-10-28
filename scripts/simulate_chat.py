"""Send chat messages to the API at a configurable rate, then wait for every verdict.

    SENTINEL_API_KEY=sk_... python scripts/simulate_chat.py --url http://api:8000 --rate 5 --count 200

Messages are drawn (cyclically) from data/test_messages.jsonl with fresh ids, so the run measures the
real queue -> worker -> verdict path, including queue wait.
"""

import argparse
import itertools
import json
import os
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("SENTINEL_URL", "http://api:8000"))
    parser.add_argument("--api-key", default=os.environ.get("SENTINEL_API_KEY"))
    parser.add_argument("--rate", type=float, default=5.0, help="messages per second")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--data", default="data/test_messages.jsonl")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for all verdicts")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("set SENTINEL_API_KEY or pass --api-key")

    samples = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    run = uuid.uuid4().hex[:6]
    client = httpx.Client(base_url=args.url, headers={"Authorization": f"Bearer {args.api_key}"}, timeout=10)

    sent: dict[str, float] = {}
    started = time.monotonic()
    for i, sample in enumerate(itertools.islice(itertools.cycle(samples), args.count)):
        message_id = f"sim-{run}-{i:05d}"
        response = client.post(
            "/messages",
            json={
                "message_id": message_id,
                "conversation_id": f"sim-{run}-{sample['conversation_id']}",
                "author_id": sample["author_id"],
                "text": sample["text"],
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        response.raise_for_status()
        sent[message_id] = time.monotonic()
        # Pace by schedule, not by sleep-after-send, so slow responses do not lower the rate.
        time.sleep(max(0.0, started + (i + 1) / args.rate - time.monotonic()))
    send_s = time.monotonic() - started
    print(f"sent {len(sent)} messages in {send_s:.1f}s ({len(sent) / send_s:.1f} msg/s)")

    latencies: list[float] = []
    verdicts: dict[str, int] = {}
    pending = set(sent)
    deadline = time.monotonic() + args.timeout
    while pending and time.monotonic() < deadline:
        for message_id in list(pending):
            body = client.get(f"/moderation/{message_id}").json()
            if body["status"] == "done":
                pending.discard(message_id)
                latencies.append(body["latency_ms"]["end_to_end"])
                verdicts[body["verdict"]] = verdicts.get(body["verdict"], 0) + 1
        if pending:
            time.sleep(1)

    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        print(f"verdicts: {verdicts}")
        print(f"end-to-end latency ms: p50 {statistics.median(ordered):.0f} · p95 {p95:.0f} · max {ordered[-1]:.0f}")
    if pending:
        print(f"{len(pending)} messages still pending after {args.timeout:.0f}s")


if __name__ == "__main__":
    main()
