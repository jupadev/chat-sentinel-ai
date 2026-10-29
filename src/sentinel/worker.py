import logging
import os
import signal
import socket
import time
from datetime import UTC, datetime

import redis

from sentinel.config import get_config
from sentinel.encoders import EncoderSet
from sentinel.llm import OllamaJudge
from sentinel.models import ChatMessage, Decision
from sentinel.pipeline import Moderator
from sentinel.queue import BatchConsumer
from sentinel.store import Store
from sentinel.webhook import WebhookNotifier

log = logging.getLogger("sentinel.worker")


class Worker:
    def __init__(self) -> None:
        config = get_config()
        self._store = Store(config.database_path)
        self._consumer = BatchConsumer(
            redis.Redis.from_url(config.redis_url, decode_responses=True),
            os.environ.get("SENTINEL_WORKER_NAME", socket.gethostname()),
            config.batch.max_size,
            config.batch.max_wait_seconds,
        )
        self._notifier = WebhookNotifier(config.webhook)

        started = time.perf_counter()
        encoders = EncoderSet(config.encoders)
        log.info("encoders loaded (%s) in %.1fs", config.encoders.runtime, time.perf_counter() - started)
        judge = None
        if config.llm.enabled:
            ollama = OllamaJudge(config.llm)
            try:
                started = time.perf_counter()
                ollama.warm_up()
                log.info("llm %s warm in %.1fs", config.llm.model, time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001 - the worker still runs; doubtful messages stay flagged
                log.warning("llm warm-up failed: %s", exc)
            judge = ollama.judge
        self._moderator = Moderator(config, encoders, judge)
        self._running = True

    def stop(self, *_: object) -> None:
        log.info("stopping after the current batch")
        self._running = False

    def run(self) -> None:
        while self._running:
            batch = self._consumer.next_batch()
            if batch:
                self._process(batch)
        self._notifier.close()

    def _process(self, batch: list[tuple[str, str]]) -> None:
        started = time.perf_counter()
        entry_ids = [entry_id for entry_id, _ in batch]
        messages = [m for m in self._store.messages_by_keys([key for _, key in batch]) if not self._store.has_verdict(m.key)]
        if not messages:
            self._consumer.ack(entry_ids)
            return

        decisions = self._moderator.screen(messages, self._store)
        # Settle the clear cases first; doubtful ones wait for the LLM one by one.
        doubtful = []
        for message, decision in zip(messages, decisions):
            if decision.doubtful:
                doubtful.append((message, decision))
            else:
                self._finish(message, decision)
        for message, decision in doubtful:
            self._finish(message, self._moderator.adjudicate(message, decision, self._store))

        self._consumer.ack(entry_ids)
        elapsed = time.perf_counter() - started
        log.info(
            "batch size=%d llm=%d encoders_ms=%.0f total_ms=%.0f throughput=%.1f msg/s",
            len(messages), len(doubtful), decisions[0].latency_ms["encoders_batch"], elapsed * 1000,
            len(messages) / elapsed,
        )  # fmt: skip

    def _finish(self, message: ChatMessage, decision: Decision) -> None:
        received = self._store.received_at(message.key)
        decision.latency_ms["end_to_end"] = round((datetime.now(UTC) - received).total_seconds() * 1000, 1)
        self._store.save_verdict(message.key, decision)
        log.info(
            "verdict %s %s stage=%s risk=%s category=%s latency=%s",
            message.message_id, decision.verdict, decision.stage, decision.risk_score, decision.category,
            decision.latency_ms,
        )  # fmt: skip
        if decision.verdict in ("flag", "hide"):
            client = self._store.client(message.client_id)
            self._notifier.notify(
                client["webhook_url"],
                client["webhook_secret"],
                {
                    "event": f"message.{'flagged' if decision.verdict == 'flag' else 'hidden'}",
                    "message_id": message.message_id,
                    "conversation_id": message.conversation_id,
                    "verdict": decision.verdict,
                    "category": decision.category,
                    "reason": decision.reason,
                    "risk_score": decision.risk_score,
                    "stage": decision.stage,
                },
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = Worker()
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()
