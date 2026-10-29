import time

import redis

STREAM = "sentinel:messages"
GROUP = "workers"


def enqueue(client: redis.Redis, key: str) -> None:
    client.xadd(STREAM, {"key": key}, maxlen=100_000, approximate=True)


class BatchConsumer:
    """Reads micro-batches from the stream: closes at max_size or max_wait after the first message."""

    def __init__(self, client: redis.Redis, name: str, max_size: int, max_wait: float):
        self._client = client
        self._name = name
        self._max_size = max_size
        self._max_wait = max_wait
        # On start, first re-deliver what this consumer read but never acknowledged (crash recovery).
        self._recovering = True
        try:
            client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _read(self, stream_id: str, count: int, block_ms: int | None) -> list[tuple[str, str]]:
        response = self._client.xreadgroup(GROUP, self._name, {STREAM: stream_id}, count=count, block=block_ms)
        if not response:
            return []
        return [(entry_id, fields["key"]) for entry_id, fields in response[0][1] if fields]

    def next_batch(self, idle_block_ms: int = 5000) -> list[tuple[str, str]]:
        if self._recovering:
            pending = self._read("0", self._max_size, None)
            if pending:
                return pending
            self._recovering = False

        batch = self._read(">", self._max_size, idle_block_ms)
        if not batch:
            return []
        deadline = time.monotonic() + self._max_wait
        while len(batch) < self._max_size:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms < 1:
                break
            batch += self._read(">", self._max_size - len(batch), remaining_ms)
        return batch

    def ack(self, entry_ids: list[str]) -> None:
        if entry_ids:
            self._client.xack(STREAM, GROUP, *entry_ids)
