import hashlib
import hmac
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from sentinel.config import WebhookConfig

log = logging.getLogger(__name__)


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """HMAC-SHA256 over "<timestamp>.<body>". Receivers recompute it and reject stale timestamps."""
    digest = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class WebhookNotifier:
    """Delivers in background threads so a slow receiver never delays moderation."""

    def __init__(self, config: WebhookConfig):
        self._config = config
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="webhook")
        self._http = httpx.Client(timeout=config.timeout_seconds)

    def notify(self, url: str | None, secret: str, payload: dict) -> None:
        if url:
            self._pool.submit(self._deliver, url, secret, payload)

    def _deliver(self, url: str, secret: str, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        for attempt in range(1, self._config.max_attempts + 1):
            timestamp = str(int(time.time()))
            headers = {
                "Content-Type": "application/json",
                "X-Sentinel-Timestamp": timestamp,
                "X-Sentinel-Signature": sign(secret, timestamp, body),
            }
            try:
                response = self._http.post(url, content=body, headers=headers)
                if response.status_code < 300:
                    return
                log.warning("webhook %s answered %s (attempt %d)", url, response.status_code, attempt)
            except httpx.HTTPError as exc:
                log.warning("webhook %s failed: %s (attempt %d)", url, exc, attempt)
            if attempt < self._config.max_attempts:
                time.sleep(2 ** (attempt - 1))
        log.error("webhook gave up for message %s", payload.get("message_id"))

    def close(self) -> None:
        self._pool.shutdown(wait=True)
