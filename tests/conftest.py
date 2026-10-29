from datetime import UTC, datetime, timedelta

import pytest

from sentinel.config import load_config
from sentinel.models import ChatMessage


@pytest.fixture(scope="session")
def config():
    return load_config("config.yaml")


@pytest.fixture
def message():
    def make(text: str, author: str = "a1", minutes: int = 0, key: str | None = None) -> ChatMessage:
        key = key or f"k-{text[:10]}-{minutes}"
        created = datetime(2026, 10, 28, 12, 0, tzinfo=UTC) + timedelta(minutes=minutes)
        return ChatMessage(key, "client", key, "conv", author, text, created)

    return make
