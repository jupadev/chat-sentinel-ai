import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Action = Literal["flag", "hide", "ignore"]
Runtime = Literal["torch", "onnx", "onnx-int8"]


class ApiConfig(BaseModel):
    rate_limit_per_minute: int = 600
    max_text_length: int = 4000


class BatchConfig(BaseModel):
    max_size: int = Field(32, ge=1)
    max_wait_seconds: float = Field(2.0, gt=0)


class ContextConfig(BaseModel):
    messages: int = Field(4, ge=0)


class Thresholds(BaseModel):
    llm_min: float = 0.40
    llm_max: float = 0.80
    hide: float = 0.98

    @model_validator(mode="after")
    def ordered(self) -> "Thresholds":
        if not 0 <= self.llm_min <= self.llm_max <= self.hide <= 1:
            raise ValueError("thresholds must satisfy 0 <= llm_min <= llm_max <= hide <= 1")
        return self


class EncoderModel(BaseModel):
    repo: str
    revision: str
    preprocess: Literal["tweet_es", "none"] = "none"
    max_length: int = 128
    activation: Literal["sigmoid", "softmax"]
    score_labels: list[str]
    weight: float = Field(ge=0, le=1)


class EncodersConfig(BaseModel):
    runtime: Runtime = "onnx-int8"
    threads: int = 2
    models_dir: Path = Path("/models/encoders")
    models: dict[str, EncoderModel]


class LlmConfig(BaseModel):
    enabled: bool = True
    base_url: str = "http://ollama:11434"
    model: str
    think: bool = False
    # Completes "You moderate a private chat inside ..." in the system prompt.
    platform: str = "a professional platform"
    timeout_seconds: float = 90
    num_ctx: int = 2048
    num_thread: int = 2
    keep_alive: int | str = -1
    can_hide: bool = False


class WebhookConfig(BaseModel):
    timeout_seconds: float = 5
    max_attempts: int = 3


class RuleConfig(BaseModel):
    name: str
    type: Literal["regex", "wordlist", "phone", "link", "repeated"]
    action: Action
    category: str
    patterns: list[str] = []
    words: list[str] = []
    normalized: bool = False
    allowed_domains: list[str] = []
    min_digits: int = 8
    min_repeats: int = 3
    window_minutes: int = 10
    min_length: int = 12


class Config(BaseModel):
    database_path: Path = Path("/data/sentinel.db")
    redis_url: str = "redis://redis:6379/0"
    api: ApiConfig = ApiConfig()
    batch: BatchConfig = BatchConfig()
    context: ContextConfig = ContextConfig()
    thresholds: Thresholds = Thresholds()
    encoders: EncodersConfig
    llm: LlmConfig
    webhook: WebhookConfig = WebhookConfig()
    rules: list[RuleConfig] = []


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("SENTINEL_CONFIG", "config.yaml"))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Environment overrides for the values that differ between a laptop and the VPS.
    for key, env in (("database_path", "SENTINEL_DATABASE_PATH"), ("redis_url", "SENTINEL_REDIS_URL")):
        if os.environ.get(env):
            data[key] = os.environ[env]
    return Config.model_validate(data)


@lru_cache
def get_config() -> Config:
    return load_config()
