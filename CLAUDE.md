# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Near-realtime, CPU-only, fully local moderation for short Rioplatense Spanish chat messages. Moderation
never blocks sending: clients `POST /messages`, a worker moderates asynchronously, and `flag`/`hide`
verdicts are delivered via HMAC-signed webhook. README.md is the source of truth for the API, deploy
steps and the VPS resource budget.

## Commands

Everything runs in Linux containers. `pyproject.toml` restricts uv to `sys_platform == 'linux'`, so
`uv sync` on macOS is not the intended workflow. Use Docker.

```bash
docker compose up -d --build                                   # api + worker + redis
docker compose --profile llm up -d ollama                      # only when llm.enabled: true
docker compose --profile tools run --rm tools pytest           # all tests
docker compose --profile tools run --rm tools pytest tests/test_pipeline.py::test_decide_by_risk_band
docker compose run --rm worker python scripts/eval.py [--no-llm] [--llm-model <tag>]   # -> reports/
docker compose --profile tools run --rm tools python scripts/bench_encoders.py         # torch vs onnx vs int8
docker compose exec api python -m sentinel.cli create-client|create-key|list-keys|revoke-key ...
./scripts/download_models.sh                                   # export encoders to ONNX (fp32 + int8)
```

Tests need only `config.yaml`, fakeredis and in-memory SQLite (`Store(":memory:")`), not models.
`pythonpath = ["src"]` is set in pytest config. No linter is configured.

## Architecture

Request flow: `api.py` (FastAPI, `create_app` factory with injectable `config`/`store`/`queue` for tests)
stores the message in SQLite, then `queue.enqueue` pushes only the message key onto the Redis stream
`sentinel:messages`. `worker.py` reads micro-batches via `BatchConsumer` (closes at `batch.max_size` or
`batch.max_wait_seconds` after the first message; re-delivers its own unacked entries on startup) and
acks only after every message in the batch has a verdict.

Moderation (`pipeline.py`) is split into two phases on purpose:
- `Moderator.screen` runs over the whole batch: `RuleEngine` (`rules.py`), then `EncoderSet.score_batch`
  (`encoders.py`, three ONNX classifiers combined with a weighted noisy-OR into `risk_score`).
  Messages already hit by a `hide` rule skip the encoders.
- `decide()` is the pure verdict function: any acting rule wins (worst by `SEVERITY`); otherwise risk
  bands from `thresholds` give `ok` / doubtful `flag` / `flag` / `hide`. Rules with `action: ignore`
  are recorded and passed to the LLM as hints but never change the verdict.
- `Moderator.adjudicate` runs one message at a time, only for `doubtful` decisions, after the worker
  has already settled the clear ones. The LLM (`llm.py`, Ollama) is off in the POC
  (`llm.enabled: false`). Any LLM failure keeps the preliminary `flag`, and `hide` from the LLM is
  downgraded to `flag` unless `llm.can_hide`.

Core invariant: **when in doubt, flag, never hide.** Keep it intact when changing `decide`,
`adjudicate` or thresholds.

`store.py` is the only persistence layer (SQLite: clients, API keys stored as SHA-256 only, messages,
verdicts, reviews). It also implements the `History` protocol (`models.py`) that rules and the LLM use
for repeated-message detection and conversation context. Everything is scoped per client: an API key
only sees its own client's messages, and `message_id` is idempotent per client.

## Configuration

`config.yaml` (validated by pydantic in `config.py`) holds thresholds, batching, encoder models with
pinned revisions and weights, LLM settings, webhook retries and all rules. Only `database_path` and
`redis_url` have env overrides (`SENTINEL_DATABASE_PATH`, `SENTINEL_REDIS_URL`); `SENTINEL_CONFIG`
points to the file. Containers mount it read-only, so restart `api` and `worker` after editing.

## Constraints

- Runtime image (`Dockerfile` target `runtime`) has onnxruntime + tokenizers only. torch/transformers/
  optimum live in the `export` dependency group and the `tools` image; don't import them from runtime
  modules (`TorchRunner` is used only for benchmarks).
- Memory/CPU limits in `docker-compose.yml` are sized for a VPS shared with another production stack;
  the README explains the budget. Don't raise them casually.
- `[tool.uv] exclude-newer` is a supply-chain guard (ignores releases younger than two weeks).
