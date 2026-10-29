#!/usr/bin/env bash
# One-off: download the encoders, export them to ONNX (fp32 + int8) and, if llm.enabled, pull the Ollama model.
# Run it before the first `docker compose up`; it is safe to re-run (skips finished models).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models/ollama data/db reports
# The app containers run as uid 1000; on Linux the bind mounts must be writable by it.
if [ "$(uname)" = "Linux" ]; then chown -R 1000:1000 models data/db reports 2>/dev/null || sudo chown -R 1000:1000 models data/db reports; fi

docker compose --profile tools build tools
docker compose --profile tools run --rm tools python -m sentinel.tools.export_models "$@"

llm=$(docker compose --profile tools run --rm -T tools python -c \
  "from sentinel.config import get_config; c = get_config().llm; print(c.model if c.enabled else '')")
if [ -n "$llm" ]; then
  docker compose --profile llm up -d ollama
  until docker compose exec -T ollama ollama list >/dev/null 2>&1; do sleep 1; done
  docker compose exec -T ollama ollama pull "$llm"
fi
echo "models ready"
