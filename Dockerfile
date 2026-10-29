FROM python:3.12.13-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home --uid 1000 sentinel
COPY pyproject.toml uv.lock README.md ./

# Runtime image for api and worker: onnxruntime + tokenizers, no torch.
FROM base AS runtime
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
COPY config.yaml ./
COPY scripts ./scripts
COPY data ./data
USER sentinel

# Tools image: model export, torch benchmarks and tests. Never deployed as a service.
FROM base AS tools
RUN uv sync --frozen --group export --group dev --no-install-project
COPY src ./src
RUN uv sync --frozen --group export --group dev
COPY config.yaml ./
COPY scripts ./scripts
COPY data ./data
COPY tests ./tests
USER sentinel
