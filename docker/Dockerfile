# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src/forge ./src/forge

# Validate the package builds cleanly before runtime image assembly.
RUN uv sync --frozen --no-dev --no-editable && uv build

FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md KNOWN_GAPS.md ./
COPY src/forge ./src/forge

RUN uv sync --frozen --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["forge"]
CMD ["--help"]
