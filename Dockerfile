# syntax=docker/dockerfile:1
# Multi-stage build: a fat builder resolves dependencies, a slim runtime ships
# only the virtualenv + source + model artifacts. Keeps the deployed image lean
# and reproducible (locked deps), and means "works on my machine" == CI == prod.

# ---------- builder ----------
FROM python:3.12-slim AS builder

# uv for fast, locked installs (pin matches the version that wrote uv.lock).
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Install only the groups the serving runtime needs (no dev/mlops/cloud) so the
# image stays small. Layer 1: deps only (cached unless lockfile changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-default-groups \
    --group ml --group text --group serving

# Layer 2: the project itself (README is referenced by pyproject for the build).
COPY README.md ./
COPY src ./src
COPY configs ./configs
COPY params.yaml ./
RUN uv sync --frozen --no-default-groups --group ml --group text --group serving

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for the running service.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
COPY params.yaml ./
# Model artifacts (DVC-tracked). Materialise them before building:
#   uv run dvc pull        (or train locally) so artifacts/ exists in the context.
COPY artifacts ./artifacts

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "threat_detection.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
