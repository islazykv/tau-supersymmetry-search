# ---- builder: install dependencies with uv ----
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# ---- runtime: lean image without uv ----
FROM python:3.13-slim

WORKDIR /app

# Copy the self-contained virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code and unified entry point
COPY src/ src/
COPY configs/ configs/
COPY run.py ./

EXPOSE 8000

# Health check for serving mode (uses stdlib — no curl needed in slim image)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Any stage can be run via:
#   docker run tau-supersymmetry-search python run.py stage=train
#   docker run tau-supersymmetry-search python run.py stage=train model=dnn
#   docker run tau-supersymmetry-search python run.py stage=preprocess
#   docker run tau-supersymmetry-search python run.py stage=tune model=dnn tuning.n_trials=100
#   docker run -p 8000:8000 tau-supersymmetry-search python run.py stage=serve
ENTRYPOINT ["python"]
CMD ["run.py", "stage=train"]
