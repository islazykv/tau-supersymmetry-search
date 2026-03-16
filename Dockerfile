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

# Copy source code and entry points
COPY src/ src/
COPY configs/ configs/
COPY preprocess.py feature_engineer.py eda.py tune.py \
     train_bdt.py train_dnn.py \
     evaluate_bdt.py evaluate_dnn.py regions.py \
     serve.py ./

EXPOSE 8000

# Health check for serving mode (uses stdlib — no curl needed in slim image)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Any script can be passed as an argument:
#   docker run tau-supersymmetry train_bdt.py
#   docker run tau-supersymmetry train_dnn.py
#   docker run tau-supersymmetry preprocess.py
#   docker run tau-supersymmetry tune.py model=dnn tuning.n_trials=100
#   docker run -p 8000:8000 tau-supersymmetry serve.py --model-type bdt --model-path models/bdt.ubj
ENTRYPOINT ["python"]
CMD ["train_bdt.py"]
