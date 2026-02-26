FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY src/ src/
COPY configs/ configs/
COPY preprocess.py train.py ./

CMD ["uv", "run", "--no-dev", "python", "train.py"]
