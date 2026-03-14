FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Copy source code and entry points
COPY src/ src/
COPY configs/ configs/
COPY preprocess.py feature_engineer.py eda.py tune.py \
     train_bdt.py train_dnn.py \
     evaluate_bdt.py evaluate_dnn.py regions.py \
     serve.py ./

EXPOSE 8000

# Any script can be passed as an argument:
#   docker run tau-supersymmetry train_bdt.py
#   docker run tau-supersymmetry train_dnn.py
#   docker run tau-supersymmetry preprocess.py
#   docker run tau-supersymmetry tune.py model=dnn tuning.n_trials=100
#   docker run -p 8000:8000 tau-supersymmetry serve.py --model-type bdt --model-path models/bdt.ubj
ENTRYPOINT ["uv", "run", "--no-dev", "python"]
CMD ["train_bdt.py"]
