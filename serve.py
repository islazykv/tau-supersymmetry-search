"""CLI entry point for the model inference server.

Usage
-----
uv run python serve.py --model-type bdt --model-path models/bdt.ubj
uv run python serve.py --model-type dnn --model-path models/dnn.pt --class-names background signal
"""

from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the tau-supersymmetry-search inference API."
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["bdt", "dnn"],
        help="Type of model to serve.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained model file (.ubj for BDT, .pt for DNN).",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Human-readable class labels (e.g. background signal).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Defer import so argparse --help is instant
    from src.serving.app import create_app

    app = create_app(
        model_type=args.model_type,
        model_path=args.model_path,
        class_names=args.class_names,
    )
    uvicorn.run(app, host=args.host, port=args.port)
