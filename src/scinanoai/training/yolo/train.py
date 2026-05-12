"""YOLO training entrypoint without module-level side effects."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...utils.logging import setup_logging

_LOG = setup_logging("scinanoai.train.yolo")


def train(*, data_yaml: Path, weights: str = "yolov8x-seg.pt", epochs: int = 100) -> None:
    from ultralytics import YOLO  # heavy import deferred

    _LOG.info("Loading YOLO weights=%s", weights)
    model = YOLO(weights)
    model.train(data=str(data_yaml), epochs=epochs)
    model.val(data=str(data_yaml))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO segmentation model")
    parser.add_argument("--data", required=True, type=Path, help="Path to data YAML")
    parser.add_argument("--weights", default="yolov8x-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(argv)
    train(data_yaml=args.data, weights=args.weights, epochs=args.epochs)


if __name__ == "__main__":
    main()
