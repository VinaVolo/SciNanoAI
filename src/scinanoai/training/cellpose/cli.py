"""CLI entrypoint for Cellpose training driven by a YAML config."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from ...utils.logging import setup_logging
from ...utils.paths import get_project_root
from .config import CellposeTrainingConfig
from .trainer import CellposeTrainer

_LOG = setup_logging("scinanoai.train.cellpose")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_path(value, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _build_config(payload: dict, root: Path) -> CellposeTrainingConfig:
    payload = dict(payload)
    payload["train_dir"] = _resolve_path(payload["train_dir"], root)
    payload["test_dir"] = _resolve_path(payload["test_dir"], root)
    if "output_dir" in payload:
        payload["output_dir"] = _resolve_path(payload["output_dir"], root)
    return CellposeTrainingConfig(**payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a Cellpose model")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args(argv)

    root = get_project_root()
    payload = _load_yaml(args.config)
    config = _build_config(payload, root)
    config.ensure_output_dirs()

    _LOG.info("Starting training. output_dir=%s", config.model_dir)
    trainer = CellposeTrainer(config)
    result = trainer.train()

    manifest = {
        "model_path": str(result.model_path),
        "config_path": str(args.config),
        "metrics_keys": list(result.metrics_history.keys()),
    }
    manifest_path = config.logging_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _LOG.info("Run manifest written: %s", manifest_path)


if __name__ == "__main__":
    main()
