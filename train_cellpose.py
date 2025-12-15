from __future__ import annotations

import os
from pathlib import Path

from src.train.cellpose_trainer import CellposeTrainer
from src.train.config import CellposeTrainingConfig


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    return None if parsed < 0 else parsed


def build_config() -> CellposeTrainingConfig:
    project_root = Path(__file__).resolve().parent
    default_train = project_root / "data" / "processed" / "bach_annot" / "train_merged"
    default_test = project_root / "data" / "processed" / "bach_annot" / "test_merged"
    output_dir = Path(os.getenv("CELLPOSE_OUTPUT_DIR", project_root / "models" / "cellpose_stream"))

    return CellposeTrainingConfig(
        train_dir=Path(os.getenv("CELLPOSE_TRAIN_DIR", default_train)),
        test_dir=Path(os.getenv("CELLPOSE_TEST_DIR", default_test)),
        image_suffix=os.getenv("CELLPOSE_IMAGE_SUFFIX", ".png"),
        mask_suffix=os.getenv("CELLPOSE_MASK_SUFFIX", "_masks.tif"),
        output_dir=output_dir,
        model_name=os.getenv("CELLPOSE_MODEL_NAME", "cellpose_full_stream_filtered"),
        pretrained_model=os.getenv("CELLPOSE_PRETRAINED_MODEL") or None,
        n_epochs=int(os.getenv("CELLPOSE_EPOCHS", "100")),
        batch_size=int(os.getenv("CELLPOSE_BATCH_SIZE", "8")),
        learning_rate=float(os.getenv("CELLPOSE_LR", "0.0001")),
        weight_decay=float(os.getenv("CELLPOSE_WEIGHT_DECAY", "0.001")),
        nimg_per_epoch=_env_optional_int("CELLPOSE_NIMG_PER_EPOCH", 200),
        nimg_test_per_epoch=_env_optional_int("CELLPOSE_NIMG_TEST_PER_EPOCH", 100),
        log_every_seconds=int(os.getenv("CELLPOSE_LOG_EVERY", "300")),
        max_train_images=_env_optional_int("CELLPOSE_MAX_TRAIN", None),
        max_test_images=_env_optional_int("CELLPOSE_MAX_TEST", None),
        channel_axis=_env_optional_int("CELLPOSE_CHANNEL_AXIS", None),
        normalize=True,
        min_train_masks=0,
        bsize=int(os.getenv("CELLPOSE_CROP_SIZE", "256")),
    )


def main() -> None:
    config = build_config()
    trainer = CellposeTrainer(config)
    result = trainer.train()
    print(f"Model saved to: {result.model_path}")
    print(f"Metrics plot: {config.logging_dir / 'metrics.png'}")


if __name__ == "__main__":
    main()
