from __future__ import annotations

from pathlib import Path

from src.train.cellpose_trainer import CellposeTrainer, TrainingResult
from src.train.config import CellposeTrainingConfig


def default_config() -> CellposeTrainingConfig:
    """Default configuration stub; adjust paths to your dataset before running."""
    project_root = Path(__file__).resolve().parents[2]
    return CellposeTrainingConfig(
        train_dir=project_root / "data" / "processed" / "bach_annot" / "train_merged",
        test_dir=project_root / "data" / "processed" / "bach_annot" / "test_merged",
        output_dir=project_root / "models" / "cellpose",
        model_name="cellpose_model",
    )


def run_training(config: CellposeTrainingConfig | None = None) -> TrainingResult:
    trainer = CellposeTrainer(config or default_config())
    return trainer.train()


if __name__ == "__main__":
    result = run_training()
    print(f"Model saved to: {result.model_path}")
