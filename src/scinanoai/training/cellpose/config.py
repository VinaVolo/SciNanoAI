from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CellposeTrainingConfig:
    """Configuration container for Cellpose training."""

    train_dir: Path
    test_dir: Path
    image_suffix: str = ".png"
    mask_suffix: str = "_masks.tif"
    output_dir: Path = Path("models/cellpose")
    model_name: str = "cellpose_model"
    pretrained_model: Optional[str] = None

    n_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    nimg_per_epoch: Optional[int] = None
    nimg_test_per_epoch: Optional[int] = None

    log_every_seconds: int = 300
    save_every: int = 100
    save_each: bool = False

    channel_axis: Optional[int] = None
    normalize: bool | dict = True
    min_train_masks: int = 0
    max_train_images: Optional[int] = None
    max_test_images: Optional[int] = None
    class_weights: Optional[Sequence[float]] = None

    scale_range: Optional[float] = None
    rescale: bool = False
    compute_flows: bool = False
    bsize: int = 256

    freeze_encoder: bool = True
    random_seed: int = 42

    extra_args: dict = field(default_factory=dict)

    @property
    def model_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name

    @property
    def logging_dir(self) -> Path:
        return self.model_dir / "logging"

    @property
    def view_dir(self) -> Path:
        return self.model_dir / "view"

    def ensure_output_dirs(self) -> None:
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.view_dir.mkdir(parents=True, exist_ok=True)
