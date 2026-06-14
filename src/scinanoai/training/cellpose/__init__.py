"""Cellpose training pipeline.

Light re-exports only. The trainer is intentionally NOT re-exported here
because importing it pulls in matplotlib / cellpose / torch — heavy
optional deps that should only be paid for when the ``ml`` extra is
installed. Consumers needing the trainer should import it directly:

    from scinanoai.training.cellpose.trainer import CellposeTrainer
"""

from .config import CellposeTrainingConfig
from .metrics import InstanceMetrics

__all__ = ["CellposeTrainingConfig", "InstanceMetrics"]
