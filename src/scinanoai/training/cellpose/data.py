from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class DatasetSplit:
    images: list[str]
    masks: list[str]


@dataclass(frozen=True)
class DatasetBundle:
    train: DatasetSplit
    test: DatasetSplit
    missing_train: list[str]
    missing_test: list[str]


class MaskDatasetBuilder:
    """Build train/test splits for Cellpose from image+mask folders.

    Expected layout inside each split directory:
    - images: `<stem><image_suffix>` (default: `.png`)
    - masks:  `<stem><mask_suffix>`  (default: `_masks.tif`)

    Masks must be instance-labelled: background=0, each object has a unique integer id (>0).
    """

    def __init__(
        self,
        *,
        train_dir: Path,
        test_dir: Path,
        image_suffix: str = ".png",
        mask_suffix: str = "_masks.tif",
    ) -> None:
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

    def build(
        self,
        *,
        max_train_images: Optional[int] = None,
        max_test_images: Optional[int] = None,
        seed: int = 42,
    ) -> DatasetBundle:
        train_images, train_masks, missing_train = self._collect_pairs(
            self.train_dir,
            max_images=max_train_images,
            seed=seed,
        )
        test_images, test_masks, missing_test = self._collect_pairs(
            self.test_dir,
            max_images=max_test_images,
            seed=seed,
        )
        return DatasetBundle(
            train=DatasetSplit(images=train_images, masks=train_masks),
            test=DatasetSplit(images=test_images, masks=test_masks),
            missing_train=missing_train,
            missing_test=missing_test,
        )

    def _collect_pairs(
        self,
        root: Path,
        *,
        max_images: Optional[int],
        seed: int,
    ) -> tuple[list[str], list[str], list[str]]:
        root = Path(root)
        if not root.exists():
            return [], [], []

        image_files = sorted(self._iter_images(root))
        pairs: list[tuple[str, str]] = []
        missing: list[str] = []
        for image_path in image_files:
            mask_path = image_path.with_name(f"{image_path.stem}{self.mask_suffix}")
            if mask_path.exists():
                pairs.append((str(image_path), str(mask_path)))
            else:
                missing.append(str(image_path))

        if max_images is not None and max_images >= 0 and len(pairs) > max_images:
            rng = random.Random(seed)
            rng.shuffle(pairs)
            pairs = pairs[:max_images]

        images = [p[0] for p in pairs]
        masks = [p[1] for p in pairs]
        return images, masks, missing

    def _iter_images(self, root: Path) -> Iterable[Path]:
        suffix = self.image_suffix
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if suffix and not path.name.endswith(suffix):
                continue
            yield path

