"""Cellpose-driven image analysis service (decoupled from ChatBot)."""

from __future__ import annotations

import base64
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..schemas import ImagePayload

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageDescription:
    index: int
    text: str


@dataclass(frozen=True)
class ImageMetric:
    image_name: str
    classification: str
    avg_area: float
    avg_radius: float
    segments: int


class ImageAnalyzer:
    """Encapsulates classification, segmentation and outlier cleaning."""

    def __init__(
        self,
        *,
        cellpose_model_path: Path | None,
        image_um_per_px: float,
    ) -> None:
        self._um_per_px = image_um_per_px
        self._cellpose_model = self._maybe_load_cellpose(cellpose_model_path)

    # -- loading ------------------------------------------------------------
    @staticmethod
    def _maybe_load_cellpose(path: Path | None):
        if path is None or not path.exists():
            _LOG.warning("Cellpose model not available; using stub segmentation.")
            return None
        try:
            import torch
            from cellpose import models as cellpose_models

            use_gpu = torch.cuda.is_available()
            model = cellpose_models.CellposeModel(gpu=use_gpu, pretrained_model=str(path))
            device = torch.cuda.get_device_name(0) if use_gpu else "CPU"
            _LOG.info("Cellpose model loaded from %s on %s", path, device)
            return model
        except Exception as exc:
            _LOG.error("Failed to load Cellpose model: %s", exc)
            return None

    # -- public API ---------------------------------------------------------
    def describe(self, images: Iterable[ImagePayload]) -> list[ImageDescription]:
        descriptions: list[ImageDescription] = []
        for idx, payload in enumerate(images, start=1):
            try:
                raw = base64.b64decode(payload.data)
                with Image.open(BytesIO(raw)) as img:
                    kb_size = len(raw) / 1024
                    descriptions.append(
                        ImageDescription(
                            index=idx,
                            text=(
                                f"Изображение {idx}: формат {img.format}, "
                                f"размер {img.width}x{img.height} px, "
                                f"~{kb_size:.1f} КБ."
                            ),
                        )
                    )
            except Exception as exc:
                _LOG.error("Failed to describe image %d: %s", idx, exc)
                descriptions.append(
                    ImageDescription(
                        index=idx, text=f"Изображение {idx}: ошибка обработки ({exc})."
                    )
                )
        return descriptions

    def analyze(self, images: list[ImagePayload]) -> tuple[str, list[ImageMetric]]:
        records: list[dict] = []
        errors: list[str] = []
        for idx, payload in enumerate(images, start=1):
            try:
                raw = base64.b64decode(payload.data)
                name = payload.name or f"image_{idx}.png"
                with Image.open(BytesIO(raw)) as img:
                    classification = self._classify_color(img)
                    segments = self._segment(img)
                if not segments:
                    records.append(self._record(idx, name, classification, area=0.0, radius=0.0))
                else:
                    for seg in segments:
                        records.append(self._record(idx, name, classification, **seg))
            except Exception as exc:
                msg = f"Изображение {idx}: не удалось обработать ({exc})."
                errors.append(msg)
                _LOG.error(msg)

        if not records:
            return "\n".join(errors), []

        df = pd.DataFrame(records)
        df = self._remove_outliers(df, "radius")
        df = self._remove_outliers(df, "area")

        metrics: list[ImageMetric] = []
        summary_lines: list[str] = list(errors)
        for name, group in df.groupby("image_name"):
            avg_area = float(group["area"].mean())
            avg_radius = float(group["radius"].mean())
            classification = group["classification"].iloc[0]
            metrics.append(
                ImageMetric(
                    image_name=name,
                    classification=classification,
                    avg_area=avg_area,
                    avg_radius=avg_radius,
                    segments=len(group),
                )
            )
            summary_lines.append(
                f"{name}: класс={classification}, средняя площадь={avg_area:.2f} мкм^2, "
                f"средний радиус={avg_radius:.2f} мкм, сегментов после очистки={len(group)}."
            )
        return "\n".join(summary_lines), metrics

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _record(idx: int, name: str, classification: str, *, area: float, radius: float) -> dict:
        return {
            "image_index": idx,
            "image_name": name,
            "material_name": name,
            "classification": classification,
            "area": area,
            "radius": radius,
        }

    @staticmethod
    def _classify_color(img: Image.Image, threshold: int = 40) -> str:
        arr = np.array(img.convert("RGB")).astype(float)
        R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        brightness = R + G + B
        mask = brightness > threshold
        if not np.any(mask):
            return "нет ярких пикселей"
        yellow_score = float(np.mean((R[mask] + G[mask]) - B[mask]))
        blue_score = float(np.mean(B[mask] - (R[mask] + G[mask])))
        return "цитоплазма (желтый)" if yellow_score > blue_score else "ядро (синий)"

    def _segment(self, img: Image.Image, threshold: int = 40) -> list[dict[str, float]]:
        arr = np.array(img.convert("RGB"))
        um_per_px = float(self._um_per_px or 0.0)
        if self._cellpose_model is not None:
            try:
                masks, _flows, _styles = self._cellpose_model.eval(
                    [arr], channels=[0, 0], progress=False
                )
                mask = masks[0] if isinstance(masks, list) else masks
                entries: list[dict[str, float]] = []
                for label in np.unique(mask):
                    if label == 0:
                        continue
                    area_px = float(np.sum(mask == label))
                    radius_px = math.sqrt(area_px / math.pi)
                    entries.append(
                        {
                            "area": area_px * (um_per_px**2) if um_per_px > 0 else area_px,
                            "radius": radius_px * um_per_px if um_per_px > 0 else radius_px,
                        }
                    )
                return entries
            except Exception as exc:
                _LOG.error("Cellpose segmentation failed: %s", exc)

        arr_gray = np.array(img.convert("L")).astype(float)
        mask = arr_gray > threshold
        area_px = float(np.sum(mask))
        if area_px == 0:
            return []
        radius_px = math.sqrt(area_px / math.pi)
        return [
            {
                "area": area_px * (um_per_px**2) if um_per_px > 0 else area_px,
                "radius": radius_px * um_per_px if um_per_px > 0 else radius_px,
            }
        ]

    @staticmethod
    def _remove_outliers(
        df: pd.DataFrame, value_col: str, group_col: str = "material_name"
    ) -> pd.DataFrame:
        cleaned_parts = []
        for _name, group in df.groupby(group_col):
            q1 = group[value_col].quantile(0.25)
            q3 = group[value_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            cleaned_parts.append(group[(group[value_col] >= lower) & (group[value_col] <= upper)])
        return pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else df
