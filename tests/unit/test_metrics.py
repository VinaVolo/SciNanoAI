"""Unit tests for InstanceMetrics."""

from __future__ import annotations

import numpy as np
import pytest

from scinanoai.training.cellpose.metrics import InstanceMetrics


@pytest.mark.unit
def test_boxes_iou_identical_returns_one() -> None:
    box = (0, 0, 10, 10)
    assert InstanceMetrics.boxes_iou(box, box) == pytest.approx(1.0)


@pytest.mark.unit
def test_boxes_iou_disjoint_returns_zero() -> None:
    assert InstanceMetrics.boxes_iou((0, 0, 4, 4), (10, 10, 14, 14)) == 0.0


@pytest.mark.unit
def test_mask_iou_complement() -> None:
    a = np.array([[True, True], [False, False]])
    b = np.array([[False, False], [True, True]])
    assert InstanceMetrics.mask_iou(a, b) == 0.0


@pytest.mark.unit
def test_mask_to_instances_extracts_ids() -> None:
    mask = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0]])
    boxes, scores, instances = InstanceMetrics.mask_to_instances(mask)
    assert len(boxes) == 2
    assert len(instances) == 2
    assert all(score == 1.0 for score in scores)
