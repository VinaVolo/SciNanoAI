from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


class InstanceMetrics:
    """Utility helpers for detection and segmentation metrics."""

    @staticmethod
    def mask_to_instances(mask: np.ndarray, score_map: np.ndarray | None = None):
        boxes, scores, instances = [], [], []
        if mask is None:
            return boxes, scores, instances
        ids = np.unique(mask)
        ids = ids[ids > 0]
        for mid in ids:
            instance = mask == mid
            if not instance.any():
                continue
            ys, xs = np.where(instance)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            boxes.append([x1, y1, x2, y2])
            instances.append(instance)
            if score_map is not None:
                scores.append(float(score_map[instance].mean()))
            else:
                scores.append(1.0)
        return boxes, scores, instances

    @staticmethod
    def boxes_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
        inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        area_a = (xa2 - xa1 + 1) * (ya2 - ya1 + 1)
        area_b = (xb2 - xb1 + 1) * (yb2 - yb1 + 1)
        union = area_a + area_b - inter_area
        return float(inter_area / union) if union > 0 else 0.0

    @staticmethod
    def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def evaluate_map(
        preds_list: List[List],
        scores_list: List[List[float]],
        gts_list: List[List],
        thresholds: List[float],
        iou_fn: Callable,
    ):
        metrics = {}
        for thr in thresholds:
            matches = []
            total_gts = 0
            for preds, scores, gts in zip(preds_list, scores_list, gts_list):
                total_gts += len(gts)
                matches.extend(InstanceMetrics._collect_matches(preds, scores, gts, thr, iou_fn))
            precision, recall, ap = InstanceMetrics._precision_recall_ap(matches, total_gts)
            metrics[thr] = {"precision": precision, "recall": recall, "ap": ap}
        map50 = metrics[0.5]["ap"]
        map5095 = float(np.mean([m["ap"] for m in metrics.values()]))
        return metrics[0.5]["precision"], metrics[0.5]["recall"], map50, map5095

    @staticmethod
    def _collect_matches(preds, scores, gts, iou_thr: float, iou_fn: Callable):
        if len(preds) == 0:
            return []
        matched = set()
        order = np.argsort(scores)[::-1]
        results = []
        for idx in order:
            best_iou, best_gt = -1.0, None
            for gi, gt in enumerate(gts):
                if gi in matched:
                    continue
                iou_val = iou_fn(preds[idx], gt)
                if iou_val > best_iou:
                    best_iou, best_gt = iou_val, gi
            if best_gt is not None and best_iou >= iou_thr:
                matched.add(best_gt)
                results.append((scores[idx], 1))
            else:
                results.append((scores[idx], 0))
        return results

    @staticmethod
    def _precision_recall_ap(matches: List[Tuple[float, int]], total_gts: int):
        if total_gts == 0 or len(matches) == 0:
            return 0.0, 0.0, 0.0
        scores = np.array([m[0] for m in matches])
        tps = np.array([m[1] for m in matches])
        order = np.argsort(scores)[::-1]
        tps = tps[order]
        fps = 1 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recalls = tp_cum / total_gts
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        ap = float(np.trapz(precisions, recalls))
        return float(precisions[-1]), float(recalls[-1]), ap
