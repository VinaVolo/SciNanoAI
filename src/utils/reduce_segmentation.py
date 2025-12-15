import json

def thin_coco_segmentations(coco, skip=1):
    """
    coco — dict с COCO-аннотациями (после json.load)
    skip — сколько точек пропускать между оставляемыми
    """
    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation")
        if isinstance(seg, list):
            thinned_seg = []
            for seg_poly in seg:
                coords = []
                # каждая точка — (x, y), значит шаг по массиву = 2 * (skip + 1)
                for i in range(0, len(seg_poly), 2 * (skip + 1)):
                    coords.extend(seg_poly[i:i+2])
                # оставляем только если точек >= 6 (3 вершины)
                if len(coords) >= 6:
                    thinned_seg.append(coords)
            ann["segmentation"] = thinned_seg
    return coco
