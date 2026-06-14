import os
import json
import cv2
import numpy as np
from skimage import measure, io
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def process_file(fname, image_dir, mask_dir):
    """Обрабатывает одно изображение и возвращает (image_info, annotations)."""
    if not fname.endswith(".png"):
        return None

    img_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    if not os.path.exists(mask_path):
        print(f"Маска для {fname} не найдена, пропускаю")
        return None

    try:
        image = io.imread(img_path)
        mask = io.imread(mask_path)
    except Exception as e:
        print(f"Ошибка при чтении {fname}: {e}")
        return None

    if mask.ndim == 3:
        mask = mask[..., 0]

    image_info = {
        "file_name": fname,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
    }

    annotations = []
    props = measure.regionprops(mask)
    for p in props:
        obj_id = p.label
        m = (mask == obj_id).astype(np.uint8)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.size < 6:
                continue

            contour = contour.squeeze().ravel().tolist()

            min_row, min_col, max_row, max_col = p.bbox
            x_min = float(min_col)
            y_min = float(min_row)
            w = float(max_col - min_col)
            h = float(max_row - min_row)

            annotations.append({
                "category_id": 1,
                "segmentation": [contour],
                "area": float(p.area),
                "bbox": [x_min, y_min, w, h],
                "iscrowd": 0,
            })

    return image_info, annotations


def run_convert_masks_to_coco(image_dir, mask_dir, output_json):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "cell"}],
    }

    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    ann_id = 1
    img_id = 1

    worker = partial(process_file, image_dir=image_dir, mask_dir=mask_dir)

    with ProcessPoolExecutor() as ex:
        for result in tqdm(ex.map(worker, files), total=len(files), desc="Processing"):
            if result is None:
                continue

            image_info, anns = result
            image_info["id"] = img_id
            coco["images"].append(image_info)

            for ann in anns:
                ann["id"] = ann_id
                ann["image_id"] = img_id
                coco["annotations"].append(ann)
                ann_id += 1

            img_id += 1
            
    with open(output_json, "w") as f:
        json.dump(coco, f, separators=(",", ":"))

    print(f"\nГотово! COCO-аннотации сохранены в {output_json}")
    print(f"Изображений: {len(coco['images'])}, аннотаций: {len(coco['annotations'])}")