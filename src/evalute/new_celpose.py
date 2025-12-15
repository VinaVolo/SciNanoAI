import os
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import numpy as np
from skimage import io as skio, measure
from cellpose import models, core
import cv2
import json

INPUT_DIR = Path("/home/vinavolo/Project/cell-semantic-segment/data/raw")
OUT_IMG_DIR = Path("/home/vinavolo/Project/cell-semantic-segment/data/processed_new/orig")
OUT_MASK_DIR = Path("/home/vinavolo/Project/cell-semantic-segment/data/processed_new/masks")
OUT_ANN_DIR = Path("/home/vinavolo/Project/cell-semantic-segment/data/processed_new/ann_per_image")  # COCO на картинку

# ==== ИНИЦИАЛИЗАЦИЯ ====
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

use_gpu = core.use_gpu()
model = models.CellposeModel(gpu=use_gpu)

# Счётчики COCO (локально для уникальности id в рамках одного изображения можно начинать с 1)
image_global_id = 1
ann_global_id = 1

counter = 0
for root, _, files in tqdm(os.walk(INPUT_DIR)):
    for fname in tqdm(sorted(files), desc="Processing"):
        fpath = Path(root) / fname
        try:
            img = skio.imread(fpath)
            counter += 1
            
            if counter < 1400:
                continue
        except Exception as e:
            print(f"⚠️ Не удалось прочитать {fpath}: {e}")
            continue

        if img is None:
            print(f"⚠️ Пустое изображение: {fpath}")
            continue

        # === 1) СЕГМЕНТАЦИЯ ===
        masks = model.eval(img)[0]

        # === 2) СОХРАНЕНИЕ ИЗОБРАЖЕНИЯ И МАСКИ (опционально) ===
        base = Path(fname).stem
        out_img = OUT_IMG_DIR / f"{base}.png"
        out_mask = OUT_MASK_DIR / f"{base}.png"
        try:
            skio.imsave(out_img, img, check_contrast=False)
            skio.imsave(out_mask, masks.astype(np.uint16), check_contrast=False)
        except Exception as e:
            print(f"⚠️ Не удалось сохранить {base}: {e}")
            continue

        # === 3) СБОРКА COCO ДЛЯ КОНКРЕТНО ЭТОГО ИЗОБРАЖЕНИЯ ===
        if img.ndim == 3:
            h, w = img.shape[0], img.shape[1]
        else:
            h, w = img.shape

        coco_one = {
            "info": {"description": "Cellpose → COCO (per-image)", "version": "1.0"},
            "licenses": [],
            "images": [{
                "id": image_global_id,
                "file_name": out_img.name,
                "width": w,
                "height": h,
            }],
            "annotations": [],
            "categories": [{"id": 1, "name": "cell", "supercategory": "cell"}],
        }

        num_instances = int(masks.max())
        for inst_id in range(1, num_instances + 1):
            inst = (masks == inst_id).astype(np.uint8)
            if inst.sum() == 0:
                continue

            # Контуры -> полигоны
            contours = measure.find_contours(inst, level=0.5)
            segmentations = []
            for cnt in contours:
                if cnt.shape[0] < 3:
                    continue
                poly = cnt[:, [1, 0]].ravel().tolist()  # (x,y)
                if len(poly) >= 6:
                    segmentations.append(poly)

            if not segmentations:
                continue

            # bbox [x,y,w,h]
            coords = np.column_stack(np.nonzero(inst))  # (row, col)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
            area = int(inst.sum())

            coco_one["annotations"].append({
                "id": ann_global_id,
                "image_id": image_global_id,
                "category_id": 1,
                "segmentation": segmentations,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_global_id += 1

        # если нет объектов — всё равно можно загрузить без аннотаций, или пропустить:
        if len(coco_one["annotations"]) == 0:
            print(f"ℹ️ Нет инстансов на {out_img.name}, пропускаю загрузку аннотаций.")

        # === 4) СОХРАНЯЕМ ВРЕМЕННЫЙ COCO ДЛЯ ЭТОЙ КАРТИНКИ ===
        ann_path = OUT_ANN_DIR / f"{base}.coco.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco_one, f, ensure_ascii=False)

        unique_ids = np.unique(masks)
        unique_ids = unique_ids[unique_ids != 0]  # пропускаем фон

        outlined = img.copy()
        for uid in unique_ids:
            obj = (masks == uid).astype(np.uint8)
            cnts, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outlined, cnts, -1, (0, 255, 0), 2)  # тонкий зелёный контур
        cv2.imwrite(f"/home/vinavolo/Project/cell-semantic-segment/data/processed_new/view/{fname}.jpg", outlined)


        image_global_id += 1

print("Готово.")
