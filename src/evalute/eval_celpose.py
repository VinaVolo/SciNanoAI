import os
import time
from tqdm import tqdm
from tqdm import trange
from pathlib import Path
import numpy as np
from cellpose import models, io, core
import os, json
import numpy as np
from skimage import measure, io
from tqdm import tqdm
    

use_gpu = core.use_gpu()
model = models.CellposeModel(gpu=use_gpu)

input_dir = Path("/home/vinavolo/Project/cell-semantic-segment/data/TEST/orig")
orig_dir = Path("/home/vinavolo/Project/cell-semantic-segment/data/TEST_CEL/orig")
mask_dir = Path("/home/vinavolo/Project/cell-semantic-segment/data/TEST_CEL/masks")

# Создаём выходные папки, если их нет
orig_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

ann_id = 1
for root, dirs, files in tqdm(os.walk(input_dir)):
    for idx, fname in enumerate(files, start=1):
        files_gen = Path(root) / fname
        img = io.imread(files_gen)
        if img is None:
            print(f"⚠️ Не удалось прочитать файл: {files_gen}")
            continue

        masks = model.eval(img)[0]

        base_name = Path(fname).stem
        orig_path = orig_dir / f"{base_name}.png"
        mask_path = mask_dir / f"{base_name}.png"

        io.imsave(orig_path, img)
        io.imsave(mask_path, masks)



    image_dir = orig_dir
    mask_dir = mask_dir

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "cell"}],
    }


    if not orig_path.endswith(".png"):
        continue


    image = io.imread(orig_path)
    mask = io.imread(mask_path)

    coco["images"].append({
        "id": i,
        "file_name": fname,
        "height": image.shape[0],
        "width": image.shape[1],
    })

    # для каждой instance-маски (если разные ID)
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue
        m = (mask == obj_id).astype(np.uint8)
        contours = measure.find_contours(m, 0.5)
        for contour in contours:
            contour = contour[:, ::-1].ravel().tolist()
            coco["annotations"].append({
                "id": ann_id,
                "image_id": i,
                "category_id": 1,
                "segmentation": [contour],
                "area": float(np.sum(m)),
                "bbox": list(map(float, measure.regionprops(m)[0].bbox)),
                "iscrowd": 0
            })
            ann_id += 1

    with open("annotations_coco.json", "w") as f:
        json.dump(coco, f)
