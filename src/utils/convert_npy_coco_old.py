import os, json
import numpy as np
from skimage import measure, io
from tqdm import tqdm

image_dir = "/home/vinavolo/Project/cell-semantic-segment/data/processed/orig"
mask_dir = "/home/vinavolo/Project/cell-semantic-segment/data/processed/masks"

coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "cell"}],
}
ann_id = 1

for i, fname in tqdm(enumerate(os.listdir(image_dir))):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    image = io.imread(img_path)
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

