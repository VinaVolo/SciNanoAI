import os
import sys
sys.path.append("..")

import json, os, shutil
from pathlib import Path
from typing import Dict, Any, Iterable, Set, List

from ultralytics import YOLO


from src.utils.paths import get_project_path

model = YOLO(os.path.join(get_project_path(), "models", "yolo11x-seg.pt"))

model.train(data="/home/vinavolo/Project/cell-semantic-segment/data/prepared/SINGLE_3_INSTANCE-3_yolo/data.yaml",
            epochs=100, 
            imgsz=640, 
            batch=16, 
            device=0, 
            workers=2)

model.val(data="/home/vinavolo/Project/cell-semantic-segment/data/prepared/SINGLE_3_INSTANCE-3_yolo/data.yaml")