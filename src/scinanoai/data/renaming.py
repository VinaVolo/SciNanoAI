import os
from src.utils.paths import get_project_path

def rename_images(path_to_img=os.path.join(get_project_path(), "data")):
    """
    Rename images in a given directory and its subdirectories to a standard format
    (directory name followed by a zero-padded index).

    Parameters:
    
    path_to_img (str): path to the directory containing the images to be renamed
    """
    for root, dirs, files in os.walk(path_to_img):
        files = sorted(files)

        rel = os.path.relpath(root, start=os.path.dirname(path_to_img))
        prefix = rel.replace(os.sep, "_")
        
        for idx, fname in enumerate(files, start=1):
            src = os.path.join(root, fname)
            _, ext = os.path.splitext(fname)
            new_name = f"{prefix}_{str(idx).zfill(3)}{ext}"
            dst = os.path.join(root, new_name)
            print(f"Renaming: {src} to {dst}")
            os.rename(src, dst)
