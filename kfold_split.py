import os
import shutil
import random
from glob import glob
from sklearn.model_selection import KFold

# Paths
ROOT = "images/doorwin.v1i.yolov8/train"
IMG_DIR = "images/images"
LBL_DIR = "images/labels"
KFOLD_ROOT = "kfold_yolo"
K = 5

# Gather all image files
images = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))
assert all(os.path.exists(os.path.join(LBL_DIR, os.path.basename(f).replace('.jpg', '.txt'))) for f in images), "Missing label files!"

kf = KFold(n_splits=K, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    fold_dir = os.path.join(KFOLD_ROOT, f"fold_{fold}")
    for split, idxs in [("train", train_idx), ("val", val_idx)]:
        img_out = os.path.join(fold_dir, "images", split)
        lbl_out = os.path.join(fold_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for i in idxs:
            img = images[i]
            lbl = os.path.join(LBL_DIR, os.path.basename(img).replace('.jpg', '.txt'))
            shutil.copy(img, img_out)
            shutil.copy(lbl, lbl_out)
    # Write data.yaml for this fold
    with open(os.path.join(fold_dir, "data.yaml"), "w") as f:
        f.write(f"train: images/train\nval: images/val\nnc: 2\nnames: ['door', 'window']\n")
print("K-fold split complete.") 