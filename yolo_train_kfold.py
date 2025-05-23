import os
from ultralytics import YOLO

KFOLD_ROOT = "kfold_yolo"
FOLDS = 5  # Number of folds
MODEL = "yolov8n.pt"  # You can change to yolov8s.pt, yolov8m.pt, etc.
EPOCHS = 50
IMG_SIZE = 640

for fold in range(FOLDS):
    fold_dir = os.path.join(KFOLD_ROOT, f"fold_{fold}")
    data_yaml = os.path.join(fold_dir, "data.yaml")
    print(f"\n=== Training Fold {fold} ===")
    model = YOLO(MODEL)
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=os.path.join(fold_dir, "runs"),
        name="detect"
    )
    print(f"Fold {fold} training complete. Results in {fold_dir}/runs/detect/")