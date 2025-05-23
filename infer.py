import os
from ultralytics import YOLO
from PIL import Image

# Path to the test image
TEST_IMAGE_PATH = "test.png"
# Output directory for results
OUTPUT_DIR = "yolo_test_results"
FOLDS = 5
MODEL_NAMES = ["best.pt", "last.pt"]

for fold in range(FOLDS):
    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    for model_name in MODEL_NAMES:
        model_path = f"kfold_yolo/fold_{fold}/runs/detect/weights/{model_name}"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        print(f"Using model: {model_path}")
        model = YOLO(model_path)
        results = model(TEST_IMAGE_PATH)
        for i, r in enumerate(results):
            im_array = r.plot()
            im = Image.fromarray(im_array)
            out_name = f"{model_name.replace('.pt','')}_test.png"
            out_path = os.path.join(fold_output_dir, out_name)
            im.save(out_path)
            print(f"Saved: {out_path}")