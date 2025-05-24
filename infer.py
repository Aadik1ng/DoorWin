import os
from ultralytics import YOLO
from PIL import Image
import tempfile
from enum import Enum

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import sys

# Path to the test image
TEST_IMAGE_PATH = "test.png"
# Output directory for results
OUTPUT_DIR = "yolo_test_results"
FOLDS = 5
MODEL_NAMES = ["best.pt", "last.pt"]

class Fold(int, Enum):
    fold_0 = 0
    fold_1 = 1
    fold_2 = 2
    fold_3 = 3
    fold_4 = 4

class ModelName(str, Enum):
    best = "best.pt"
    last = "last.pt"

app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    fold: Fold = Form(Fold.fold_0),
    model_name: ModelName = Form(ModelName.best)
):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load model
    model_path = f"kfold_yolo/fold_{fold.value}/runs/detect/weights/{model_name.value}"
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}
    model = YOLO(model_path)
    results = model(tmp_path)
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array)
        out_path = tmp_path + "_result.png"
        im.save(out_path)
        return FileResponse(out_path, media_type="image/png")
    return {"error": "No result"}

if __name__ == "__main__":
    # Batch inference on test.png as before
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