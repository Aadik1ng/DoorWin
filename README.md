# DoorWin: K-Fold YOLO Inference and Training

## Project Overview
This project implements a k-fold cross-validation pipeline for object detection using YOLO (You Only Look Once) models. It supports both batch inference and an interactive Streamlit UI for testing images with models trained on different folds. The main use case is robust evaluation and deployment of YOLO models for door detection in images.

---

## Project Structure
```
DoorWin/
├── DoorWin/
│   ├── Lib/
│   │   └── site-packages/           # Python packages (YOLO, torch, etc.)
│   └── ...
├── images/
│   ├── images/                      # Test images for inference
│   └── labels/                      # (Optional) Labels for test images
├── kfold_yolo/
│   ├── fold_0/
│   │   ├── images/train/val/        # Training/validation images for fold 0
│   │   ├── labels/train/val/        # Training/validation labels for fold 0
│   │   └── runs/detect/weights/     # YOLO weights (best.pt, last.pt)
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
├── yolo_test_results/
│   ├── fold_0/                      # Inference results for fold 0
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
├── infer.py                         # Batch inference script
├── streamlit_infer.py                # Streamlit UI for inference
├── kfold_split.py                   # (Optional) Script for k-fold data splitting
├── yolo_train_kfold.py              # (Optional) Script for k-fold YOLO training
├── yolov8n.pt                       # (Optional) Pretrained YOLO model
├── test.png                         # Example test image
└── ...
```

---

## Methodologies Used

### 1. K-Fold Cross-Validation for YOLO
- The dataset is split into 5 folds (can be changed in code).
- For each fold, a YOLO model is trained using 4 folds for training and 1 for validation.
- Each fold produces its own set of weights (`best.pt`, `last.pt`).
- This approach provides a robust estimate of model performance and helps prevent overfitting.

### 2. Batch Inference (`infer.py`)
- Runs inference on a single image (`test.png`) using all 5 fold models.
- For each fold, both `best.pt` and `last.pt` weights are used.
- Results are saved in `yolo_test_results/fold_X/` with clear filenames indicating the model and image.

### 3. Streamlit Inference UI (`streamlit_infer.py`)
- Provides a web interface for uploading an image, selecting a fold, and choosing a model (`best.pt` or `last.pt`).
- Displays the inference result and saves it in the appropriate results directory.
- Useful for quick testing and demonstration.

---

## How to Run

### 1. Batch Inference
Run the following command to perform inference on `test.png` with all folds:
```bash
python infer.py
```
Results will be saved in `yolo_test_results/fold_X/`.

### 2. Streamlit Inference UI
Start the Streamlit app with:
```bash
streamlit run streamlit_infer.py
```
- Open the provided local URL in your browser.
- Upload an image, select a fold and model, and view/save the results interactively.

---

## Requirements
- Python 3.8+
- torch
- ultralytics (YOLO)
- streamlit
- pillow

Install dependencies (if needed):
```bash
pip install ultralytics streamlit pillow
```

---

## Notes & Known Issues
- **Streamlit + torch on Windows:** You may see harmless errors in the terminal related to `torch` and Streamlit's file watcher. These do not affect inference or UI functionality.
- Results are organized by fold and model for easy comparison.
- For custom datasets or more folds, adjust the code and directory structure as needed.

---

## Contact
For questions or contributions, please open an issue or pull request. 