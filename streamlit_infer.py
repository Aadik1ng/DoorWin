import os
os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = "*/torch/*,*/ultralytics/*"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import warnings
warnings.filterwarnings("ignore")
st.title("YOLO K-Fold Inference UI")

# UI for image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# UI for fold and model selection
fold = st.selectbox("Select Fold", list(range(5)), format_func=lambda x: f"Fold {x}")
model_name = st.selectbox("Select Model", ["best.pt", "last.pt"]) 

if uploaded_file is not None:
    # Save uploaded image to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_img_path = tmp_file.name

    st.image(tmp_img_path, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Inference"):
        model_path = f"kfold_yolo/fold_{fold}/runs/detect/weights/{model_name}"
        if not os.path.exists(model_path):
            st.error(f"Model not found: {model_path}")
        else:
            model = YOLO(model_path)
            results = model(tmp_img_path)
            for i, r in enumerate(results):
                im_array = r.plot()
                im = Image.fromarray(im_array)
                # Save result
                output_dir = os.path.join("yolo_test_results", f"fold_{fold}")
                os.makedirs(output_dir, exist_ok=True)
                out_name = f"{model_name.replace('.pt','')}_{os.path.basename(tmp_img_path)}"
                out_path = os.path.join(output_dir, out_name)
                im.save(out_path)
                st.image(im, caption=f"Result: {model_name}", use_container_width=True)
                st.success(f"Saved: {out_path}") 