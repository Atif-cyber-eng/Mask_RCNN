
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import gdown
from inference import load_model

st.title(" Tumor Detection using Mask R-CNN")
st.write("Upload an MRI or medical image to detect tumor regions.")

# ---- Download model from Google Drive (only first time)
MODEL_PATH = "mask_rcnn_tumor.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "1ljJ00NffDxOue3dm-k7IPAwwzgPgF7sm"  # 👈 replace with your Google Drive file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    st.write("📥 Downloading trained model weights...")
    gdown.download(url, MODEL_PATH, quiet=False)

# ---- Load trained model
model = load_model(MODEL_PATH)

# ---- Upload and predict
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    results = model.detect([image_np], verbose=0)[0]

    rois = results["rois"]
    masks = results["masks"]
    scores = results["scores"]

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image_np)

    for i, roi in enumerate(rois):
        y1, x1, y2, x2 = roi
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Tumor {scores[i]:.2f}",
                color="yellow", fontsize=10, backgroundcolor="black")

    ax.axis("off")
    st.pyplot(fig)
