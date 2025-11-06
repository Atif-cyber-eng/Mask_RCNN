import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import gdown
from inference import load_model, get_display_masks

st.set_page_config(page_title="Tumor Detection (Mask R-CNN)", layout="centered")
st.title("🧠 Tumor Detection with Mask R-CNN")
st.write("Upload an image (jpg/png). The app supports both bounding boxes and segmentation masks.")

# ---- Download model from Google Drive if missing
MODEL_PATH = "mask_rcnn_tumor.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "1ljJ00NffDxOue3dm-k7IPAwwzgPgF7sm"
    url = f"https://drive.google.com/uc?id={file_id}"
    st.info("Downloading trained model weights from Google Drive (this may take a minute)...")
    gdown.download(url, MODEL_PATH, quiet=False)

# ---- Load trained model (cached)
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

# UI options
show_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)
show_masks = st.sidebar.checkbox("Show segmentation masks", value=True)
min_score = st.sidebar.slider("Minimum detection confidence", 0.0, 1.0, 0.6, 0.01)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    results = model.detect([image_np], verbose=0)[0]
    rois = results.get("rois", [])
    masks = results.get("masks", np.empty((image_np.shape[0], image_np.shape[1], 0), dtype=bool))
    scores = results.get("scores", np.ones(rois.shape[0]) if rois.shape[0]>0 else np.array([]))

    # filter by score
    keep = [i for i, s in enumerate(scores) if s >= min_score]
    if len(keep) == 0:
        st.warning("No detections above the confidence threshold.")
    else:
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image_np)
        for i in keep:
            y1, x1, y2, x2 = rois[i]
            if show_masks and masks.size and masks.shape[-1] > 0:
                disp_mask = get_display_masks(masks[:, :, i])
                ax.imshow(disp_mask, alpha=0.4)
            if show_boxes:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor="r", facecolor="none")
                ax.add_patch(rect)
                score = scores[i] if i < len(scores) else None
                if score is not None:
                    ax.text(x1, y1 - 8, f"Tumor {score:.2f}", color="yellow", fontsize=10, backgroundcolor="black")
        ax.axis("off")
        st.pyplot(fig)
else:
    st.write("No image uploaded yet. Try the 'test_images' in your Drive or upload one from your computer.")
