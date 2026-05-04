
import streamlit as st
import numpy as np
import torch
import pickle
from PIL import Image

# -------------------------------
# Load data
# -------------------------------
@st.cache_resource
def load_all():
    with open("ensemble_weights.pkl", "rb") as f:
        ensemble_weights = pickle.load(f)

    with open("single_model.pkl", "rb") as f:
        single_weights = pickle.load(f)

    with open("color_mappings.pkl", "rb") as f:
        data = pickle.load(f)

    return ensemble_weights, single_weights, data

# -------------------------------
# Forward + Predict
# -------------------------------
def forward(X, params):
    for i, (W, b) in enumerate(params):
        X = X @ W + b
        if i < len(params) - 1:
            X = torch.relu(X)
    return torch.softmax(X, dim=1)

def predict(params, X):
    probs = forward(X, params)
    return torch.argmax(probs, dim=1)

# -------------------------------
# Label → Color
# -------------------------------
def labels_to_color(mask, label_to_color):
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in label_to_color.items():
        color_img[mask == label] = color

    return color_img

# -------------------------------
# Color → Label
# -------------------------------
def color_to_label(mask, color_map):
    h, w, _ = mask.shape
    label_mask = np.zeros((h, w), dtype=np.int64)

    for color, label in color_map.items():
        matches = np.all(mask == color, axis=-1)
        label_mask[matches] = label

    return label_mask

# -------------------------------
# ✅ FIXED CLASS LEGEND (HARDCODED)
# -------------------------------
class_legend = {
    0: ("Building", (60, 16, 152)),
    1: ("Land (unpaved)", (132, 41, 246)),
    2: ("Road", (110, 193, 228)),
    3: ("Vegetation", (254, 221, 58)),
    4: ("Water", (226, 169, 41)),
    5: ("Unlabeled", (155, 155, 155))
}

# -------------------------------
# Legend UI
# -------------------------------
def show_legend():
    st.markdown("### Class Legend")

    for label, (name, color) in class_legend.items():
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; margin-bottom:6px;">
                <div style="width:20px; height:20px; background-color:rgb{color}; margin-right:10px;"></div>
                <span>{name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------
# UI
# -------------------------------
st.title("Satellite Image Segmentation (Single Neural Network and Ensemble)")

ensemble_weights, single_weights, data = load_all()
label_to_color = data["label_to_color"]
color_to_label_map = data["color_to_label"]

# ✅ Show legend
show_legend()

# Upload
img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
gt_file  = st.file_uploader("Upload Ground Truth (Optional)", type=["png","jpg","jpeg"])

if img_file:
    image = Image.open(img_file).convert("RGB").resize((256, 256))
    img = np.array(image) / 255.0
    H, W, _ = img.shape

    X = img.reshape(-1, 3)
    X = torch.tensor(X, dtype=torch.float32)

    # -------------------------------
    # Predictions
    # -------------------------------
    single_pred = predict(single_weights, X)
    single_mask = single_pred.reshape(H, W).numpy()
    single_img = labels_to_color(single_mask, label_to_color)

    preds = [predict(m, X) for m in ensemble_weights]
    preds = torch.stack(preds)
    ensemble_pred = torch.mode(preds, dim=0).values
    ensemble_mask = ensemble_pred.reshape(H, W).numpy()
    ensemble_img = labels_to_color(ensemble_mask, label_to_color)

    # -------------------------------
    # If GT is uploaded
    # -------------------------------
    if gt_file:
        gt_image = Image.open(gt_file).convert("RGB")
        gt_np = np.array(gt_image)

        gt_mask = color_to_label(gt_np, color_to_label_map)

        if gt_mask.shape != (H, W):
            gt_mask = np.array(
                Image.fromarray(gt_mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
            )

        gt_img = labels_to_color(gt_mask, label_to_color)

        # Accuracy
        single_acc = (single_mask == gt_mask).mean()
        ensemble_acc = (ensemble_mask == gt_mask).mean()

        # Display
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(gt_img, caption="Ground Truth")

        with col3:
            st.image(single_img, caption="Single Model")

        with col4:
            st.image(ensemble_img, caption="Ensemble")

        st.markdown("### Accuracy")
        st.write(f"Single Model : {single_acc:.4f}")
        st.write(f"Ensemble     : {ensemble_acc:.4f}")

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(single_img, caption="Single Model")

        with col3:
            st.image(ensemble_img, caption="Ensemble")

        st.info("Ground truth not provided so accuracy not shown.")

    st.success("Segmentation Completed!")