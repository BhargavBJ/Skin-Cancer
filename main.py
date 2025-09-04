import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

st.set_page_config(page_title="Skin Lesion Classifier (HAM10000)", page_icon="🧴", layout="wide")

st.title("🧴 Skin Lesion Classifier (HAM10000)")
st.write("Upload a dermatoscopic image to get predicted lesion type and class probabilities. "
         "This app expects a PyTorch model trained for 7 HAM10000 classes.")

# -----------------------------
# Sidebar: Model
# -----------------------------
st.sidebar.header("Model")
uploaded_model = st.sidebar.file_uploader("Upload model weights (.pth/.pt)", type=["pth", "pt"])

# Default model path
default_model_path = "resnet18_skin_cancer.pth"
use_default = False
if os.path.exists(default_model_path) and uploaded_model is None:
    use_default = st.sidebar.checkbox(f"Use bundled weights: {default_model_path}", value=True)

device = torch.device("cpu")  # force CPU
st.sidebar.write(f"Device: {device}")

# -----------------------------
# HAM10000 labels
# -----------------------------
label_map = {
    "akiec": "Actinic keratoses / Intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}
idx_to_key = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]
idx_to_name = [label_map[k] for k in idx_to_key]

# -----------------------------
# Preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def build_resnet18(num_classes=7):
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

@st.cache_resource(show_spinner=False)
def load_model_from_bytes(buf: bytes):
    by = io.BytesIO(buf)
    obj = torch.load(by, map_location="cpu")
    if isinstance(obj, dict):
        model = build_resnet18(7)
        model.load_state_dict(obj, strict=False)
        return model.eval()
    else:
        # full model
        model = obj
        model.eval()
        return model

@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return load_model_from_bytes(data)

def get_model():
    if uploaded_model is not None:
        return load_model_from_bytes(uploaded_model.read())
    if use_default and os.path.exists(default_model_path):
        return load_model_from_path(default_model_path)
    st.warning("Please upload model weights (.pth/.pt) from the sidebar.")
    return None

def predict_image(model, pil_img: Image.Image):
    img = pil_img.convert("RGB")
    t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    return probs, top_idx

# -----------------------------
# UI: Image uploader
# -----------------------------
st.header("Upload Image(s)")
images = st.file_uploader("Upload dermatoscopic image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

model = get_model()

if images and model is not None:
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Batch Results")
        results = []
        for f in images:
            try:
                pil = Image.open(f)
            except Exception as e:
                st.error(f"Could not open {f.name}: {e}")
                continue
            probs, top_idx = predict_image(model, pil)
            results.append((f.name, probs, top_idx, pil))

        # Summary table
        rows = []
        for name, probs, top_idx, _ in results:
            row = {"image": name, "pred_class": idx_to_name[top_idx]}
            for i, cls in enumerate(idx_to_name):
                row[f"p({cls})"] = float(probs[i])
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    with cols[1]:
        st.subheader("Per-Image Details")
        for name, probs, top_idx, pil in results:
            st.markdown(f"**{name}** — Predicted: **{idx_to_name[top_idx]}**")
            st.image(pil, caption=name, use_column_width=True)

            fig = plt.figure()
            plt.bar(range(len(idx_to_name)), probs)
            plt.xticks(range(len(idx_to_name)), idx_to_key, rotation=30)
            plt.ylabel("Probability")
            plt.title("Class probabilities")
            st.pyplot(fig)
            plt.close(fig)

elif model is None:
    st.info("Load the model to enable predictions.")
elif not images:
    st.info("Upload one or more images to get predictions.")
