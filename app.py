import streamlit as st
from PIL import Image
import numpy as np
import os
import gdown
import tensorflow as tf

from utils import preprocess, infer_and_gradcam

# -------------------------------------------------
# Google Drive model download
# -------------------------------------------------
FILE_ID = "1thbvn-z9RqusPlNURPy7rmIRnWM3k3c2"
MODEL_PATH = "final_resnet_model.keras"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("AI-Based Skin Cancer Diagnosis System")
st.write("Upload a skin lesion image to get prediction and Grad-CAM visualization.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    with st.spinner("Predicting..."):
        result = infer_and_gradcam(model, img)

    st.write(f"**Prediction:** {result['label']}")
    st.write(f"**Probability:** {result['probability']:.4f}")

    st.subheader("Grad-CAM")
    st.image(result["gradcam_img"], use_column_width=True)
