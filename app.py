import streamlit as st
from PIL import Image
import os
import gdown
import tensorflow as tf
import numpy as np

from utils import predict_class, generate_gradcam, CLASS_INDICES

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
        # Step 1: Predict
        pred = predict_class(model, img)

        # Step 2: Generate Grad-CAM
        gradcam_img = generate_gradcam(model, img, pred["preprocessed"], pred_idx=pred["pred_index"])

    # ---------------------------
    # Display Results
    # ---------------------------
    st.subheader("Prediction Report")
    st.markdown(pred["report"])

    st.subheader("Grad-CAM Visualization")
    st.image(gradcam_img, use_column_width=True)

    # ---------------------------
    # Probability Bar Chart
    # ---------------------------
    st.subheader("Class Probabilities")
    probs = model.predict(pred["preprocessed"])[0]
    class_names = list(CLASS_INDICES.values())

    prob_dict = {name: float(prob) for name, prob in zip(class_names, probs)}
    st.bar_chart(prob_dict)
