import streamlit as st
from PIL import Image
import os
import gdown
import tensorflow as tf
from utils import infer_and_integrated_gradients

FILE_ID = "1thbvn-z9RqusPlNURPy7rmIRnWM3k3c2"
MODEL_PATH = "final_resnet_model.keras"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("AI-Based Skin Cancer Diagnosis System")
uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)
    with st.spinner("Predicting..."):
        result = infer_and_integrated_gradients(model, img)
    st.subheader("Prediction Results")
    st.write(f"**Predicted Label:** {result['label']}")
    st.write(f"**Probability:** {result['probability']:.4f}")
    st.subheader("Integrated Gradients Overlay")
    st.image(result["gradcam_img"], use_column_width=True)
