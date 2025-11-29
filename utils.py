import streamlit as st
from PIL import Image
import os
import gdown
import tensorflow as tf

from utils import predict_class, generate_gradcam

# ---------------------------
# Download & load model
# ---------------------------
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

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("AI-Based Skin Cancer Diagnosis System")
st.write("Upload a skin lesion image to get prediction and Grad-CAM visualization.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    # Predict first
    with st.spinner("Predicting class..."):
        result = predict_class(model, img)

    st.write(f"**Prediction:** {result['label']}")
    st.write(f"**Probability:** {result['probability']:.4f}")
    st.info(result['report'])

    # Show Grad-CAM button
    if st.button("Show Grad-CAM"):
        with st.spinner("Generating Grad-CAM..."):
            gradcam_img = generate_gradcam(model, img, result['preprocessed'], pred_idx=None)
        st.subheader("Grad-CAM")
        st.image(gradcam_img, use_column_width=True)
