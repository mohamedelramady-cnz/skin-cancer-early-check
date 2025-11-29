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
st.write("Upload a skin lesion image to get prediction and optional Grad-CAM visualization.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    with st.spinner("Predicting..."):
        result = infer_and_gradcam(model, img)  # or your prediction function

    # ---------------------------
    # Display Prediction
    # ---------------------------
    st.write(f"**Prediction:** {result['label']}")
    st.write(f"**Probability:** {result['probability']:.4f}")

    # Optional Grad-CAM
    if st.checkbox("Generate Grad-CAM visualization?"):
        st.subheader("Grad-CAM")
        st.image(result["gradcam_img"], use_column_width=True)

    # ---------------------------
    # Dynamic Prediction Report
    # ---------------------------
    # Here you add the report & probability chart
    class_names = list(result["all_probs"].keys())      # dict keys of all classes
    probs = list(result["all_probs"].values())         # dict values
    pred = {"pred_index": np.argmax(probs)}            # predicted index
    gradcam_img = result.get("gradcam_img", None)      # grad-cam image if exists

    st.subheader("Prediction Report")
    report_text = ""
    for i, name in enumerate(class_names):
        prob = float(probs[i])
        if i == pred["pred_index"]:
            report_text += f"**➡ {name}: {prob:.4f}**  \n"
        else:
            report_text += f"{name}: {prob:.4f}  \n"

    st.markdown(report_text)

    st.subheader("Class Probabilities")
    import pandas as pd
    df_probs = pd.DataFrame({"Class": class_names, "Probability": probs})
    df_probs = df_probs.sort_values("Probability", ascending=False)
    st.bar_chart(df_probs.set_index("Class"))
