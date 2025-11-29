import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Update class labels to match your training
CLASS_INDICES = {
    0: "Cancer",
    1: "Nevus",
    2: "Benign"
}

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(pil_img, size=(224, 224)):
    pil_img = pil_img.resize(size)
    arr = np.array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# Prediction + report
# ---------------------------
def predict_class(model, pil_img):
    x = preprocess(pil_img)
    preds = model.predict(x)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_INDICES[pred_idx]
    pred_prob = float(preds[pred_idx])

    # Create a simple report
    report = f"The model predicts this image as **{pred_label}** " \
             f"with probability **{pred_prob:.4f}**."

    return {
        "pred_index": pred_idx,
        "label": pred_label,
        "probability": pred_prob,
        "preprocessed": x,
        "report": report
    }

# ---------------------------
# Grad-CAM (generate after prediction)
# ---------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            conv_layer = find_last_conv_layer(layer)
            if conv_layer is not None:
                return conv_layer
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None

def make_gradcam_heatmap(img_array, model, pred_index=None):
    last_conv = find_last_conv_layer(model)
    grad_model = tf.keras.models.Model([model.inputs],
                                       [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    return heatmap

def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    img = np.array(pil_img)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = heatmap_color * alpha + img * (1 - alpha)
    overlay = np.uint8(np.clip(overlay, 0, 255))
    return Image.fromarray(overlay)

def generate_gradcam(model, pil_img, preprocessed_img, pred_idx=None):
    heatmap = make_gradcam_heatmap(preprocessed_img, model, pred_index=pred_idx)
    gradcam_img = overlay_heatmap(pil_img, heatmap)
    return gradcam_img
