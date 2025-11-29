import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ---------------------------
# Class labels (match your model training)
# ---------------------------
CLASS_INDICES = {
    0: "Cancer",
    1: "Melanocytic Nevus",
    2: "Benign – Other"
}

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(pil_img, size=(224, 224)):
    arr = np.array(pil_img.resize(size)) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

# ---------------------------
# Find last Conv2D layer inside a nested Functional model
# ---------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            conv_layer = find_last_conv_layer(layer)  # recursive search
            if conv_layer is not None:
                return conv_layer
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None

# ---------------------------
# Grad-CAM heatmap
# ---------------------------
def make_gradcam_heatmap(img_array, model, pred_index=None):
    # Access the base ResNet model inside your model
    try:
        base_resnet = model.get_layer("resnet50")  # your base model name
    except ValueError:
        raise ValueError("Base ResNet50 model not found inside your model.")

    # Find the last Conv2D inside ResNet
    last_conv_layer = None
    for layer in reversed(base_resnet.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found inside ResNet50.")

    # Build a grad model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    return heatmap


# ---------------------------
# Overlay Grad-CAM on image
# ---------------------------
def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    img = np.array(pil_img)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = heatmap_color * alpha + img * (1 - alpha)
    overlay = np.uint8(np.clip(overlay, 0, 255))
    return overlay

# ---------------------------
# Prediction + Grad-CAM
# ---------------------------
def infer_and_integrated_gradients(model, pil_img, top_k=3):
    x = preprocess(pil_img)
    preds = model.predict(x)[0]

    # Top-k predictions
    top_indices = preds.argsort()[-top_k:][::-1]
    top_labels = [CLASS_INDICES[i] for i in top_indices]
    top_probs = [float(preds[i]) for i in top_indices]

    # Grad-CAM for top-1
    idx = top_indices[0]
    heatmap = make_gradcam_heatmap(x, model, pred_index=idx)
    gradcam_img = overlay_heatmap(pil_img, heatmap, alpha=0.4)

    return {
        "top_labels": top_labels,
        "top_probs": top_probs,
        "gradcam_img": gradcam_img,
        "label": top_labels[0],
        "probability": top_probs[0]
    }
