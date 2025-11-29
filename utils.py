import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# CLASS LABELS
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
# Grad-CAM
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
    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv.output, model.output])

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
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

# ---------------------------
# Prediction + optional GradCAM
# ---------------------------
def predict(model, pil_img, gradcam=False):
    x = preprocess(pil_img)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))

    result = {
        "label": CLASS_INDICES.get(idx, str(idx)),
        "probability": float(preds[idx]),
        "all_probs": {CLASS_INDICES[i]: float(p) for i, p in enumerate(preds)}
    }

    if gradcam:
        heatmap = make_gradcam_heatmap(x, model, pred_index=idx)
        result["gradcam_img"] = overlay_heatmap(pil_img, heatmap)

    return result
