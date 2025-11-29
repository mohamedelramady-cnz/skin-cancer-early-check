import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# CLASS LABELS (update to match your training order)
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
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")


def make_gradcam_heatmap(img_array, model, pred_index=None):
    last_conv = find_last_conv_layer(model)
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv).output, model.output])

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


# ---------------------------
# Prediction + GradCAM
# ---------------------------
def infer_and_gradcam(model, pil_img):
    x = preprocess(pil_img)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))

    heatmap = make_gradcam_heatmap(x, model, pred_index=idx)
    grad_img = overlay_heatmap(pil_img, heatmap)

    return {
        "label": CLASS_INDICES.get(idx, str(idx)),
        "probability": float(preds[idx]),
        "gradcam_img": grad_img
    }
