import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Class labels
CLASS_INDICES = {
    0: "Cancer",
    1: "Nevus",
    2: "Benign"
}

# Preprocess image
def preprocess(pil_img, size=(224, 224)):
    arr = np.array(pil_img.resize(size)) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

# Integrated Gradients prediction + overlay
def infer_and_integrated_gradients(model, pil_img, alpha=0.6, steps=50):
    img_array = np.array(pil_img.resize((224, 224))) / 255.0
    img_array = img_array.astype(np.float32)
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0))

    preds = model(img_tensor)
    pred_idx = int(tf.argmax(preds[0]))
    pred_prob = float(preds[0][pred_idx])

    # Simple gradient overlay for demonstration (works on Dense-heavy models)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        logits = model(img_tensor)
        target = logits[:, pred_idx]
    grads = tape.gradient(target, img_tensor)[0].numpy()
    grads = np.maximum(grads, 0)
    grads -= grads.min()
    grads /= grads.max() + 1e-8

    heatmap = (grads * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted((img_array*255).astype(np.uint8), 1-alpha, heatmap, alpha, 0)

    return {
        "label": CLASS_INDICES[pred_idx],
        "probability": pred_prob,
        "gradcam_img": overlay
    }
