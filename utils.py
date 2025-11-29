import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ---------------------------
# Class labels (update as per your model)
# ---------------------------
CLASS_INDICES = {
    0: "Cancer",
    1: "Nevus",
    2: "Benign"
}

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(pil_img, size=(224, 224)):
    """
    Resize and normalize PIL image for model input.
    """
    pil_img = pil_img.resize(size)
    arr = np.array(pil_img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

# ---------------------------
# Integrated Gradients prediction + visualization
# ---------------------------
def infer_and_integrated_gradients(model, pil_img, alpha=0.6, steps=50):
    """
    Predict class and generate Integrated Gradients overlay.
    Works with Dense-heavy models like your enhanced ResNet.
    """
    # Preprocess
    img_array = np.array(pil_img.resize((224, 224))) / 255.0
    img_array = img_array.astype(np.float32)
    img_array_exp = np.expand_dims(img_array, 0)

    # Prediction
    preds = model.predict(img_array_exp)
    pred_idx = int(np.argmax(preds[0]))
    pred_prob = float(preds[0][pred_idx])

    # Integrated Gradients
    baseline = np.zeros_like(img_array)
    interpolated = np.array([baseline + (i/steps)*(img_array - baseline) for i in range(steps+1)])
    
    # Convert to tf.Tensor
    interpolated = tf.convert_to_tensor(interpolated, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        logits = model(interpolated)
        target = logits[:, pred_idx]
    grads = tape.gradient(target, interpolated)
    
    # Average over interpolation steps
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (img_array - baseline) * avg_grads
    integrated_grads -= tf.reduce_min(integrated_grads)
    integrated_grads /= tf.reduce_max(integrated_grads) + 1e-8

    # Convert to heatmap
    heatmap = (integrated_grads.numpy() * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted((img_array*255).astype(np.uint8), 1-alpha, heatmap, alpha, 0)

    return {
        "label": CLASS_INDICES.get(pred_idx, str(pred_idx)),
        "probability": pred_prob,
        "gradcam_img": overlay
    }
