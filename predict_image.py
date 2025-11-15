import tensorflow as tf
import cv2
import numpy as np

# -----------------------------
# Load model once at import time
# -----------------------------
model_path = "models/fire_detection_model.keras"
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully")

# Optional: print model summary
# model.summary()


def predict(image_path):
    """
    Takes an image path, preprocesses it, and returns the fire/no-fire prediction.
    """
    img_size = (224, 224)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, img_size)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    pred_prob = model.predict(img)[0][0]
    label = "❄️ NO FIRE" if pred_prob > 0.5 else "🔥 FIRE DETECTED"

    return f"{label} (probability: {pred_prob:.4f})"