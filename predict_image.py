# predict_image.py
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = (300, 300)  # EfficientNetB3 input
MODEL_PATH = "models/fire_detection_model.keras"
FIRE_THRESHOLD = 0.5  # Probability above this is considered FIRE

# -----------------------------
# Load model once
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# -----------------------------
# Prediction function
# -----------------------------
def predict(image_path):
    """
    Predicts fire/no fire from an image using EfficientNetB3.
    Returns string: "🔥 Fire (probability: 0.95)" or "❄️ No Fire (probability: 0.80)"
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and resize image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # (1, 300, 300, 3)
    img_array = preprocess_input(img_array)    # [-1,1] scaling

    # Predict
    predictions = model.predict(img_array)  # shape: (1, 2)
    probabilities = predictions[0]

    # Assuming index 0 = fire, index 1 = no fire
    fire_confidence = probabilities[0]
    nofire_confidence = probabilities[1]

    if fire_confidence >= FIRE_THRESHOLD:
        label = "🔥 Fire"
        prob = fire_confidence
    else:
        label = "❄️ No Fire"
        prob = nofire_confidence

    return f"{label} (probability: {prob:.4f})"

# -----------------------------
# Test run
# -----------------------------
if __name__ == "__main__":
    test_image = "test.jpg"
    if os.path.exists(test_image):
        result = predict(test_image)
        print(f"Prediction result for {test_image}: {result}")
    else:
        print(f"Place a test image named '{test_image}' in the current directory to test.")
