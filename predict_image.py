import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = (300, 300)  # EfficientNetB3 input size
MODEL_PATH = "models/fire_detection_model.keras"
FIRE_THRESHOLD = 0.5  # Adjust if needed

# -----------------------------
# Load model once at import time
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Try to extract class names if available
try:
    class_names = model.class_names
except AttributeError:
    # fallback to default order; update if needed
    class_names = ['fire', 'nofire']
print("Class names:", class_names)

def predict(image_path):
    """
    Predict fire/no-fire for a given image.
    Returns a string like: "🔥 Fire (probability: 0.8453)"
    """
    # Load image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    processed_img = preprocess_input(img_array)

    # Predict
    predictions = model.predict(processed_img)[0]  # shape: (num_classes,)
    
    # Map class names dynamically
    try:
        fire_index = class_names.index('fire')
        nofire_index = class_names.index('nofire')
    except ValueError:
        # fallback
        fire_index = 0
        nofire_index = 1

    fire_confidence = predictions[fire_index]
    nofire_confidence = predictions[nofire_index]

    # Decide label
    if fire_confidence >= FIRE_THRESHOLD:
        label = "🔥 Fire"
        prob = fire_confidence
    else:
        label = "❄️ No Fire"
        prob = nofire_confidence

    return f"{label} (probability: {prob:.4f})"

# -----------------------------
# Test prediction
# -----------------------------
if __name__ == '__main__':
    test_image = "test.jpg"
    if os.path.exists(test_image):
        result = predict(test_image)
        print(f"Prediction result for {test_image}: {result}")
    else:
        print(f"Place a test image named '{test_image}' in the current directory to run a test.")
