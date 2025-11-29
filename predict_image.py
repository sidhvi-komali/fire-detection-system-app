# predict_image.py
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = (224, 224) # Fix to match model input
MODEL_PATH = "models/fire_detection_model.keras"
FIRE_THRESHOLD = 0.5

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
    Predicts fire/no fire from an image.
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # -------------------------
    # Load + preprocess image
    # -------------------------
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # (1, H, W, 3)
    img_array = preprocess_input(img_array)
    
    # -------------------------
    # Run prediction
    # -------------------------
    predictions = model.predict(img_array)
    
    # Binary model output → single probability (fire)
    fire_confidence = float(predictions[0][0])

    # Binary model output → single probability (nofire)
    nofire_confidence = 1-fire_confidence
    
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
