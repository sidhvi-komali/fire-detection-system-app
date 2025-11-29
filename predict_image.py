import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "models/fire_detection_model.keras"
FIRE_THRESHOLD = 0.5  # Probability above this is considered FIRE

# -----------------------------
# Load model once
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train or place model at that path.")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Get input shape from model (ignores batch dimension)
MODEL_INPUT_SHAPE = model.input_shape[1:3]  # (height, width)

def predict(image_path):
    """
    Loads an image, preprocesses it for the model, and returns a classification string.
    Automatically resizes to the model's input size.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load and resize image to match model input
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=MODEL_INPUT_SHAPE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension (1, H, W, 3)

    # Preprocess image for EfficientNet
    processed_img = preprocess_input(img_array)

    # Predict
    predictions = model.predict(processed_img, verbose=0)
    probabilities = predictions[0]

    # Determine Fire/No Fire
    # Assumes class index 0 = 'fire', index 1 = 'nofire'
    fire_confidence = probabilities[0]
    nofire_confidence = probabilities[1] if len(probabilities) > 1 else 1 - fire_confidence

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
