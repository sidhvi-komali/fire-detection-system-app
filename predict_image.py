import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore # ⬅️ NEW/CORRECT IMPORT

# -----------------------------
# Configuration
# -----------------------------
# The image size MUST match the input size of the EfficientNetB3 model (300, 300)
IMAGE_SIZE = (300, 300) 
MODEL_PATH = "models/fire_detection_model.keras"

# Prediction threshold: probability above this is considered FIRE.
# Based on the Softmax output, the probability is for the index corresponding to 'fire'.
FIRE_THRESHOLD = 0.5 # Resetting to 0.5 for new Softmax model

# -----------------------------
# Load model once at import time
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train or place model at that path.")

try:
    # Load the best model saved by ModelCheckpoint
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    # Handle potential loading issues (e.g., custom objects)
    print(f"Error loading model: {e}")
    raise e


def predict(image_path):
    """
    Loads an image, preprocesses it for EfficientNetB3, and returns a classification.
    """
    # Use Keras utilities for reliable image loading
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=IMAGE_SIZE
    )
    if img is None:
        raise ValueError(f"Image not found at: {image_path}")

    # Convert to array and add batch dimension
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Shape becomes (1, 300, 300, 3)

    # Apply EfficientNet's specific preprocessing (scales to [-1, 1])
    processed_img = preprocess_input(img_array)

    # Predict: Output is a 2-element array [P(Fire), P(No Fire)]
    predictions = model.predict(processed_img)
    probabilities = predictions[0]

    # The class names list (train_ds.class_names) determines the index order:
    # Based on your output: ['fire', 'nofire'], so Fire is index 0.
    fire_confidence = probabilities[0]
    
    # Determine the label based on the Fire confidence
    if fire_confidence >= FIRE_THRESHOLD:
        label = "🔥 Fire"
        pred_prob = fire_confidence
    else:
        label = "❄️ No Fire"
        # Display the confidence for No Fire for clarity
        pred_prob = probabilities[1] 
        
    return f"{label} (probability: {pred_prob:.4f})"


if __name__ == '__main__':
    # Simple test run (you need a test image here)
    if os.path.exists('test.jpg'):
        result = predict('test.jpg')
        print(f"Prediction result for test.jpg: {result}")
    else:
        print("Note: To test this script, place an image named 'test.jpg' in the current directory.")
