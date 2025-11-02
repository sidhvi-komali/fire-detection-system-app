import tensorflow as tf
import cv2
import numpy as np

# -----------------------------
# 1️⃣ Load model
# -----------------------------
model_path = "models/fire_detection_model.keras"
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully")

model.summary()

# -----------------------------
# 2️⃣ Image variable (replace dynamically later)
# -----------------------------
image_path = "dataset/nofire/0.jpg"  # replace with your image path
img_size = (224, 224)

# -----------------------------
# 3️⃣ Preprocess image (must match training)
# -----------------------------
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Image not found at path: {image_path}")

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to match model input
img = cv2.resize(img, img_size)

# Normalize to [0,1]
img = img / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

# -----------------------------
# 4️⃣ Predict
# -----------------------------
pred_prob = model.predict(img)[0][0]
label = "fire" if pred_prob > 0.5 else "nofire"

# -----------------------------
# 5️⃣ Debug prints
# -----------------------------
print(f"Prediction probability: {pred_prob:.4f}")
print(f"Predicted label: {label}")
