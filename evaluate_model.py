import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths
# -----------------------------
model_path = "models/fire_detection_model.keras"
dataset_dir = "dataset"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)  # create results folder if missing

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully")

# -----------------------------
# Load validation dataset
# -----------------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224,224),
    batch_size=32
)

# Normalize images
val_ds = val_ds.map(lambda x, y: (x/255.0, y))

# -----------------------------
# Extract true labels and predictions
# -----------------------------
y_true = np.concatenate([y.numpy() for x, y in val_ds])
y_pred_probs = model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# -----------------------------
# Class names
# -----------------------------
class_names = sorted(next(os.walk(dataset_dir))[1])
print("Classes:", class_names)

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
cm_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"✅ Confusion matrix graph saved at {cm_path}")

# -----------------------------
# Classification report
# -----------------------------
report = classification_report(y_true, y_pred, target_names=class_names)
report_path = os.path.join(results_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"✅ Classification report saved at {report_path}")

# -----------------------------
# Accuracy
# -----------------------------
accuracy = np.mean(y_true == y_pred)
accuracy_path = os.path.join(results_dir, "accuracy.txt")
with open(accuracy_path, "w") as f:
    f.write(f"Validation Accuracy: {accuracy:.4f}\n")
print(f"✅ Accuracy saved at {accuracy_path}")

print("🎉 Evaluation complete. All results are in the results/ folder.")
