import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import os

# --- 1️⃣ Configuration ---
dataset_dir = "dataset"
model_dir = "models"
results_dir = os.path.join("results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# EfficientNetB3 configuration
image_size = (300, 300) 
batch_size = 64 # Increased to leverage M3 performance
seed = 123
epochs = 15

# --- 2️⃣ Load datasets ---
train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Dataset loaded. Classes found ({num_classes}): {class_names}")


# --- 3️⃣ Data augmentation ---
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
], name="data_augmentation")


# --- 4️⃣ Build model (Transfer Learning) ---

# Define the path to the downloaded weights file (for explicit loading)
WEIGHTS_PATH = tf.keras.utils.get_file(
    'efficientnetb3_notop.h5',
    'https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5',
    cache_subdir='models',
)

base_model = EfficientNetB3(
    input_shape=image_size + (3,),
    include_top=False,
    # Set weights to None to prevent the automatic load that caused the shape mismatch
    weights=None 
)

# Freeze the base model initially
base_model.trainable = False 

# Explicitly load the weights after model creation to bypass the internal Keras check
try:
    # Use skip_mismatch=True as a safeguard against subtle layer mismatches
    base_model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    print("✅ EfficientNetB3 pre-trained weights loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading weights manually: {e}. Proceeding without pre-trained weights.")


# Create the full model pipeline
inputs = layers.Input(shape=image_size + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x) # EfficientNet specific preprocessing
x = base_model(x, training=False) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
# Two nodes and softmax for robust classification
outputs = layers.Dense(num_classes, activation='softmax')(x) 
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3), # Higher LR for the new head
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("\n--- Model Summary (Frozen Base) ---")
model.summary()


# --- 5️⃣ Initial Training (Head Only) ---
print("\n--- Stage 1: Training Classification Head ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[]
)


# --- 6️⃣ Fine-Tuning Setup ---
base_model.trainable = True # Unfreeze the base model

# Fine-tune only the last 100 layers
fine_tune_at = len(base_model.layers) - 100 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with a very low learning rate (1e-5) for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model Summary (Fine-Tuning) ---")
model.summary()


# --- 7️⃣ Callbacks ---
# Model will be saved as fire_detection_model.keras
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "fire_detection_model.keras"),
    monitor="val_accuracy",
    save_best_only=True
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)


# --- 8️⃣ Fine-Tuning (Full Model) ---
print("\n--- Stage 2: Fine-Tuning the Full Model ---")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)


# --- 9️⃣ Plot and Save Training Curves ---
def plot_and_save_accuracy_and_loss(history, history_fine, results_dir):
    """Plots and saves the combined training history."""
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history.history['accuracy']), color='gray', linestyle='--', label='Fine-Tuning Start')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy Over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(x=len(history.history['loss']), color='gray', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss Over Epochs')
    
    plt.savefig(os.path.join(results_dir, "accuracy_and_loss_over_epochs.png"))
    plt.close()
    print(f"✅ Training curves saved to {results_dir}/accuracy_and_loss_over_epochs.png")
    # Trigger image generation
    print("")


# --- 🔟 Generate and Save Confusion Matrix ---
def plot_and_save_confusion_matrix(model, ds, class_names, results_dir, title="Confusion Matrix"):
    """Computes, plots, and saves the confusion matrix for a dataset."""
    
    # 1. Get true labels and predictions
    true_labels = np.concatenate([y for x, y in ds], axis=0)
    
    # Reset dataset to ensure predictions match labels order
    ds_reset = ds.unbatch().batch(batch_size)
    predictions = model.predict(ds_reset)
    
    # Get the index of the highest probability
    predicted_labels = np.argmax(predictions, axis=1)

    # 2. Compute the matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 3. Plot the matrix
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 4. Save the plot
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    print(f"✅ Confusion Matrix saved to {results_dir}/confusion_matrix.png")
    # Trigger image generation
    print("")

# Execute the plotting functions
plot_and_save_accuracy_and_loss(history, history_fine, results_dir)
# Note: Use the model saved by the checkpoint, which is the best model
best_model = tf.keras.models.load_model(os.path.join(model_dir, "fire_detection_model.keras"))
plot_and_save_confusion_matrix(best_model, val_ds, class_names, results_dir, 
                               title="Validation Confusion Matrix (Best Model)")


# --- Final Output ---
print(f"\nFinal Class Mapping: {class_names}. (Index 0 is {class_names[0]}, Index 1 is {class_names[1]})")
print("---")
print(f"Model saved as: {os.path.join(model_dir, 'fire_detection_model.keras')}")
print(f"Results saved in: {results_dir}")
print("---")
