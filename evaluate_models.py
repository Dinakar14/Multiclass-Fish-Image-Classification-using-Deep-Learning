# =========================================
# evaluate_models.py
# Multiclass Fish Image Classification
# Model Evaluation Script
# =========================================

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# ------------------------------
# Configuration
# ------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = "data/test"
MODEL_PATH = "models/best_transfer_model.h5"

# ------------------------------
# Check model exists
# ------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Train the model first."
    )

# ------------------------------
# Load Model
# ------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ------------------------------
# Test Data Generator
# ------------------------------
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_data.class_indices.keys())
print("Classes:", class_names)

# ------------------------------
# Predictions
# ------------------------------
predictions = model.predict(test_data, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# ------------------------------
# Evaluation Metrics
# ------------------------------
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ------------------------------
# Optional: Save Metrics to File
# ------------------------------
with open("model_evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_true, y_pred)))

print("\n📁 Evaluation results saved to model_evaluation.txt")
print("🎯 Model evaluation completed successfully!")
