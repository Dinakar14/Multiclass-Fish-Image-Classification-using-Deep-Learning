# =====================================
# app.py
# Multiclass Fish Image Classification
# Streamlit Deployment
# =====================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "models/best_transfer_model.h5"
DATA_DIR = "data/train"   # IMPORTANT

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Load Class Names SAFELY
# -----------------------------
class_names = sorted(
    [folder for folder in os.listdir(DATA_DIR)
     if os.path.isdir(os.path.join(DATA_DIR, folder))]
)

NUM_CLASSES = len(class_names)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image to predict its category")

uploaded_file = st.file_uploader(
    "Choose a fish image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # -----------------------------
    # SAFE CHECK (NO MORE INDEX ERROR)
    # -----------------------------
    if predicted_class < NUM_CLASSES:
        st.success(f"Prediction: {class_names[predicted_class]}")
        st.info(f"Confidence: {confidence:.2f}")
    else:
        st.error("Prediction index out of range. Check class mapping.")
