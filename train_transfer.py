# =========================================================
# train_transfer.py
# Multiclass Fish Image Classification
# Aggressive Transfer Learning (Evaluation Ready)
# =========================================================

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30               # force learning
DATA_DIR = "data/train"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# DATA GENERATORS
# ---------------------------------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes

print("\n🔹 CLASS INDICES (VERY IMPORTANT)")
print(train_data.class_indices)
print("🔹 NUMBER OF CLASSES:", NUM_CLASSES)

# ---------------------------------------------------------
# CLASS WEIGHTS (CRITICAL FOR IMBALANCE)
# ---------------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("\n🔹 CLASS WEIGHTS")
print(class_weights)

# ---------------------------------------------------------
# LOAD PRETRAINED MODEL
# ---------------------------------------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# ---------------------------------------------------------
# AGGRESSIVE FINE-TUNING
# ---------------------------------------------------------
base_model.trainable = True

# Freeze only very early layers
for layer in base_model.layers[:200]:
    layer.trainable = False

# ---------------------------------------------------------
# CUSTOM CLASSIFIER HEAD (STRONG)
# ---------------------------------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.6)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ---------------------------------------------------------
# COMPILE MODEL (LOW LR FOR STABILITY)
# ---------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------
# CALLBACKS (NO EARLY STOPPING)
# ---------------------------------------------------------
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_transfer_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint]
)

# ---------------------------------------------------------
# SAVE FINAL MODEL
# ---------------------------------------------------------
model.save(os.path.join(MODEL_DIR, "final_transfer_model.h5"))

print("\n✅ TRAINING COMPLETED SUCCESSFULLY")
