import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json
import os

# Paths
DATA_DIR = "Data"
MODEL_DIR = "Models"

# Create Models folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ===============================
# 🔥 TRAIN DATA (WITH AUGMENTATION)
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        # rotate images
    zoom_range=0.2,           # zoom in/out
    horizontal_flip=True,     # flip image
    shear_range=0.1,          # slight distortion
    width_shift_range=0.1,    # move left/right
    height_shift_range=0.1,   # move up/down
    validation_split=0.2      # 80 train / 20 validation
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ===============================
# 🧠 MODEL
# ===============================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(train_generator.num_classes, activation='softmax')
])

# ===============================
# ⚙️ COMPILE
# ===============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 🚀 TRAIN
# ===============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12
)

# ===============================
# 💾 SAVE MODEL
# ===============================
model.save(os.path.join(MODEL_DIR, "drink_model.keras"))

# Save class names
with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ Model and class names saved successfully!")