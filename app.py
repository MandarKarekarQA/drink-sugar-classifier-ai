import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# ===============================
# ⚙️ CONFIG
# ===============================
MODEL_PATH = "models/drink_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.65  # 🔥 you can adjust this

# ===============================
# 📦 LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping
class_labels = {v: k for k, v in class_indices.items()}

# ===============================
# 🎨 STREAMLIT UI
# ===============================
st.set_page_config(page_title="Drink Classifier", layout="centered")

st.title("🥤 Drink Classifier AI")
st.write("Upload an image of a drink to classify it.")

uploaded_file = st.file_uploader("Upload a drink image", type=["jpg", "jpeg", "png"])

# ===============================
# 🔍 PREDICTION
# ===============================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index])

    predicted_label = class_labels[predicted_index]

    # ===============================
    # 🏷️ FRIENDLY LABELS
    # ===============================
    if predicted_label == "regular_sugary_soft_drinks":
        display_label = "🥤 Regular Sugary Soft Drink"
        message = "Likely regular sugary soft drink. Please verify by checking the nutrition label."
    elif predicted_label == "water_unsweetened_drinks":
        display_label = "💧 Water / Unsweetened Drink"
        message = "Likely water, plain tea, black coffee, or another unsweetened drink."
    else:
        display_label = "🧃 Zero Sugar / Diet Soft Drink"
        message = "Likely zero-sugar or diet soft drink. Please verify the label if needed."

    # ===============================
    # 📊 DISPLAY RESULT
    # ===============================
    st.markdown(f"## Prediction: {display_label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # 🔥 CONFIDENCE RULE
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Not confident. Please verify manually.")
    else:
        if predicted_label == "regular_sugary_soft_drinks":
            st.warning(message)
        elif predicted_label == "water_unsweetened_drinks":
            st.success(message)
        else:
            st.success(message)

    # ===============================
    # 📈 TECHNICAL SCORES
    # ===============================
    with st.expander("Show technical prediction scores"):
        for i, prob in enumerate(predictions):
            label = class_labels[i]

            if label == "regular_sugary_soft_drinks":
                label_name = "🥤 Regular Sugary Soft Drink"
            elif label == "water_unsweetened_drinks":
                label_name = "💧 Water / Unsweetened Drink"
            else:
                label_name = "🧃 Zero Sugar / Diet Soft Drink"

            st.write(f"{label_name}: {prob:.2f}")

    # ===============================
    # ⚠️ DISCLAIMER
    # ===============================
    st.caption(
        "Note: This model predicts from visual appearance only. "
        "It does not read nutrition labels, ingredients, or exact sugar grams."
    )