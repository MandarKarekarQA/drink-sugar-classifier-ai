# 🥤 SweetSense AI - Sugary vs Non-Sugary Drink Classifier

## Project Overview

SweetSense AI is a beginner deep learning computer vision project that classifies drink images into:

- Sugary Drink
- Non-Sugary Drink
- Image Not Recognized / Uncertain

The project uses transfer learning with MobileNetV2 and a Streamlit web app.

## Problem Statement

Many drinks look healthy but may contain sugar. This project demonstrates how computer vision can be used to classify drink images based on visual appearance.

## Important Disclaimer

This AI model may make mistakes. Please verify the result manually by checking the drink's nutrition label or ingredients.

The model predicts based on visual appearance only. It does not read sugar content, nutrition labels, or ingredients.

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- Pillow
- NumPy

## Folder Structure

```text
sugary vs non sugary drinks/
│
├── Data/
│   ├── sugary/
│   └── non_sugary/
│
├── Models/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── train_model.py
├── clean_images.py
├── requirements.txt
├── README.md
└── .gitignore