import streamlit as st
import numpy as np
import tensorflow as tf
import requests
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Load the CNN model from Hugging Face
@st.cache_resource
def load_model():
    model_path = f"https://huggingface.co/sebastiancgeorge/ensembled_waste_classification/blob/main/cnn_model.keras"
    response = requests.get(model_path)
    model_bytes = BytesIO(response.content)
    model = tf.keras.models.load_model(model_bytes)
    return model

# Load the model once
model = load_model()

# Define class labels
CLASS_LABELS = ["Recyclable Waste", "Organic Waste"]

# Image Preprocessing Function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for the model
    image = np.array(image) / 255.0  # Normalize
    image = np.reshape(image, (1, 224, 224, 3)).astype(np.float32)
    return image

# Prediction Function
def predict_waste(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    class_idx = np.argmax(prediction, axis=-1)[0]
    confidence = float(np.max(prediction))
    return CLASS_LABELS[class_idx], confidence

# Streamlit UI
st.title("♻️ Waste Classification App")
st.write("Upload an image to classify it as **Recyclable Waste** or **Organic Waste**.")

# Upload File Section
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make Prediction
    if st.button("Classify Waste"):
        label, confidence = predict_waste(image)
        
        # Display Result
        st.success(f"🟢 **Prediction:** {label}")
        st.write(f"🔢 **Confidence:** {confidence:.2%}")

        # Show Confidence Bar
        st.progress(int(confidence * 100))
