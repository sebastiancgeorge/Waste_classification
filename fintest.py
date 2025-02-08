import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import requests
from huggingface_hub import hf_hub_download
from PIL import Image
import tempfile
import os

# Define class labels
CLASS_LABELS = ["Recyclable Waste", "Organic Waste"]

# Load the model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="sebastiancgeorge/ensembled_waste_classification",
        filename="cnn_model.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model once
model = load_model()

# Function to preprocess an image
def preprocess_image(image):
    image = image.convert('RGB')  # Ensure RGB format
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to classify an image
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]
    return predictions

# Streamlit UI
st.title("♻️ Waste Classification App")
st.write("Classify waste as **Recyclable** or **Organic** using a CNN model.")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📷 Image Classification", "📂 Batch Classification", "📹 Real-time Camera"])

# **Tab 1: Image Classification**
with tab1:
    st.header("📷 Classify a Single Image")

    option = st.radio("Choose Input Type:", ["Upload Image", "Image URL", "Use Camera"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
    
    elif option == "Image URL":
        image_url = st.text_input("Enter Image URL:")
        if image_url:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                image = Image.open(response.raw)
            else:
                st.error("Invalid URL. Please try again.")

    elif option == "Use Camera":
        camera_image = st.camera_input("Capture an image")
        if camera_image:
            image = Image.open(camera_image)

    # Perform classification if an image is provided
    if "image" in locals():
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predictions = classify_image(image)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]

        # Display Prediction
        st.write(f"### 🏷️ Prediction: **{predicted_class}**")
        st.write("### 🔎 Confidence Scores:")
        for label, confidence in zip(CLASS_LABELS, predictions):
            st.write(f"- **{label}:** {confidence:.2%}")

# **Tab 2: Batch Classification**
with tab2:
    st.header("📂 Classify Multiple Images")
    uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            predictions = classify_image(image)
            predicted_class = CLASS_LABELS[np.argmax(predictions)]

            # Display Prediction
            st.write(f"### 🏷️ Prediction: **{predicted_class}**")
            for label, confidence in zip(CLASS_LABELS, predictions):
                st.write(f"- **{label}:** {confidence:.2%}")
            st.markdown("---")

# **Tab 3: Real-time Camera Classification**
with tab3:
    st.header("📹 Real-time Waste Classification")

    # Start Webcam
    cap = cv2.VideoCapture(0)

    if st.button("Start Camera"):
        st.write("📸 Capturing real-time frames...")
        frame_window = st.image([])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Classify frame
            predictions = classify_image(image)
            predicted_class = CLASS_LABELS[np.argmax(predictions)]

            # Display the image and prediction
            frame_window.image(image, caption=f"🏷️ Prediction: {predicted_class}")

    if st.button("Stop Camera"):
        cap.release()
        st.write("Camera stopped.")
