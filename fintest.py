import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from PIL import Image

# Define model repo details
HF_REPO_ID = "sebastiancgeorge/ensembled_waste_classification"
MODEL_FILENAME = "cnn_model.keras"

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
CLASS_LABELS = ["Recyclable Waste", "Organic Waste"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size to model's expected input
    image = np.array(image) / 255.0   # Normalize
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.expand_dims(image, axis=0)

# Function to classify an image
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]
    return predictions

# Function to display confidence visualization
def plot_confidence(predictions):
    fig, ax = plt.subplots()
    ax.bar(CLASS_LABELS, predictions, color=["blue", "green"])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    st.pyplot(fig)

# Streamlit UI
st.title("Waste Classification App")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Image Classification", "Batch Classification", "Camera Classification"])

# **Tab 1: Image Classification**
with tab1:
    st.header("Upload an Image for Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        predictions = classify_image(image)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]
        
        st.write(f"### Prediction: {predicted_class} ({predictions[np.argmax(predictions)]:.2f} confidence)")
        plot_confidence(predictions)

# **Tab 2: Batch Classification**
with tab2:
    st.header("Upload Multiple Images for Batch Classification")
    uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            
            predictions = classify_image(image)
            predicted_class = CLASS_LABELS[np.argmax(predictions)]
            
            st.write(f"**{uploaded_file.name}:** {predicted_class} ({predictions[np.argmax(predictions)]:.2f} confidence)")
            plot_confidence(predictions)

# **Tab 3: Camera-based Classification**
with tab3:
    st.header("Classify Waste Using Your Camera")
    cam_image = st.camera_input("Take a photo")

    if cam_image:
        image = Image.open(cam_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        predictions = classify_image(image)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]
        
        st.write(f"### Prediction: {predicted_class} ({predictions[np.argmax(predictions)]:.2f} confidence)")
        plot_confidence(predictions)
