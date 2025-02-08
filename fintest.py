import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from PIL import Image

# Define class labels
CLASS_LABELS = ["Organic Waste", "Recyclable Waste"]

# Load the model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="sebastiancgeorge/ensembled_waste_classification",
        filename="cnn_model.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

# Load model once
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

# Function to display confidence scores with colors
def display_confidence_scores(predictions, predicted_class):
    st.write("### 🔎 Confidence Scores:")
    for label, confidence in zip(CLASS_LABELS, predictions):
        color = "green" if label == predicted_class else "red"
        icon = "✅" if label == predicted_class else "❌"
        st.markdown(f"<p style='color:{color}; font-size:20px;'>{icon} {label}: {confidence:.2%}</p>", unsafe_allow_html=True)

# Function to show About info
def show_about():
    st.markdown("""
    ## ℹ️ About This Project
    - This waste classification model is an **ensembled CNN** trained on **organic and recyclable waste images**.
    - **Purpose:** To assist in **automated waste sorting** for better recycling practices.
    - **Model Approach:** It combines multiple CNN architectures to improve accuracy and generalization.
    - **Dataset:** Trained using various publicly available waste classification datasets.
    - **Developer:** [Your Name or Organization]
    """)

# Streamlit UI
st.title("♻️ Waste Classification App")
st.write("Classify waste as **Recyclable** or **Organic** using an ensembled CNN model.")

# **Settings Menu (Three-Dots Button)**
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if st.button("🔄 Rerun App"):
        st.experimental_rerun()

    if st.button("ℹ️ About"):
        show_about()

# Create Tabs
tab1, tab2 = st.tabs(["📷 Single Image Classification", "📂 Batch Classification"])

# **Tab 1: Single Image Classification**
with tab1:
    st.header("📷 Classify a Single Image")

    # Separate tabs for each input type
    tab1_1, tab1_2, tab1_3 = st.tabs(["📂 Upload Image", "🌍 Image URL", "📸 Use Camera"])

    image = None  # Placeholder for selected image

    # **Option 1: Upload Image**
    with tab1_1:
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    # **Option 2: Image URL**
    with tab1_2:
        image_url = st.text_input("Enter Image URL:")
        if image_url:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                image = Image.open(response.raw)
            else:
                st.error("Invalid URL. Please try again.")

    # **Option 3: Use Camera**
    with tab1_3:
        camera_image = st.camera_input("Capture an image")
        if camera_image:
            image = Image.open(camera_image)

    # Perform classification if an image is provided
    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        predictions = classify_image(image)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]

        # Display Prediction
        st.write(f"### 🏷️ Prediction: **{predicted_class}**")
        display_confidence_scores(predictions, predicted_class)

# **Tab 2: Batch Classification**
with tab2:
    st.header("📂 Classify Multiple Images")
    uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        organic_count = 0
        recyclable_count = 0
        total_images = len(uploaded_files)

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            predictions = classify_image(image)
            predicted_class = CLASS_LABELS[np.argmax(predictions)]

            # Count category occurrences
            if predicted_class == "Organic Waste":
                organic_count += 1
            else:
                recyclable_count += 1

            # Display Prediction
            st.write(f"### 🏷️ Prediction: **{predicted_class}**")
            display_confidence_scores(predictions, predicted_class)
            st.markdown("---")

        # Display Batch Classification Report
        st.write("## 📊 Classification Report")
        st.write(f"### 🟢 Organic Waste: {organic_count} ({(organic_count/total_images) * 100:.2f}%)")
        st.write(f"### 🔵 Recyclable Waste: {recyclable_count} ({(recyclable_count/total_images) * 100:.2f}%)")
