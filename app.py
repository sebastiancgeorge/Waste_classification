import tempfile
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

yolo_model = YOLO("yolov8n.pt")  # Use YOLOv8 pre-trained model

waste_model = tf.keras.models.load_model("ensemble_waste_classifier.h5")

class_labels = ["Organic", "Recyclable"]

def preprocess_image(image):
    img = image.resize((224, 224))  # Resize
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def classify_real_time():
    st.warning("🔴 **Press 'q' to close the live feed**")
    
    cap = cv2.VideoCapture(0)  # Open webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO Object Detection
        results = yolo_model(rgb_frame)

        for result in results:
            boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])  # Extract box coordinates

                # Extract object image from the frame
                object_img = frame[y1:y2, x1:x2]
                object_img_pil = Image.fromarray(cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB))

                # Classify the object
                img_array = preprocess_image(object_img_pil)
                prediction = waste_model.predict(img_array)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_labels[predicted_class]} ({confidence:.2f}%)"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Real-time Waste Classification", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

st.set_page_config(page_title="Waste Classifier", layout="wide")
st.title("♻ Waste Classification System")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📂 Batch Classification", "🎥 Real-time Detection"])

# Tab 1: Single Image Upload
with tab1:
    st.write("Upload an image of waste to classify it as **Organic** or **Recyclable**.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Show results
        st.markdown(f"### 🏷️ Predicted Class: **{class_labels[predicted_class]}**")
        st.markdown(f"### 🔥 Confidence: **{confidence:.2f}%**")

# Tab 2: Batch Classification
with tab2:
    st.write("Upload multiple images to classify them all at once.")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images = [Image.open(file) for file in uploaded_files]
        results = classify_images(images)

        for img, label, confidence in results:
            st.image(img, caption=f"Class: {label} | Confidence: {confidence:.2f}%", use_column_width=True)

# Tab 3: Real-time Classification
with tab3:
    st.write("Detect and classify waste objects from a live webcam feed.")
    if st.button("🔴 Start Real-time Detection"):
        classify_real_time()