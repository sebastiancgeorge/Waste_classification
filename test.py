# Set Streamlit page config first, before any other Streamlit commands
import streamlit as st
st.set_page_config(page_title="Waste Classifier", layout="wide")

# Import libraries
import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Constants
CLASS_LABELS = ["Recyclable Waste", "Organic Waste"]
REPO_ID = "sebastiancgeorge/ensembled_waste_classification"
MODEL_FILENAME = "ensemble_waste_classifier.keras"

@st.cache_resource
def load_models():
    """Load models from HuggingFace and local storage"""
    try:
        # Download waste classifier from HuggingFace
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        waste_model = tf.keras.models.load_model(model_path)

        # Load YOLO model for object detection
        yolo_model = YOLO("yolov8n.pt")
        
        return waste_model, yolo_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model input"""
    img = image.resize((224, 224))  # Resize for model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Expand dims
    return img_array

def predict_func(img, model):
    """Predict waste classification"""
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    
    img_resized = cv2.resize(img, (224, 224)) / 255.0  # Resize & normalize
    img_resized = np.reshape(img_resized, (1, 224, 224, 3)).astype(np.float32)  # Expand dims

    prediction = model.predict(img_resized)
    class_idx = np.argmax(prediction, axis=-1)[0]

    print(f"🟢 Prediction: {CLASS_LABELS[class_idx]}")
    plt.show()

    return CLASS_LABELS[class_idx], np.max(prediction) * 100  # Return label & confidence

def create_confidence_visualization(prediction):
    """Create confidence score visualization"""
    fig, ax = plt.subplots(figsize=(8, 3))
    confidence_scores = prediction[0] * 100
    
    colors = ['#2ecc71' if confidence_scores[0] > confidence_scores[1] else '#e74c3c',
              '#e74c3c' if confidence_scores[0] > confidence_scores[1] else '#2ecc71']
    
    ax.bar(CLASS_LABELS, confidence_scores, color=colors)
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Classification Confidence')
    
    # Add value labels on bars
    for i, v in enumerate(confidence_scores):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    return fig

def process_single_image(image, waste_model, yolo_model):
    """Process a single image with both models"""
    try:
        # Basic classification
        img_array = preprocess_image(image)
        prediction = waste_model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Object detection with YOLO
        results = yolo_model(np.array(image))
        annotated_image = np.array(image).copy()
        
        detected_objects = []
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                object_img = annotated_image[y1:y2, x1:x2]
                if object_img.size > 0:  # Check if valid object was detected
                    object_img_pil = Image.fromarray(object_img)
                    obj_prediction = waste_model.predict(preprocess_image(object_img_pil))
                    obj_class = np.argmax(obj_prediction)
                    obj_conf = np.max(obj_prediction) * 100
                    
                    # Draw box and label
                    color = (0, 255, 0) if obj_class == 0 else (255, 0, 0)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    label = f"{CLASS_LABELS[obj_class]} ({obj_conf:.1f}%)"
                    cv2.putText(annotated_image, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    detected_objects.append({
                        'class': CLASS_LABELS[obj_class],
                        'confidence': obj_conf,
                        'location': (x1, y1, x2, y2)
                    })
        
        return {
            'prediction': prediction,
            'class': CLASS_LABELS[predicted_class],
            'confidence': confidence,
            'annotated_image': Image.fromarray(annotated_image),
            'detected_objects': detected_objects
        }
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Main app
st.title("♻️ Waste Classification System")

# Load models
if not st.session_state.models_loaded:
    with st.spinner("Loading models... Please wait."):
        waste_model, yolo_model = load_models()
        if waste_model is not None and yolo_model is not None:
            st.session_state.models_loaded = True
            st.session_state.waste_model = waste_model
            st.session_state.yolo_model = yolo_model
        else:
            st.error("Failed to load models. Please refresh the page.")
            st.stop()

# Create tabs
tab1, tab2 = st.tabs(["📷 Single Image", "📂 Batch Processing"])

# Single Image Tab
with tab1:
    st.write("Upload an image to classify it as **Recyclable Waste** or **Organic Waste**.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        result = process_single_image(image, st.session_state.waste_model, 
                                       st.session_state.yolo_model)
        
        if result:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(result['annotated_image'], caption="Analyzed Image", 
                        use_column_width=True)
            
            with col2:
                st.markdown(f"### Overall Classification")
                st.markdown(f"**Class:** {result['class']}")
                st.markdown(f"**Confidence:** {result['confidence']:.1f}%")
                
                if result['detected_objects']:
                    st.markdown("### Detected Objects")
                    for i, obj in enumerate(result['detected_objects'], 1):
                        st.markdown(f"""
                        **Object {i}:**
                        - Class: {obj['class']}
                        - Confidence: {obj['confidence']:.1f}%
                        """)

            st.subheader("Confidence Distribution")
            conf_fig = create_confidence_visualization(result['prediction'])
            st.pyplot(conf_fig)

# Batch Processing Tab
with tab2:
    st.write("Upload multiple images to process them all at once.")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file)
            process_single_image(image, st.session_state.waste_model, 
                                 st.session_state.yolo_model)
