import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask_rcnn import DentalCariesMaskRCNN
from models.recommendation import DentalRecommendationSystem
from utils.evaluation import DentalEvaluator
from configs.model_config import ModelConfig

def load_models(config):
    """Load the detection and recommendation models"""
    # Load detection model
    detection_model = DentalCariesMaskRCNN(
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim
    )
    
    # Load saved weights if available
    model_path = Path('checkpoints/best_model.pth')
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        detection_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load recommendation model
    recommendation_model = DentalRecommendationSystem(num_labels=3)
    
    return detection_model, recommendation_model

def process_image(image, detection_model, evaluator):
    """Process the uploaded image and return predictions"""
    # Convert PIL Image to tensor
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # Make prediction
    with torch.no_grad():
        prediction = detection_model.evaluate_step([image_tensor])[0]
    
    return prediction

def main():
    st.set_page_config(page_title="Dental Caries Detection System", layout="wide")
    
    # Title and description
    st.title("Dental Caries Detection System")
    st.markdown("""
    This system detects and analyzes dental caries in X-ray images using deep learning.
    Upload an X-ray image to get:
    - Caries detection and segmentation
    - Severity classification
    - Personalized recommendations
    """)
    
    # Load models
    config = ModelConfig()
    detection_model, recommendation_model = load_models(config)
    evaluator = DentalEvaluator(num_classes=config.num_classes)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a dental X-ray image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Process image
        prediction = process_image(image, detection_model, evaluator)
        
        # Display predictions
        with col2:
            st.subheader("Detection Results")
            fig = evaluator.visualize_predictions(
                image=torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0,
                prediction=prediction,
                score_threshold=0.5
            )
            st.pyplot(fig)
        
        # Patient history form
        st.subheader("Patient History")
        with st.form("patient_history"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                brushing_frequency = st.number_input(
                    "Daily brushing frequency",
                    min_value=0,
                    max_value=5,
                    value=2
                )
            
            with col2:
                uses_fluoride = st.checkbox("Uses fluoride toothpaste", value=True)
            
            with col3:
                sugar_consumption = st.selectbox(
                    "Sugar consumption level",
                    options=['low', 'moderate', 'high'],
                    index=1
                )
            
            submitted = st.form_submit_button("Get Recommendations")
        
        if submitted:
            # Prepare patient history
            patient_history = {
                'brushing_frequency': brushing_frequency,
                'uses_fluoride': uses_fluoride,
                'sugar_consumption': sugar_consumption
            }
            
            # Get severity scores (example - you would need to implement this based on your model's output)
            severity_scores = torch.tensor([0.2, 0.5, 0.3])  # Example scores
            
            # Get recommendations
            recommendations = recommendation_model.get_recommendations(
                severity_scores,
                patient_history
            )
            
            # Display recommendations
            st.subheader("Personalized Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Display detection metrics
            st.subheader("Detection Metrics")
            num_caries = len(prediction['boxes'])
            avg_confidence = prediction['scores'].mean().item()
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Number of Caries Detected", num_caries)
            with metrics_col2:
                st.metric("Average Confidence", f"{avg_confidence:.2%}")

if __name__ == "__main__":
    main() 