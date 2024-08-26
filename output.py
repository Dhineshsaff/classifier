import streamlit as st
import os
import cv2
import numpy as np
import joblib
from PIL import Image

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = img.flatten()  # Flatten the image
    img = img.reshape(1, -1)  # Reshape for prediction
    return img

# Function to dynamically get class names from the dataset folder
def get_class_names(dataset_path):
    class_names = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    return class_names

# Load the trained model
model = joblib.load('new_svm_model.pkl')

# Get class names from the dataset folder
dataset_path = r"C:\Users\Mazveen\project\dataset"
class_names = get_class_names(dataset_path)

# Streamlit app title
st.title("TrashNet Image Classification")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the image in the app
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = load_and_preprocess_image(image)

    # Make a prediction
    prediction = model.predict(processed_image)

    # Display the prediction with the corresponding class name
    predicted_class_name = prediction[0]  # Directly use the predicted label
    st.write(f"Predicted Class: **{predicted_class_name}**")
