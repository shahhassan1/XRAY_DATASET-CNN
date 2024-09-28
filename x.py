import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model (make sure to save your model after training it)
# Example: model.save('xray_model.h5') after training
model_path = 'my_xray_model.keras'
model = load_model(model_path)

# Set up the Streamlit app
st.title("Pneumonia Detection from X-ray")

# Image upload widget
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])

def predict(image_path, model):
    # Load the image with target size (224x224)
    img = load_img(image_path, target_size=(224, 224))
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    
    # Rescale the image (normalize it)
    img_array = img_array / 255.0
    
    # Expand dimensions since the model expects a batch of images
    img_array = np.expand_dims(img_array, axis=0)

    # Get the prediction from the model
    prediction = model.predict(img_array)

    # Return the prediction as "Normal" or "Pneumonia"
    return "Pneumonia" if prediction[0] > 0.5 else "Normal"

if uploaded_file is not None:
    # Save the uploaded file to the local system (optional, you can directly use PIL Image)
    img = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded X-ray image', use_column_width=True)
    
    # Save the image to a temporary location
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predict and display the result
    result = predict("temp.jpg", model)
    st.write(f"Prediction: {result}")
