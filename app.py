import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('traffic.keras')

model = load_model()

def predict(image):
    # Preprocess the image
    image = image.resize((224, 224))  # Adjust size according to your model's requirement
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return str(predicted_class[0])  # Convert to string for display

def main():
    st.title("Traffic Signal Classifier")

    st.write("Upload an image of a traffic signal to classify it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make prediction
        prediction = predict(image)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
