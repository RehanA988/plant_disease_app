import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌿 Plant Disease App")

# load model
model = tf.keras.models.load_model("plant_disease_model.keras")

class_names = ["Healthy", "Diseased"]

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction (THIS IS REQUIRED)
    prediction = model.predict(img)

    st.write(prediction)  # debug

    class_index = np.argmax(prediction)

    st.success(f"Prediction: {class_names[class_index]}")
