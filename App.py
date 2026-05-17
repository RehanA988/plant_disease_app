import streamlit as st

import numpy as np
from PIL import Image

st.title("🌿 Plant Disease Detection App")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.keras")
    return model

model = load_model()

class_names = ["Healthy", "Diseased"]

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    st.success(f"Prediction: {class_names[class_index]}")
