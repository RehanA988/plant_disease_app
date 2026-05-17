

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load Model
from tensorflow.keras.applications import efficientnet

model = tf.keras.models.load_model(

    "/content/drive/MyDrive/plant_disease_model.keras",

    custom_objects={

        'preprocess_input': efficientnet.preprocess_input

    },

    compile=False,

    safe_mode=False

)

# Class Names
class_names = [

    "Apple_Apple Scab",
    "Apple_Black Rot",
    "Apple_Cedar Apple Rust",
    "Apple_Healthy",
    "Cherry_Healthy",
    "Cherry_Powdery Mildew",
    "Corn_Common Rust",
    "Corn_Gray Leaf Spot",
    "Corn_Healthy",
    "Corn_Northern Leaf Blight",
    "Grape_Black Rot",
    "Grape_Esca",
    "Grape_Healthy",
    "Grape_Leaf Blight",
    "Peach_Bacterial Spot",
    "Peach_Healthy",
    "Pepper_Bacterial Spot",
    "Pepper_Healthy",
    "Potato_Early Blight",
    "Potato_Healthy",
    "Potato_Late Blight",
    "Strawberry_Healthy",
    "Strawberry_Leaf Scorch",
    "Tomato_Bacterial Spot",
    "Tomato_Early Blight",
    "Tomato_Healthy",
    "Tomato_Late Blight"

]

# Title
st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader(

    "Upload Leaf Image",

    type=["jpg","jpeg","png"]

)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image)

    image = image.resize((224,224))

    img_array = np.array(image)

    img_array = np.expand_dims(

        img_array,

        axis=0

    )

    # EfficientNet Preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(

        img_array

    )

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(

        f"Prediction: {predicted_class}"

    )

    st.info(

        f"Confidence: {confidence:.2f}%"

    )
