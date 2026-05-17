import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Plant Disease App")

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to continue")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Fake prediction (temporary)
    st.markdown("### 🧠 Result")
    st.success("Prediction system abhi connect nahi hai")
    st.info("Next step: TensorFlow model add karna hoga")
