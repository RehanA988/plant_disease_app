import streamlit as st
from PIL import Image

st.title("🌿 Plant Disease App (Test Mode)")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)
    st.success("App working fine ✅")
