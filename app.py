import streamlit as st
from PIL import Image

st.title("💄 Shade Lip Cream Recommendation AI")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg","png","jpeg"])

shades = [
    "natural nude",
    "warm peach",
    "soft pink",
    "deep red",
    "brown nude"
]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto kamu", use_column_width=True)

    st.success("Rekomendasi shade kamu:")
    st.write(shades[0])



