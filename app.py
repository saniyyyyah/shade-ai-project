import streamlit as st
from PIL import Image
import numpy as np

st.title("💄 Shade Lip Cream Recommendation AI")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg","png","jpeg"])

shades = [
    "deep red",
    "brown nude",
    "warm peach",
    "soft pink",
    "natural nude"
]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto kamu", use_column_width=True)

    # ubah ke array
    img = np.array(image)

    # ambil rata-rata warna (brightness)
    brightness = img.mean()

    # logika sederhana
    if brightness < 80:
        result = shades[0]
    elif brightness < 120:
        result = shades[1]
    elif brightness < 160:
        result = shades[2]
    elif brightness < 200:
        result = shades[3]
    else:
        result = shades[4]

    st.success("Rekomendasi shade kamu:")
    st.write(result)



