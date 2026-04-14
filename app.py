import streamlit as st
from PIL import Image
import numpy as np

st.title("💄 Shade Recommendation Based on Undertone")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto kamu", use_column_width=True)

    img = np.array(image)

    # ambil rata-rata warna
   # selisih warna
rg = r - g
rb = r - b
gb = g - b

# tentukan undertone
if rg > 10 and rb > 10:
    undertone = "Warm"
elif rb < -10:
    undertone = "Cool"
else:
    undertone = "Neutral"
    st.subheader(f"Undertone kamu: {undertone}")

    # rekomendasi shade
    if undertone == "Warm":
        shades = ["Coral", "Peach", "Warm Nude", "Terracotta"]
    elif undertone == "Cool":
        shades = ["Rose", "Berry", "Pink", "Plum"]
    else:
        shades = ["Mauve", "Soft Pink", "Natural Nude", "Dusty Rose"]

    st.success("Rekomendasi Shade:")
    for s in shades:
        st.write("💄", s)
