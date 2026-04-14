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
    avg_color = img.mean(axis=(0,1))
    r, g, b = avg_color

    st.write(f"RGB: R={int(r)}, G={int(g)}, B={int(b)}")

    # logika undertone (lebih realistis)
    if r > g and r > b:
        undertone = "Warm"
    elif b > r and b > g:
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
