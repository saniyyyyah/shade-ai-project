import streamlit as st
from PIL import Image
import numpy as np

st.title("💄 Shade Recommendation Based on Undertone")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto kamu", use_column_width=True)

    # ubah ke array
    img = np.array(image)

# crop tengah (area wajah sederhana)
h, w, _ = img.shape
face = img[h//4:3*h//4, w//4:3*w//4]

# pakai area wajah
avg_color = face.mean(axis=(0,1))
r, g, b = avg_color

    # ambil rata-rata warna RGB
    avg_color = img.mean(axis=(0,1))
    r, g, b = avg_color

    st.write(f"RGB: R={int(r)}, G={int(g)}, B={int(b)}")

    # hitung selisih warna
    rg = r - g
    rb = r - b

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
