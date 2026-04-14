import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("💄 Shade AI")

uploaded_file = st.file_uploader("Upload foto", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img = np.array(image)

    # debug ukuran gambar
    st.write("Shape image:", img.shape)

    h, w, _ = img.shape

    # crop aman (hindari kosong)
    face = img[max(0, h//6):min(h, h//1.5),
               max(0, w//6):min(w, w//1.5)]

    st.write("Face shape:", face.shape)

    if face.size == 0:
        st.error("Crop wajah gagal")
    else:
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)

        hue = hsv[:, :, 0].mean()

        st.write("Hue:", hue)

        # logika undertone
        if hue < 25:
            undertone = "Warm"
        elif hue > 95:
            undertone = "Cool"
        else:
            undertone = "Neutral"

        st.success(f"Undertone: {undertone}")

        # rekomendasi
        if undertone == "Warm":
            shades = ["Coral", "Peach", "Terracotta"]
        elif undertone == "Cool":
            shades = ["Rose", "Berry", "Plum"]
        else:
            shades = ["Mauve", "Nude"]

        st.subheader("Rekomendasi Shade")

        for s in shades:
            st.write("💄", s)
