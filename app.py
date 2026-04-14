import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Shade AI", layout="centered")
st.title("💄 Shade AI - Undertone Detection")

uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img = np.array(image)

    st.write("Shape image:", img.shape)

    h, w, _ = img.shape

    # crop wajah aman
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)

    face = img[y1:y2, x1:x2]

    st.write("Face shape:", face.shape)

    if face.size == 0:
        st.error("Crop wajah gagal")
    else:
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)

        hue = hsv[:, :, 0]
        hue = hue[(hue > 5) & (hue < 170)]

        if len(hue) == 0:
            avg_hue = 50
        else:
            avg_hue = hue.mean()

        st.write("Hue:", avg_hue)

        # =====================
        # UNDERTONE DETECTION
        # =====================
        if avg_hue < 25:
            undertone = "Warm"
        elif avg_hue > 95:
            undertone = "Cool"
        else:
            undertone = "Neutral"

        st.success(f"✨ Undertone: {undertone}")

        # =====================
        # RECOMMENDATION
        # =====================
        if undertone == "Warm":
            shades = ["Coral", "Peach", "Terracotta"]
        elif undertone == "Cool":
            shades = ["Rose", "Berry", "Plum"]
        else:
            shades = ["Mauve", "Nude"]

        st.subheader("💄 Rekomendasi Shade")

        for s in shades:
            st.write("💋", s)

        st.info("AI menganalisis warna kulit berdasarkan distribusi HSV wajah")
