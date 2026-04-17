import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("💄 Shade AI TEST")

uploaded_file = st.file_uploader("Upload foto", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    img = np.array(image)
    h, w, _ = img.shape

    face = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    if face.size == 0:
        st.error("Crop gagal")
    else:
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        hue = hsv[:,:,0].mean()

        if hue < 25:
            undertone = "Warm"
        elif hue > 95:
            undertone = "Cool"
        else:
            undertone = "Neutral"

        st.success(f"Undertone: {undertone}")

        # hasil dummy dulu
        st.write("💄 Coral")
        st.write("💄 Peach")
        st.write("💄 Nude")
