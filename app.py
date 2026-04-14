import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Shade AI", layout="centered")

st.title("💄 Shade Recommendation Based on Undertone")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Foto kamu", use_column_width=True)

    # convert ke numpy
    img = np.array(image)

    # crop tengah (area wajah sederhana)
    h, w, _ = img.shape
    face = img[h//4:3*h//4, w//4:3*w//4]

    # convert ke HSV
    face_hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)

    # ambil hue channel
    hue = face_hsv[:,:,0]

    # buang nilai ekstrem (biar lebih stabil)
    hue = hue[(hue > 5) & (hue < 170)]

    # rata-rata hue
    avg_hue = hue.mean()

    st.write(f"🎯 Hue detected: {int(avg_hue)}")

    # klasifikasi undertone (lebih stabil)
    if avg_hue < 25:
        undertone = "Warm"
    elif avg_hue > 95:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    st.success(f"✨ Undertone kamu: {undertone}")

    # rekomendasi shade
    if undertone == "Warm":
        shades = ["Coral", "Peach", "Terracotta", "Orange Nude"]
    elif undertone == "Cool":
        shades = ["Rose", "Berry", "Plum", "Pink"]
    else:
        shades = ["Mauve", "Soft Pink", "Natural Nude", "Dusty Rose"]

    # warna visual
    colors = {
        "Coral": "#FF7F50",
        "Peach": "#FFCBA4",
        "Terracotta": "#E2725B",
        "Orange Nude": "#D2691E",
        "Rose": "#FF007F",
        "Berry": "#8A2BE2",
        "Plum": "#8E4585",
        "Pink": "#FFC0CB",
        "Mauve": "#E0B0FF",
        "Soft Pink": "#F4C2C2",
        "Natural Nude": "#D2B48C",
        "Dusty Rose": "#DCAE96"
    }

    st.subheader("💄 Rekomendasi Shade:")

    for s in shades:
        st.markdown(
            f"<div style='display:flex;align-items:center;margin-bottom:5px;'>"
            f"<div style='width:25px;height:25px;background:{colors[s]};margin-right:10px;border-radius:5px;'></div>"
            f"{s}</div>",
            unsafe_allow_html=True
        )
