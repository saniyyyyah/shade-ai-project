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

    # ambil rata-rata warna dari area wajah
    avg_color = face.mean(axis=(0,1))
    r, g, b = avg_color

    st.write(f"RGB: R={int(r)}, G={int(g)}, B={int(b)}")

    # hitung selisih warna
    # normalisasi RGB (biar tidak tergantung lighting)
total = r + g + b
r_norm = r / total
g_norm = g / total
b_norm = b / total

# logika undertone
if r_norm > g_norm and r_norm > b_norm:
    undertone = "Warm"
elif b_norm > r_norm:
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

    # warna visual
    colors = {
        "Coral": "#FF7F50",
        "Peach": "#FFCBA4",
        "Warm Nude": "#C68642",
        "Terracotta": "#E2725B",
        "Rose": "#FF007F",
        "Berry": "#8A2BE2",
        "Pink": "#FFC0CB",
        "Plum": "#8E4585",
        "Mauve": "#E0B0FF",
        "Soft Pink": "#F4C2C2",
        "Natural Nude": "#D2B48C",
        "Dusty Rose": "#DCAE96"
    }

    st.success("Rekomendasi Shade:")

    for s in shades:
        st.markdown(
            f"<div style='display:flex;align-items:center;'>"
            f"<div style='width:20px;height:20px;background:{colors[s]};margin-right:10px;'></div>"
            f"{s}</div>",
            unsafe_allow_html=True
        )
