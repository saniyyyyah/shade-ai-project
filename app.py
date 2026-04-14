import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

st.title("💄 Shade Recommendation + DINOv2 AI")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg","png","jpeg"])

# load DINOv2 model (sekali saja)
@st.cache_resource
def load_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Foto kamu", use_column_width=True)

    img = np.array(image)

    # crop wajah sederhana
    h, w, _ = img.shape
    face = img[h//4:3*h//4, w//4:3*w//4]

    # =====================
    # HSV UNDERTONE (lama kamu)
    # =====================
    face_hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    hue = face_hsv[:,:,0]
    hue = hue[(hue > 5) & (hue < 170)]
    avg_hue = hue.mean()

    if avg_hue < 25:
        undertone = "Warm"
    elif avg_hue > 95:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    st.success(f"Undertone: {undertone}")

    # =====================
    # DINOv2 FEATURE
    # =====================
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(face).unsqueeze(0)

    with torch.no_grad():
        features = model(input_tensor)

    # normalisasi
    features = features / features.norm(dim=-1, keepdim=True)

    st.subheader("🧠 Fitur wajah (DINOv2)")
    st.write(features[0][:10])  # tampilkan sebagian

    # =====================
    # REKOMENDASI SHADE
    # =====================
    if undertone == "Warm":
        shades = ["Coral", "Peach", "Terracotta"]
    elif undertone == "Cool":
        shades = ["Rose", "Berry", "Plum"]
    else:
        shades = ["Mauve", "Soft Pink", "Nude"]

    st.subheader("💄 Rekomendasi Shade")
    for s in shades:
        st.write("💄", s)
