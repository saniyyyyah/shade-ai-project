import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip
import torchvision.transforms as transforms
import timm

st.title("💄 Shade AI (STABLE FINAL)")

uploaded_file = st.file_uploader("Upload foto", type=["jpg","png","jpeg"])

# =====================
# LOAD MODELS (RINGAN)
# =====================
@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'RN50',  # 🔥 ringan
        pretrained='openai'
    )
    model.eval()
    return model, preprocess

@st.cache_resource
def load_dino():
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
    model.eval()
    return model

clip_model, clip_preprocess = load_clip()
dino_model = load_dino()

# =====================
# MAIN
# =====================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    img = np.array(image)
    h, w, _ = img.shape

    face = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    if face.size == 0:
        st.error("Crop gagal")
        st.stop()

    # =====================
    # UNDERTONE
    # =====================
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0].mean()

    if hue < 25:
        undertone = "Warm"
    elif hue > 95:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    st.success(f"✨ Undertone: {undertone}")

    # =====================
    # SHADE
    # =====================
    shade_map = {
        "Warm": ["coral lipstick", "peach lipstick", "terracotta lipstick"],
        "Cool": ["pink lipstick", "berry lipstick", "plum lipstick"],
        "Neutral": ["nude lipstick", "mauve lipstick", "rose lipstick"]
    }

    shade_texts = shade_map[undertone]

    # =====================
    # CLIP
    # =====================
    image_input = clip_preprocess(Image.fromarray(face)).unsqueeze(0)
    text_tokens = open_clip.tokenize(shade_texts)

    with torch.no_grad():
        img_feat = clip_model.encode_image(image_input)
        txt_feat = clip_model.encode_text(text_tokens)

    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    clip_sim = img_feat @ txt_feat.T

    # =====================
    # DINO
    # =====================
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dino_input = transform(face).unsqueeze(0)

    with torch.no_grad():
        dino_feat = dino_model(dino_input)

    dino_feat = dino_feat / dino_feat.norm(dim=-1, keepdim=True)
    dino_weight = float(torch.sigmoid(dino_feat.mean()))

    # =====================
    # FUSION
    # =====================
    final = clip_sim * (1 + 0.2 * dino_weight)
    final = (final * 10).softmax(dim=-1)

    probs, idxs = final[0].topk(3)

    # =====================
    # OUTPUT
    # =====================
    st.subheader("💄 Rekomendasi Shade")

    for i in range(3):
        st.write(f"{shade_texts[idxs[i]]} — {probs[i].item()*100:.1f}%")

    st.caption("⚠️ Diffusion dinonaktifkan di web (keterbatasan server)")
