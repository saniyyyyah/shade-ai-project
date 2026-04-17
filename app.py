import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip

st.title("💄 Shade AI (HSV + CLIP)")

uploaded_file = st.file_uploader("Upload foto", type=["jpg","png","jpeg"])

# load CLIP
@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    model.eval()
    return model, preprocess

clip_model, clip_preprocess = load_clip()

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
    # HSV UNDERTONE
    # =====================
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0].mean()

    if hue < 25:
        undertone = "Warm"
    elif hue > 95:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    st.success(f"Undertone: {undertone}")

    # =====================
    # SHADE LIST
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
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).softmax(dim=-1)

    top_probs, top_idxs = similarity[0].topk(3)

    # =====================
    # OUTPUT
    # =====================
    st.subheader("💄 Rekomendasi AI")

    for i in range(3):
        st.write(f"{shade_texts[top_idxs[i]]} — {top_probs[i].item()*100:.1f}%")
