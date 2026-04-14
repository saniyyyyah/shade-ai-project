import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip

st.set_page_config(page_title="Shade AI Hybrid", layout="centered")
st.title("💄 Shade AI (HSV + CLIP Hybrid)")

uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg","png","jpeg"])

# =====================
# LOAD CLIP
# =====================
@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    model.eval()
    return model, preprocess

clip_model, clip_preprocess = load_clip()

# =====================
# MAIN
# =====================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img = np.array(image)

    # crop wajah aman
    h, w, _ = img.shape
    face = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    # =====================
    # HSV UNDERTONE
    # =====================
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    hue = hue[(hue > 5) & (hue < 170)]

    if len(hue) == 0:
        avg_hue = 50
    else:
        avg_hue = hue.mean()

    if avg_hue < 25:
        undertone = "Warm"
    elif avg_hue > 95:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    st.success(f"✨ Undertone: {undertone}")

    # =====================
    # FILTER SHADE BY UNDERTONE
    # =====================
    shade_map = {
        "Warm": ["coral lip cream", "peach lip cream", "terracotta lip cream"],
        "Cool": ["rose lip cream", "berry lip cream", "plum lip cream"],
        "Neutral": ["mauve lip cream", "nude lip cream", "rose lip cream"]
    }

    shade_texts = shade_map[undertone]

    # =====================
    # CLIP RECOMMENDATION
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
    st.subheader("🤖 Rekomendasi AI (Hybrid CLIP)")

    for i in top_idxs:
        st.write("💄", shade_texts[i])
