import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip
import torchvision.transforms as transforms
import timm

st.set_page_config(page_title="Shade AI Hybrid", layout="centered")
st.title("💄 Shade AI (HSV + CLIP + DINOv2 + Fusion AI)")

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
# LOAD DINOv2 (FIXED)
# =====================
@st.cache_resource
def load_dino():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    return model

dino_model = load_dino()

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

    avg_hue = hue.mean() if len(hue) > 0 else 50

    if avg_hue < 25:
        undertone = "Warm"
    elif avg_hue > 95:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    st.success(f"✨ Undertone: {undertone}")

    # =====================
    # SHADE LIST
    # =====================
    shade_map = {
        "Warm": ["coral lip cream", "peach lip cream", "terracotta lip cream"],
        "Cool": ["rose lip cream", "berry lip cream", "plum lip cream"],
        "Neutral": ["mauve lip cream", "nude lip cream", "rose lip cream"]
    }

    shade_texts = shade_map[undertone]

    # =====================
    # CLIP FEATURE
    # =====================
    image_input = clip_preprocess(Image.fromarray(face)).unsqueeze(0)
    text_tokens = open_clip.tokenize(shade_texts)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_sim = (image_features @ text_features.T)

    # =====================
    # DINO FEATURE
    # =====================
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dino_input = transform(face).unsqueeze(0)

    with torch.no_grad():
        dino_features = dino_model.forward_features(dino_input)

    dino_features = dino_features.mean(dim=1)
    dino_features = dino_features / dino_features.norm(dim=-1, keepdim=True)

    # =====================
    # FUSION (INTELLIGENCE LAYER)
    # =====================
    dino_boost = torch.ones_like(clip_sim) * 0.15

    final_score = (0.7 * clip_sim) + (0.3 * dino_boost)
    final_score = final_score.softmax(dim=-1)

    top_probs, top_idxs = final_score[0].topk(3)

    # =====================
    # OUTPUT
    # =====================
    st.subheader("🤖 AI Recommendation (CLIP + DINO Fusion)")

    for rank, idx in enumerate(top_idxs):
        st.write(
            f"💄 {shade_texts[idx]} — {top_probs[rank].item()*100:.1f}%"
        )

    st.subheader("🧠 DINOv2 Feature Preview")
    st.write(dino_features[0][:5])
