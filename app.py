import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip
import torchvision.transforms as transforms
import timm

st.title("💄 Shade AI (CLIP + DINO REAL FUSION)")

uploaded_file = st.file_uploader("Upload foto", type=["jpg","png","jpeg"])

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
# LOAD DINO
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

    st.success(f"✨ Undertone: {undertone}")

    # =====================
    # SHADE TEXT
    # =====================
    shade_map = {
        "Warm": [
            "bright coral glossy lipstick for warm undertone skin",
            "soft peach natural lipstick warm skin tone",
            "deep terracotta matte bold lipstick tan skin"
        ],
        "Cool": [
            "cool pink glossy lipstick fair skin",
            "dark berry bold lipstick cool undertone",
            "plum elegant matte lipstick cool tone"
        ],
        "Neutral": [
            "natural nude everyday lipstick neutral skin",
            "mauve soft elegant lipstick neutral tone",
            "rose balanced lipstick natural makeup look"
        ]
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
        dino_feat = dino_model.forward_features(dino_input)

    dino_feat = dino_feat.mean(dim=1)
    dino_feat = dino_feat / dino_feat.norm(dim=-1, keepdim=True)

    # =====================
    # 🔥 DINO → IMAGE FEATURE BOOST
    # =====================
    dino_strength = torch.sigmoid(dino_feat.mean())

    # gabung ke CLIP
    fusion_sim = clip_sim * (1 + 0.5 * dino_strength)

    # final softmax
    final_score = (fusion_sim * 10).softmax(dim=-1)

    top_probs, top_idxs = final_score[0].topk(3)

    # =====================
    # OUTPUT
    # =====================
    st.subheader("🤖 AI
