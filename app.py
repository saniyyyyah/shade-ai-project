import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip
import torchvision.transforms as transforms

st.set_page_config(page_title="Shade AI Hybrid", layout="centered")
st.title("💄 Shade AI (HSV + CLIP + DINOv2)")

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
# LOAD DINOv2
# =====================
@st.cache_resource
def load_dino():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
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
    # SHADE FILTER BY UNDERTONE
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
    # DINOv2 FEATURE
    # =====================
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dino_input = transform(face).unsqueeze(0)

    with torch.no_grad():
        dino_features = dino_model(dino_input)

    dino_features = dino_features / dino_features.norm(dim=-1, keepdim=True)

    # =====================
    # OUTPUT
    # =====================
    st.subheader("🤖 CLIP Recommendation")

    for rank, idx in enumerate(top_idxs):
        st.write(
            f"💄 {shade_texts[idx]} — {top_probs[rank].item()*100:.1f}%"
        )

    st.subheader("🧠 DINOv2 Feature Preview")
    st.write(dino_features[0][:5])
