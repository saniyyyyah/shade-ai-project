import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import open_clip
from diffusers import StableDiffusionInpaintPipeline

st.set_page_config(page_title="Shade AI", layout="centered")
st.title("💄 Shade Recommendation AI (DINOv2 + CLIP + Diffusion)")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg","png","jpeg"])

# =====================
# LOAD MODELS
# =====================

@st.cache_resource
def load_dino():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    return model, preprocess

@st.cache_resource
def load_diffusion():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32
    )
    return pipe

dino_model = load_dino()
clip_model, clip_preprocess = load_clip()
pipe = load_diffusion()

# =====================
# MAIN
# =====================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Foto kamu", use_column_width=True)

    img = np.array(image)

    # crop wajah
    h, w, _ = img.shape
    face = img[h//4:3*h//4, w//4:3*w//4]

    # =====================
    # HSV UNDERTONE
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

    st.success(f"✨ Undertone: {undertone}")

    # =====================
    # DINOv2
    # =====================
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(face).unsqueeze(0)

    with torch.no_grad():
        dino_features = dino_model(input_tensor)

    dino_features = dino_features / dino_features.norm(dim=-1, keepdim=True)

    st.subheader("🧠 DINOv2 Features")
    st.write(dino_features[0][:5])

    # =====================
    # CLIP
    # =====================
    shade_texts = [
        "coral lip cream",
        "peach lip cream",
        "terracotta lip cream",
        "rose lip cream",
        "berry lip cream",
        "plum lip cream",
        "mauve lip cream",
        "nude lip cream"
    ]

    clip_input = clip_preprocess(Image.fromarray(face)).unsqueeze(0)

    with torch.no_grad():
        image_features = clip_model.encode_image(clip_input)
        text_tokens = open_clip.tokenize(shade_texts)
        text_features = clip_model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).softmax(dim=-1)

    top_probs, top_idxs = similarity[0].topk(3)

    st.subheader("🤖 CLIP Recommendation")

    for i in top_idxs:
        st.write("💄", shade_texts[i])

    # =====================
    # DIFFUSION
    # =====================
    st.subheader("💋 Lip Cream Simulation (Diffusion)")

    init_image = image.resize((512, 512))

    mask = Image.new("L", (512, 512), 0)
    mask_np = np.array(mask)
    mask_np[300:400, 200:320] = 255
    mask = Image.fromarray(mask_np)

    prompt = f"{undertone} lip cream on lips, natural makeup"

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask,
            num_inference_steps=20
        ).images[0]

    st.image(result, caption="Hasil Simulasi")
