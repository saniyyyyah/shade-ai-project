import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip
import torchvision.transforms as transforms
import timm

# =====================
# TRY DIFFUSION
# =====================
try:
    from diffusers import StableDiffusionInpaintPipeline
    diffusion_available = True
except:
    diffusion_available = False

st.title("💄 Shade AI (FINAL CLOUD SAFE)")

uploaded_file = st.file_uploader("Upload foto", type=["jpg","png","jpeg"])

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    model.eval()
    return model, preprocess

@st.cache_resource
def load_dino():
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
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
    final_score = (clip_sim * (1 + 0.3 * dino_weight))
    final_score = (final_score * 10).softmax(dim=-1)

    top_probs, top_idxs = final_score[0].topk(3)

    # =====================
    # OUTPUT
    # =====================
    st.subheader("🤖 Rekomendasi AI")

    for i in range(3):
        st.write(f"💄 {shade_texts[top_idxs[i]]} — {top_probs[i].item()*100:.1f}%")

    st.subheader("🧠 DINO Influence")
    st.write("DINO weight:", round(dino_weight, 3))

    # =====================
    # DIFFUSION SAFE MODE
    # =====================
    if diffusion_available:
        st.subheader("💋 Simulasi Lipstick AI")

        try:
            @st.cache_resource
            def load_diffusion():
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float32
                )
                return pipe

            pipe = load_diffusion()

            init_image = image.resize((512, 512))

            mask = Image.new("L", (512, 512), 0)
            mask_np = np.array(mask)
            mask_np[300:400, 200:320] = 255
            mask = Image.fromarray(mask_np)

            best_shade = shade_texts[top_idxs[0]]

            prompt = f"{best_shade} on lips, natural makeup, realistic face"

            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    image=init_image,
                    mask_image=mask,
                    num_inference_steps=10
                ).images[0]

            st.image(result, caption="Hasil Simulasi Lipstick")

        except:
            st.warning("⚠️ Diffusion gagal (server tidak kuat)")

    else:
        st.info("💡 Diffusion hanya bisa jalan di laptop (local)")
