import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip
import torchvision.transforms as transforms
import timm
from diffusers import StableDiffusionInpaintPipeline

st.title("💄 Shade AI (FINAL + DIFFUSION)")

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
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    model.eval()
    return model

dino_model = load_dino()

# =====================
# LOAD DIFFUSION
# =====================
@st.cache_resource
def load_diffusion():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32
    )
    return pipe

pipe = load_diffusion()

# =====================
# MAIN
# =====================
if
