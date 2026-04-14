import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import open_clip

st.set_page_config(page_title="Shade AI", layout="centered")
st.title("💄 Shade AI (Stable Version)")

uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg","png","jpeg"])

@st.cache_resource
def load_dino():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
   
