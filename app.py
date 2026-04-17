import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import open_clip
import torchvision.transforms as transforms
import timm
import torch.nn.functional as F

st.set_page_config(page_title="Shade AI Hybrid", layout="centered")
st.title("💄 Shade AI (FINAL AI UPGRADE)")

uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg","png","jpeg"])

# =====================
# LOAD CLIP
# =====================
@st.cache_resource
def load_clip():
    model, _, preprocess
