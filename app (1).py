
import streamlit as st
from PIL import Image
import torch
import open_clip

st.title("💄 Shade Lip Cream Recommendation AI")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg","png","jpeg"])

shades = [
    "natural nude lip cream",
    "warm peach lip cream",
    "soft pink lip cream",
    "deep red lip cream",
    "brown nude lip cream"
]

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

text = tokenizer(shades)
text_features = model.encode_text(text)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto kamu", use_column_width=True)

    st.write("Memproses AI...")

    face_feature = text_features[0].unsqueeze(0)  # sementara dummy

    similarity = (face_feature @ text_features.T)
    best_idx = similarity.argmax().item()

    st.success("Rekomendasi shade kamu:")
    st.write(shades[best_idx])
