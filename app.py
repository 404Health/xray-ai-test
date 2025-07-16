import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
import torchvision.transforms as T
import matplotlib.pyplot as plt

# open image
uploaded = st.file_uploader("Upload a chest X‑ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    processor = AutoProcessor.from_pretrained("lxyuan/vit-xray-pneumonia-classification")
    model = AutoModelForImageClassification.from_pretrained("lxyuan/vit-xray-pneumonia-classification")
else:
    st.warning("⚠️ Please upload an image to proceed.")
    st.stop()

# Preprocess image into model-compatible tensor
inputs = processor(images=image, return_tensors="pt")

# Disable gradient calculation (for faster inference)
with torch.no_grad():
    outputs = model(**inputs)

# Get logits and predicted label
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)
st.subheader("Prediction")
st.success(f"🧠 The model predicts: **{predicted_label}**")

probs = torch.nn.functional.softmax(logits, dim=-1)
confidence = probs[0][predicted_class_idx].item()
st.write(f"Confidence: **{confidence:.2%}**")
