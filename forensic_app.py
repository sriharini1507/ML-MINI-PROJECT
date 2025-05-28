import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import json
import os

st.set_page_config(page_title="Crime Scene Analyzer", layout="centered")
st.title("🔍 Crime Scene Analyzer")
st.write("Upload crime scene images to predict forensic labels and get a narrative.")

# === Debug line
st.info("📦 App started. Checking model...")

MODEL_PATH = "resnet50_se_forensic_classifier.h5"
class_labels = ['weapon', 'bloodstain', 'footprints']
TOGETHER_API_KEY = "97f2e2e43d184a56e60f1895332ede2ccdb9e7fb7260d7c74fe2c74989c17d3a"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

uploaded_files = st.file_uploader("📂 Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

def predict_image(img):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img) / 255.0
    x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    class_idx = np.argmax(preds)
    label = class_labels[class_idx]
    confidence = float(preds[0][class_idx])
    return label, confidence

def generate_narrative(label):
    prompt = f"""
You are a forensic assistant. Based only on the label '{label}', describe what it likely means in the context of a crime scene. 
Do NOT assume causes, tools, or scenarios that are not explicitly tied to this label. Keep your summary short, neutral, and observational.

Label: {label.upper()}
Response:
"""
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.3,
        "top_k": 30,
        "top_p": 0.8
    }
    try:
        response = requests.post("https://api.together.xyz/inference", headers=headers, data=json.dumps(body))
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        st.error(f"❌ Narrative generation failed: {str(e)}")
        return "Narrative generation failed."


# === Handle uploads
if uploaded_files:
    st.write(f"📁 {len(uploaded_files)} image(s) uploaded.")
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=file.name, use_column_width=True)
        label, confidence = predict_image(image)
        narrative = generate_narrative(label)

        st.markdown("### 🔍 Prediction")
        st.write(f"**Label:** `{label}`")
        st.write(f"**Confidence:** `{confidence:.2f}`")

        st.markdown("### 🧠 AI Narrative")
        st.write(narrative)
else:
    st.warning("⚠️ Please upload at least one image.")
