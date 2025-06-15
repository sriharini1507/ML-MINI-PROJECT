import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import json
import os

st.set_page_config(page_title="Crime Scene Analyzer", layout="centered")
st.title("🔍 Crime Scene Analyzer")
st.write("Upload crime scene images to predict forensic labels and get a narrative.")

# Model file path and labels
MODEL_PATH = "resnet50_forensic_classifier_final.pth"
class_labels = ['weapon', 'bloodstain', 'footprints']
TOGETHER_API_KEY = ""

# Check model file presence
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

# Define model class
class PyTorchResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load resnet50 base (pretrained=False since loading custom weights)
        self.base_model = models.resnet50(pretrained=False)
        # Replace final layer with custom classifier
        self.base_model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.base_model(x)

# Load model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchResNet(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Image preprocessing transform (same as ResNet50 training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(img: Image.Image):
    img_t = preprocess(img).unsqueeze(0).to(device)  # add batch dim and move to device
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)
        label = class_labels[class_idx.item()]
        return label, confidence.item()

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
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        st.error(f"❌ Narrative generation failed: {str(e)}")
        return "Narrative generation failed."

uploaded_files = st.file_uploader("📂 Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"📁 {len(uploaded_files)} image(s) uploaded.")
    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
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
