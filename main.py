# === main.py ===

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import numpy as np
import requests
import json
import io
from PIL import Image
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load ResNet model (updated format)
MODEL_PATH = "resnet50_updated.keras"  # Make sure you've converted .h5 to this format
class_labels = ['weapon', 'bloodstain', 'footprints']
TOGETHER_API_KEY = "97f2e2e43d184a56e60f1895332ede2ccdb9e7fb7260d7c74fe2c74989c17d3a"  # Replace with your actual key

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ ResNet model loaded.")
except Exception as e:
    print("❌ Error loading model:", e)
    raise

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))
        x = np.array(img) / 255.0
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        class_idx = np.argmax(preds)
        label = class_labels[class_idx]
        confidence = float(preds[0][class_idx])

        prompt = f"""
You are a forensic analyst AI. Based on the scene classification '{label}', give a short and realistic summary of what might have happened. Do not exaggerate or imagine things not directly implied by the label. Only provide concise observations grounded in that classification. This is based on crimes only.
Scene classification: {label.upper()}
"""

        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        body = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9
        }

        try:
            response = requests.post("https://api.together.xyz/inference", headers=headers, data=json.dumps(body))
            narrative = response.json()["choices"][0]["text"].strip()
        except Exception as e:
            print("❌ Narrative generation failed:", e)
            narrative = "Narrative generation failed."

        results.append({
            "filename": file.filename,
            "label": label,
            "confidence": confidence,
            "narrative": narrative
        })

    return {"results": results}
