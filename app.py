"""
Optional Flask API for server-side inference.

GitHub Pages cannot host this (static files only) — deploy it separately on
something like Render, Railway, Fly.io, or Hugging Face Spaces, then point
frontend/script.js's API_URL at wherever this ends up running, and set
MODE = "server" in that file.

Run locally:
    pip install -r requirements.txt
    python app.py
"""

import base64
import io

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # allow the GitHub Pages frontend (different origin) to call this API

MODEL_PATH = "drowsiness_model.keras"
IMG_SIZE = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")


def decode_image(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)  # keep [0,255], matches training
    return np.expand_dims(arr, axis=0)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if "image" not in payload:
        return jsonify({"error": "missing 'image' field"}), 400

    try:
        batch = decode_image(payload["image"])
    except Exception as e:
        return jsonify({"error": f"could not decode image: {e}"}), 400

    prob = float(model.predict(batch, verbose=0)[0][0])
    label = "drowsy" if prob > 0.5 else "alert"
    return jsonify({"probability": prob, "label": label})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
