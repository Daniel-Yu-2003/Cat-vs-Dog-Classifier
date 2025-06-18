from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Add this
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests

app = Flask(__name__)
CORS(app) 

model = load_model('light_cat_dog.h5')

@app.route('/predict', methods=['POST'])

def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file'].read()
    img = Image.open(io.BytesIO(file)).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    label = "cat" if prediction < 0.5 else "dog"
    confidence = float(1 - prediction) if label == "cat" else float(prediction)
    
    return jsonify({
        'prediction': label,
        'confidence': round(confidence * 100, 2)
    })

@app.route('/predictO', methods=['POST'])

def predictO():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file'].read()
    img = Image.open(io.BytesIO(file)).resize((128, 128))

    processor = AutoImageProcessor.from_pretrained("Dricz/cat-vs-dog-resnet-50")
    model = AutoModelForImageClassification.from_pretrained("Dricz/cat-vs-dog-resnet-50")

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    label = "cat" if probs[0] > probs[1] else "dog"
    confidence = probs[0] if probs[0] > probs[1] else probs[1]
    return jsonify({
        'prediction': label,
        'confidence': round(confidence * 100, 2)
    })

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
