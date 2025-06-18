from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Dricz/cat-vs-dog-resnet-50")
model = AutoModelForImageClassification.from_pretrained("Dricz/cat-vs-dog-resnet-50")

# Load image
url = "https://images.dog.ceo/breeds/husky/n02110185_1469.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess and predict
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# Interpret results
probs = torch.softmax(logits, dim=-1).squeeze().tolist()
labels = model.config.id2label
print({labels[i]: probs[i] for i in range(len(labels))})