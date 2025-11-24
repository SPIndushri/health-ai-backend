from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils.preprocess import preprocess_image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn
import os

app = FastAPI()

# Allow Flutter / Web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load Model & Labels
# -----------------------
MODEL_PATH = "model/FinalMP_model.keras"
LABELS_PATH = "model/final_classes.npy"

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True).item()

# Convert dict → ordered list
label_list = [cls for cls, idx in sorted(labels.items(), key=lambda x: x[1])]

# -----------------------
# Prediction API
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]

    predicted_idx = int(np.argmax(preds))
    result = label_list[predicted_idx]

    return {
        "prediction": result,
        "confidence": float(np.max(preds))
    }

@app.get("/")
def home():
    return {"status": "Backend is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
