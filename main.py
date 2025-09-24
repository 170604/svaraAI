# main.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# Define the FastAPI app
app = FastAPI(title="SvaraAI Reply Classification API")

# --- 1. Load artifacts ---
# Load the model, tokenizer, and label encoder once at startup
try:
    MODEL_DIR = "model/transformer"
    ARTIFACT_DIR = "artifacts"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    le = joblib.load(f"{ARTIFACT_DIR}/label_encoder.joblib")
    print("✅ Model, tokenizer, and label encoder loaded successfully.")

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    # In a real application, you might want to handle this more gracefully
    model, tokenizer, le = None, None, None

# --- 2. Define data models ---
# Pydantic model for the input data
class PredictionRequest(BaseModel):
    text: str

# Pydantic model for the output data
class PredictionResponse(BaseModel):
    predicted_label: str

# --- 3. Helper Function ---
def clean_text(s: str) -> str:
    """A simple text cleaning function."""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\@\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- 4. Define the prediction endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Accepts text input and returns the predicted classification label.
    """
    if not all([model, tokenizer, le]):
        return {"error": "Model artifacts not loaded. Please check server logs."}

    # Clean and prepare the input text
    text = clean_text(request.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    # Make prediction
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = le.inverse_transform([predicted_class_id])[0]
    
    return PredictionResponse(predicted_label=predicted_label)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API is running. Go to /docs for documentation."}