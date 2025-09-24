# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import uvicorn
import re
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

ARTIFACT_DIR = "artifacts"
MODEL_DIR = "model/transformer"

app = FastAPI(title="Reply Classifier API", version="1.0")

class InRequest(BaseModel):
    text: str
    model: Optional[str] = "auto"  # "auto" | "baseline" | "transformer"

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\@\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# load baseline if available
baseline = None
tfidf = None
label_encoder = None
logreg = None
if os.path.exists(os.path.join(ARTIFACT_DIR, "label_encoder.joblib")):
    label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
if os.path.exists(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib")) and os.path.exists(os.path.join(ARTIFACT_DIR, "logreg.joblib")):
    tfidf = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
    logreg = joblib.load(os.path.join(ARTIFACT_DIR, "logreg.joblib"))

# load transformer pipeline if available
transformer_pipeline = None
if os.path.isdir(MODEL_DIR):
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        transformer_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    except Exception as e:
        print("Transformer load failed:", e)
        transformer_pipeline = None

@app.get("/")
def root():
    return {"status": "ok", "models_available": {
        "transformer": transformer_pipeline is not None,
        "baseline": (tfidf is not None and logreg is not None)
    }}

@app.post("/predict")
def predict(req: InRequest):
    text = req.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty 'text' provided.")
    which = req.model.lower() if req.model else "auto"
    cleaned = clean_text(text)

    # prefer transformer if auto
    if which == "auto" and transformer_pipeline is not None:
        which = "transformer"
    if which == "transformer":
        if transformer_pipeline is None:
            raise HTTPException(status_code=500, detail="Transformer model not available on server.")
        out = transformer_pipeline(cleaned)[0]  # list of dicts
        best = max(out, key=lambda x: x['score'])
        # convert label to int id if needed
        label_raw = best['label']
        confidence = float(best['score'])
        # transform label id to text label via label_encoder if possible
        try:
            if label_raw.startswith("LABEL_"):
                label_id = int(label_raw.replace("LABEL_", ""))
            else:
                label_id = int(label_raw)
            label_name = label_encoder.inverse_transform([label_id])[0] if label_encoder is not None else str(label_id)
        except:
            # fallback: model might store string labels - try mapping using order
            # convert probabilities list -> choose argmax idx
            # best may be e.g. {'label': 'positive', 'score': 0.87} -> then label_raw is textual label
            label_name = label_raw if not label_raw.startswith("LABEL_") else label_raw
        return {"label": label_name, "confidence": round(confidence, 4), "model": "transformer"}

    elif which == "baseline":
        if tfidf is None or logreg is None or label_encoder is None:
            raise HTTPException(status_code=500, detail="Baseline model not available on server.")
        x_tfidf = tfidf.transform([cleaned])
        pred_id = int(logreg.predict(x_tfidf)[0])
        probs = logreg.predict_proba(x_tfidf)[0]
        confidence = float(np.max(probs))
        label_name = label_encoder.inverse_transform([pred_id])[0]
        return {"label": label_name, "confidence": round(confidence, 4), "model": "baseline"}
    else:
        raise HTTPException(status_code=400, detail="Unknown model selection. Use 'auto', 'baseline' or 'transformer'.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
