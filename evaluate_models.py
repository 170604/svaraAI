# evaluate_models.py
"""
Load artifacts and evaluate baseline and transformer on a holdout split.
Outputs accuracy and weighted F1 for both.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
import re

DATA_PATH = "reply_classification_dataset.csv"
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "model/transformer"

def clean_text(s):
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\@\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def eval_baseline(X_test, y_test):
    tfidf = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(ARTIFACT_DIR, "logreg.joblib"))
    le = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    X_test_tfidf = tfidf.transform(X_test)
    preds = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    print("Baseline LogisticRegression")
    print("Accuracy:", acc)
    print("Weighted F1:", f1)
    print(classification_report(y_test, preds, target_names=le.classes_))

def eval_transformer(X_test, y_test):
    # load label encoder
    le = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    if not os.path.isdir(MODEL_DIR):
        print("Transformer model not found - skip")
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    preds = []
    for t in X_test:
        out = nlp(t)[0]  # list of dicts for labels
        # find argmax
        best = max(out, key=lambda x: x['score'])
        label = best['label']
        # huggingface often labels as 0/1/..., mapping needed
        try:
            label_id = int(label.replace("LABEL_", "")) if label.startswith("LABEL_") else int(label)
        except:
            # if the model uses string labels, fallback to mapping order
            label_id = int(best.get('label', 0))
        preds.append(label_id)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    print("Transformer")
    print("Accuracy:", acc)
    print("Weighted F1:", f1)
    try:
        print(classification_report(y_test, preds, target_names=le.classes_))
    except:
        pass

def main(test_size=0.2, random_state=42):
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].fillna("").astype(str).apply(clean_text)
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    le = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    y = le.transform(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=test_size, random_state=random_state, stratify=y)
    eval_baseline(X_test, y_test)
    eval_transformer(X_test.tolist(), y_test)

if __name__ == "__main__":
    main()
