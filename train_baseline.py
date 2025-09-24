import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import re
import argparse

DATA_PATH = "reply_classification_dataset.csv"
ARTIFACT_DIR = "artifacts"

def clean_text(s):
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\@\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main(test_size=0.2, random_state=42):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    if 'reply' in df.columns and 'text' not in df.columns:
        df.rename(columns={'reply': 'text'}, inplace=True)

    df.rename(columns={"reply": "text", "category": "label"}, inplace=True)

    df['text'] = df['text'].fillna("").astype(str).apply(clean_text)
    df['label'] = df['label'].astype(str).str.lower().str.strip()

    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=test_size, random_state=random_state, stratify=y)

    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=2000, solver='liblinear')
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Baseline LogisticRegression results")
    print("Accuracy:", acc)
    print("Weighted F1:", f1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    joblib.dump(tfidf, os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(ARTIFACT_DIR, "logreg.joblib"))
    joblib.dump(le, os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    print("Saved artifacts to", ARTIFACT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args.test_size, args.random_state)