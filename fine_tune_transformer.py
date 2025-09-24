# fine_tune_transformer.py
"""
Fine-tune distilbert-base-uncased for reply classification.
Saves:
 - model/transformer/ (huggingface format)
 - artifacts/label_encoder.joblib (reused)
"""
import os
import pandas as pd
import numpy as np
import re
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import torch
import joblib

DATA_PATH = "reply_classification_dataset.csv"
MODEL_DIR = "model/transformer"
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

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def main(model_name="distilbert-base-uncased", epochs=3, per_device_train_batch_size=16, lr=2e-5, test_size=0.2, seed=42):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    df['text'] = df['text'].fillna("").astype(str).apply(clean_text)
    df['label'] = df['label'].astype(str).str.lower().str.strip()

    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    joblib.dump(le, os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))

    # create HF Dataset
    dataset = Dataset.from_pandas(df[['text', 'label_id']])
    dataset = dataset.rename_column("label_id", "labels")
    dataset = dataset.class_encode_column("labels")

    # train/test split
    dataset = dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="labels")
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)
    train_ds = train_ds.map(preprocess, batched=True)
    eval_ds = eval_ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Saved transformer model to", MODEL_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(model_name=args.model_name, epochs=args.epochs, per_device_train_batch_size=args.batch_size, lr=args.lr, test_size=args.test_size, seed=args.seed) 