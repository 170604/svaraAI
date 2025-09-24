# SvaraAI - End-to-End Reply Classification System

## 1. Project Overview

This project is a complete, end-to-end machine learning system built for the SvaraAI AI/ML Engineer Internship Assignment. The application can classify text-based replies into predefined categories in real-time. It involves three main stages: model training, API development, and containerization with Docker.

---

## 2. Features

* **Fine-Tuned Model**: Utilizes a `distilbert-base-uncased` model fine-tuned for sequence classification on the provided dataset.
* **REST API**: A robust and efficient API built with FastAPI to serve the classification model.
* **Real-time Prediction**: A `/predict` endpoint that accepts new text and returns a predicted label in JSON format.
* **Containerized**: Fully containerized with Docker, ensuring easy deployment and portability.

---

## 3. Tech Stack

* **Machine Learning**: PyTorch, Transformers, Scikit-learn
* **API**: FastAPI, Uvicorn
* **Containerization**: Docker
* **Data Handling**: Pandas, Joblib

---

## 4. Setup and Installation

### Prerequisites

* Docker Desktop installed and running.
* An environment with Python 3.10+.

### Steps

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Train the Model**:
    (Optional, as the trained model is already included) If you want to retrain the model, run the training script:
    ```bash
    python fine_tune_transformer.py
    ```

3.  **Build the Docker Image**:
    ```bash
    docker build -t svara-api .
    ```

4.  **Run the Docker Container**:
    ```bash
    docker run -p 8000:8000 svara-api
    ```
    The application will be available at `http://127.0.0.1:8000`.

---

## 5. Usage

You can interact with the API in two main ways:

### A. Using the Interactive Docs (Swagger UI)

Navigate to `http://127.0.0.1:8000/docs` in your web browser to access the interactive API documentation, where you can test the `/predict` endpoint directly.

### B. Using cURL

Send a `POST` request from your terminal:

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Thank you for your excellent support!"
}'
```

**Example Response**:
```json
{
  "predicted_label": "appreciation"
}
```
