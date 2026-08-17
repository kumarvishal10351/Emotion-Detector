import re
import pickle
import numpy as np
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
MODEL_PATH = "Artifacts/BIGRU.keras"
TOKENIZER_PATH = "Artifacts/Tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 50
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
EMOTION_EMOJIS = {'sadness': '😔', 'joy': '😊', 'love': '😍', 'anger': '😠', 'fear': '😨', 'surprise': '😮'}

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to be classified")

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "I feel so happy and excited today"}]
        }
    }

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading model and tokenizer...')
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    dl_model['model'] = model
    dl_model['tokenizer'] = tokenizer
    print('Model loaded successfully')
    yield
    dl_model.clear()
    print('Model unloaded successfully')

app = FastAPI(title="Emotion Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount('/static', StaticFiles(directory="static"), name="static")

@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')

@app.get('/health', response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="Server is running",
        model_loaded=bool(dl_model.get("model") and dl_model.get("tokenizer"))
    )

@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    model = dl_model.get("model")
    tokenizer = dl_model.get("tokenizer")
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or Tokenizer is not loaded."
        )

    preprocessed_text = preprocess_text(text_input.text)
    if not preprocessed_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text contains no valid characters after preprocessing."
        )

    # Tokenization and Padding
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # Prediction
    predictions = model.predict(padded_sequence, verbose=0)[0]
    top_idx = int(np.argmax(predictions))
    predicted_emotion = EMOTION_LABELS[top_idx]
    confidence = float(predictions[top_idx])

    all_probabilities = {
        label: round(float(prob), 4)
        for label, prob in zip(EMOTION_LABELS, predictions)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=predicted_emotion,
        emoji=EMOTION_EMOJIS.get(predicted_emotion, ''),
        confidence=round(confidence, 4),
        all_probabilities=all_probabilities
    )
