from fastapi import APIRouter
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder

from app.models import emotion_model

router = APIRouter()

# 요청 데이터 형식 정의
class TextInput(BaseModel):
    text: str

# 파인튜닝된 모델과 토크나이저 로드
model = emotion_model.current_model # 현재 서비스에 사용되는 모델 경로
tokenizer = emotion_model.current_tokenizer # 현재 서비스에 사용되는 토크나이저 경로

# 클래스 디코딩

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['기쁨','당황','분노','불안','슬픔'])  # 실제 학습에 사용한 순서대로


# 예측 함수
def predict_emotion(text: str):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors='tf')
    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    pred_class = np.argmax(probs, axis=1)[0]
    decoded_label = label_encoder.inverse_transform([pred_class])[0]
    return decoded_label, probs[0].tolist()

# POST API 엔드포인트
@router.post("/predict_emotion")
def predict(input_data: TextInput):
    text = input_data.text
    emotion, prob = predict_emotion(text)

    return {
        "input": text,
        "predicted_emotion": emotion,
        "probabilities": prob,
        "question": message
    }
