from fastapi import APIRouter
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from fastapi.responses import JSONResponse
import os
import json

#db
from database import get_connection
from datetime import datetime

router = APIRouter()

# 요청 데이터 형식 정의
class TextInput(BaseModel):
    text: str


# 클래스 디코딩

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['기쁨','당황','분노','불안','슬픔'])  # 실제 학습에 사용한 순서대로


# 예측 함수
def predict_emotion(text: str):

    # 서비스에 사용중인 json파일 불러오기
    METRICS_FILE = os.path.join(os.path.dirname(__file__), '..','..','models','emotion','emotion_model_run.json')
    # 모델 경로
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..','..', 'models', 'emotion')

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    latest_model_name = metrics[0]["model_name"]
    model_path = os.path.join(model_dir, latest_model_name)

    model = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors='tf')
    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    pred_class = np.argmax(probs, axis=1)[0]
    decoded_label = label_encoder.inverse_transform([pred_class])[0]
    return decoded_label, probs[0].tolist()

# POST API 엔드포인트
@router.post("/predict_emotion")
def predict(input_data: TextInput, request: Request):

    print(reauest.body)
    
    text = input_data.text
    emotion, prob = predict_emotion(text)

    # 현재 날짜 (월 단위)
    date_str = datetime.now().strftime("%Y-%m-%d")

    # DB 저장
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO emotionData (date, input, output) VALUES (%s, %s, %s)"
            cursor.execute(sql, (date_str, text, emotion))
        conn.commit()
    finally:
        conn.close()

    return JSONResponse(content={"emotion": emotion})
