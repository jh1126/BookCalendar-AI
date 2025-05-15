from fastapi import APIRouter
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
import os
from fastapi.responses import JSONResponse
import json

#db
from database import get_connection
from datetime import datetime

router = APIRouter()

# OpenAI API 키 경로
api_key_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'secret', 'gpt_api.txt')
# API 키 읽기
with open(api_key_path, 'r') as file:
    openai_api_key = file.read().strip()
    
client = OpenAI(api_key=openai_api_key)

# 요청 데이터 형식 정의 
class TextInput(BaseModel):
    text: str

# 클래스 디코딩 (선택)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['질문','추천']) 

# 예측 함수
def predict_intent(text: str):

    # 서비스에 사용중인 json파일 불러오기
    METRICS_FILE = os.path.join(os.path.dirname(__file__), '..','..','models','intent','intent_model_run.json')
    # 모델 경로
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..','..', 'models', 'intent')

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

# 질문 응답 생성 함수
def answer_book_question(text: str):
    prompt = (
        f"사용자가 다음과 같은 질문을 했습니다:\n"
        f"\"{text}\"\n\n"
        f"이 질문은 도서의 내용이나 주제, 인물, 감정 등과 관련된 질문입니다. "
        f"성실하고 구체적인 답변을 해주세요."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 문학 작품에 대해 깊이 있는 답변을 제공하는 독서 토론 전문가야."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

# 도서 추천 함수
def generate_recommendation(text: str):
    prompt = (
        f"사용자가 다음과 같은 요청을 보냈습니다:\n"
        f"\"{text}\"\n\n"
        f"이 요청을 참고하여 사용자가 원하는 장르, 카테고리, 분위기, 스토리 등을 고려한 책을 세 권 추천해 주세요. 만약 요청에 책 이름이 있다면 비슷한 종류의 책을 추천해 주세요. "
        f"추천 이유도 간단히 설명해 주세요."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 사용자의 요청의 조건에 맞는 책을 추천하는 조언자야."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

# FastAPI 라우터
@router.post("/predict_intent")
def predict(input_data: TextInput):
    text = input_data.text
    intent, prob = predict_intent(text)

    if intent == "질문":
        result = answer_book_question(text)

        # 현재 날짜 (월 단위)
        date_str = datetime.now().strftime("%Y-%m")

        # DB 저장
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO emotionData (date, input, output) VALUES (%s, %s, %s)"
                cursor.execute(sql, (date_str, text, emotion))
            conn.commit()
        finally:
            conn.close()
        
        return JSONResponse(content={"message": result})

    elif intent == "추천":
        result = generate_recommendation(text)
        
        # 현재 날짜 (월 단위)
        date_str = datetime.now().strftime("%Y-%m")

        # DB 저장
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO emotionData (date, input, output) VALUES (%s, %s, %s)"
                cursor.execute(sql, (date_str, text, emotion))
            conn.commit()
        finally:
            conn.close()
            
        return JSONResponse(content={"message": result})
    

    else:
        return JSONResponse(content={"message": "현재는 '질문' 또는 '추천' 의도만 지원됩니다."})
