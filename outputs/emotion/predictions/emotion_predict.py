from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder

# FastAPI 인스턴스 생성
app = FastAPI()

# 요청 데이터 형식 정의 (Pydantic 모델)
class TextInput(BaseModel):
    text: str


# 파인튜닝된 모델과 토크나이저 로드
model = TFBertForSequenceClassification.from_pretrained('') #모델 경로
tokenizer = BertTokenizer.from_pretrained('') #모델 경로

# 클래스 디코딩

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['기쁨','당황','분노','불안','슬픔'])  # 실제 학습에 사용한 순서대로!


# 예측 함수
def predict_emotion(text: str):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors='tf')
    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    pred_class = np.argmax(probs, axis=1)[0]
    decoded_label = label_encoder.inverse_transform([pred_class])[0]
    return decoded_label, probs[0].tolist()


# 안녕하세요! 오늘의 Daily 독후감에서는 '{emotion}'의 감정을 담고 있네요. 
# 이 감정을 글로 표현하면서 오늘은 어떤 장면이 '{emotion}'의 감정을 더욱 느껴지게 했나요?

# ex) 이 감정을 글로 표현하면서 오늘은 어떤 장면이 '당황'의 감정을 더욱 느껴지게 했나요?


# POST API 엔드포인트
@app.post("/predict_emotion")
async def predict(input_data: TextInput):
    text = input_data.text
    emotion, prob = predict_emotion(text)

    message = (
        f"오늘의 Daily 독후감에서는 '{emotion}'의 감정을 담고 있네요. "
        f"이 감정을 글로 표현하면서 오늘은 어떤 장면이 '{emotion}'의 감정을 더욱 느껴지게 했나요?")
    
    return {
        "input": text,
        "predicted_emotion": emotion,
        "probabilities": prob,
        "question": message
    }
