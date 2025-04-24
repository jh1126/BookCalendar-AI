from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import openai

# FastAPI 인스턴스 생성
app = FastAPI()

# OpenAI API 키 설정
openai.api_key = "openai-api-key"

# 요청 데이터 형식 정의 (Pydantic 모델)
class TextInput(BaseModel):
    text: str


# 파인튜닝된 모델과 토크나이저 로드
model = TFBertForSequenceClassification.from_pretrained('') #모델 경로
tokenizer = BertTokenizer.from_pretrained('') #모델 경로

# 클래스 디코딩 (선택)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['질문','추천']) 

# 예측 함수
def predict_intent(text: str):
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

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "너는 문학 작품에 대해 깊이 있는 답변을 제공하는 독서 토론 전문가야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message["content"].strip()

# 도서 추천 함수
def generate_recommendation(text: str):
    prompt = (
        f"사용자가 다음과 같은 요청을 보냈습니다:\n"
        f"\"{text}\"\n\n"
        f"이 요청을 참고하여 사용자의 감정, 관심사, 분위기 등을 고려한 책을 한 권 추천해 주세요. "
        f"추천 이유도 간단히 설명해 주세요."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 감정과 문학적 맥락에 맞는 책을 추천하는 조언자야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=150
    )

# FastAPI 라우터
@app.post("/predict_emotion")
async def predict(input_data: TextInput):
    text = input_data.text
    intent, prob = predict_intent(text)

    if intent == "질문":
        result = answer_book_question(text)
        return {
            "input": text,
            "predicted_intent": intent,
            "probabilities": prob,
            "answer_to_question": result
        }

    elif intent == "추천":
        result = generate_recommendation(text)
        return {
            "input": text,
            "predicted_intent": intent,
            "probabilities": prob,
            "book_recommendation": result
        }

    else:
        return {
            "input": text,
            "predicted_intent": intent,
            "probabilities": prob,
            "message": "현재는 '질문' 또는 '추천' 의도만 지원됩니다."
        }
    