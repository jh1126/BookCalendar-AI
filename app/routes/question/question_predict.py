from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import json

router = APIRouter()

# 요청 형식 정의
class ParagraphRequest(BaseModel):
    paragraph: str

# 현재 선택된 모델 이름 불러오기
def get_current_question_model():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'question', 'question_model_run.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("model_name")
            elif isinstance(data, dict):
                return data.get("model_name")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 설정 파일 오류: {e}")
    raise HTTPException(status_code=404, detail="현재 설정된 질문 생성 모델이 없습니다.")

# 질문 생성 함수
def generate_questions(paragraph: str):
    model_name = get_current_question_model()
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    model_path = os.path.join(base_path, 'models', 'question', model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        input_text = f"질문 생성: {paragraph}"
        inputs = tokenizer([input_text], return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=64, num_return_sequences=2, num_beams=5)

        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated[:2]  # 질문 2개
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로딩 또는 생성 오류: {e}")

# FastAPI 라우터
@router.post("/predict_question")
def predict(input_data: ParagraphRequest):
    paragraph = input_data.paragraph
    try:
        questions = generate_questions(paragraph)
        if not isinstance(questions, list) or len(questions) < 2:
            raise ValueError("질문이 2개 이상 생성되지 않았습니다.")
        return {
            "question1": questions[0],
            "question2": questions[1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        questions = generate_questions(paragraph)

        if not isinstance(questions, list) or len(questions) < 2:
            raise ValueError("질문이 2개 이상 생성되지 않았습니다.")

        return {
            "question1": questions[0],
            "question2": questions[1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

