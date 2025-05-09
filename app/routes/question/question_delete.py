from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json

router = APIRouter()

# 요청 바디 구조
class TextInput(BaseModel):
    loadModelName: str

# JSON 경로 설정
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
QUESTION_MODEL_RUN_PATH = os.path.join(BASE_PATH, 'app', 'models', 'question', 'question_model_run.json')

# 모델 이름 저장 함수
def save_question_model(model_name: str):
    data = [
        { "model_name": model_name }
    ]

    os.makedirs(os.path.dirname(QUESTION_MODEL_RUN_PATH), exist_ok=True)

    try:
        with open(QUESTION_MODEL_RUN_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 저장 실패: {e}")

# 엔드포인트
@router.post("/set_question")
def set_question_model(data: TextInput):
    save_question_model(data.loadModelName)
