from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json

router = APIRouter()

class TextInput(BaseModel):
    loadModelName: str

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(BASE_DIR, '..', '..')
QUESTION_MODEL_RUN_PATH = os.path.join(ROOT_DIR, 'models', 'question', 'question_model_run.json')

# 모델 이름 저장 함수 (리스트로 저장)
def save_question_model(model_name: str):
    data = [{ "model_name": model_name }]

    try:
        os.makedirs(os.path.dirname(QUESTION_MODEL_RUN_PATH), exist_ok=True)
        with open(QUESTION_MODEL_RUN_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 저장 실패: {e}")

@router.post("/set_question")
def set_model_version(data: TextInput):
    save_question_model(data.loadModelName)

