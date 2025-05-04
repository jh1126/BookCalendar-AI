from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json

# 라우터 선언
router = APIRouter()

# 요청 바디 모델
class TextInput(BaseModel):
    loadModelName: str

# 경로 상수
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
INTENT_MODEL_RUN_PATH = os.path.join(BASE_PATH, 'models', 'intent', 'intent_model_run.json')

# 모델 버전 저장 함수
def save_model_version(model_name: str):
    # 저장할 내용: 모델 이름만 포함된 단일 딕셔너리 리스트
    data = [{"model_name": model_name}]
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(INTENT_MODEL_RUN_PATH), exist_ok=True)

    try:
        with open(INTENT_MODEL_RUN_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 버전 저장 실패: {e}")

# POST 엔드포인트
@router.post("/set_intent")
def set_model_version(data: TextInput):
    save_model_version(data.loadModelName)
