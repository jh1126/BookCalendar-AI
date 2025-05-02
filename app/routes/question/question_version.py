# app/routes/question/question_version.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os, json
from app.services.question.question_model_loader import load_question_model

router = APIRouter()

# 현재 사용 중인 질문 생성 모델 저장 파일 경로
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..')
CURRENT_MODEL_PATH = os.path.join(BASE_PATH, 'app', 'models', 'question', 'question_model_run.json')

# 요청 형식
class ModelLoadRequest(BaseModel):
    loadModelName: str  # 사용할 모델 이름

@router.post("/questionModelLoad")
def load_question_model_version(data: ModelLoadRequest):
    model_name = data.loadModelName

    try:
        # 모델 테스트 로드 (정상 여부 확인용, 실제 서비스에서는 추론 API에서 다시 로드함)
        model, tokenizer = load_question_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로딩 실패: {e}")

    # 현재 사용 중인 모델 이름을 JSON에 저장
    with open(CURRENT_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({ "modelName": model_name }, f, indent=2)

    return { "message": f"모델 '{model_name}' 로드 및 설정 완료." }
