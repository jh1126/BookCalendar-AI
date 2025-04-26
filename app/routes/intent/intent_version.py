from fastapi import APIRouter
from app.services.intent model_loader import load_intent_model
from app.models import intent model

# 모델 버전 선택 라우터
router = APIRouter()

# @router.post("/set_intent/{version}")
def set_model_version(version: str):
    model, tokenizer = load_intent_model(version)
    intent model.current_model = model
    intent model.current_tokenizer = tokenizer
    intent model.current_version = version
    return {"message": f"질문 의도 분류 모델 버전 {version}이 로드되었습니다."}
