from fastapi import APIRouter
from app.services.emotion_model_loader import load_emotion_model
from app.models import emotion_model

# 모델 버전 선택 라우터
router = APIRouter()

@router.post("/set_emotion/{version}")
def set_model_version(version: str):
    model, tokenizer = load_emotion_model(version)
    emotion_model.current_model = model
    emotion_model.current_tokenizer = tokenizer
    emotion_model.current_version = version
    return {"message": f"다중 감정 분류 모델 버전 {version}이 로드되었습니다."}
