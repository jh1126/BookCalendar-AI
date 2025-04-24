from fastapi import APIRouter
from app.services.emotion model_loader import load_emotion_model
from app.models import emotion model

router = APIRouter()

@router.post("/set_version/{version}")
def set_model_version(version: str):
    model, tokenizer = load_emotion_model(version)
    emotion model.current_model = model
    emotion model.current_tokenizer = tokenizer
    emotion model.current_version = version
    return {"message": f"다중 감정 분류 모델 버전 {version}이 로드되었습니다."}
