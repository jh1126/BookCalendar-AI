from fastapi import APIRouter
from app.services.model_loader import load_model
from app.models import current_model

router = APIRouter()

@router.post("/set_version/{version}")
def set_model_version(version: str):
    model, tokenizer = load_model(version)
    current_model.current_model = model
    current_model.current_tokenizer = tokenizer
    current_model.current_version = version
    return {"message": f"모델 버전 {version}이 로드되었습니다."}
