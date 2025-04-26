# 배포된 모델의 추론 파라미터 수정 함수
from fastapi import APIRouter
from app.models import emotion model

router = APIRouter()

@router.post("/set_inference_config")
def set_inference_config(temperature: float = 1.0, max_length: int = 128):
    current_model.inference_config = {
        "temperature": temperature, 
        "max_length": max_length
    }
    return {"message": "추론 파라미터가 변경되었습니다."}
