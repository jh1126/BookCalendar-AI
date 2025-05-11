from fastapi import APIRouter
from pydantic import BaseModel
import os
import json

router = APIRouter()

# 자동화 설정 경로
AUTO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Auto_model.json")

class AutoTrainRequest(BaseModel):
    questionAuto: int
    intentAuto: int
    emotionAuto: int

@router.post("/autoTrain")
def set_auto_train(req: AutoTrainRequest):
    # 저장할 내용 구성
    new_data = {
        "questionAuto": req.questionAuto,
        "intentAuto": req.intentionAuto,
        "emotionAuto": req.emotionAuto
    }

    # 파일이 없을 경우만 생성 (디렉토리 포함)
    if not os.path.exists(AUTO_PATH):
        os.makedirs(os.path.dirname(AUTO_PATH), exist_ok=True)

    # 기존 내용 완전히 덮어쓰기
    with open(AUTO_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
