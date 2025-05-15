from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json

set_train_month_router = APIRouter()


class MonthConfig(BaseModel):
    questionDataLoad: str
    intentDataLoad: str
    emotionDataLoad: str

@set_train_month_router.post("/set_train_month")
def set_train_month(config: MonthConfig):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    target_path = os.path.join(PROJECT_ROOT, "data", "train_data_month.json")
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(config.dict(), f, ensure_ascii=False, indent=4)
        return {"message": "훈련 월 정보 저장 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 실패: {e}")
