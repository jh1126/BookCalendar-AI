from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os
import json

set_train_month_router = APIRouter()


class MonthConfig(BaseModel):
    questionDataLoad: str
    intentDataLoad: str
    emotionDataLoad: str

@set_train_month_router.post("/set_train_month")
def set_train_month(config: MonthConfig):
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        target_path = data_dir / "train_data_month.json"

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(config.dict(), f, ensure_ascii=False, indent=4)

        return {"message": f"훈련 월 정보 저장 완료 (→ {target_path})"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 실패: {e}")
