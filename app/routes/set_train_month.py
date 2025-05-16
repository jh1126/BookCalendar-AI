from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json

router = APIRouter()

class MonthData(BaseModel):
    dataLoad: str  # 월 정보 예: "2025-05"

def save_month_data(category: str, value: str):
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        data_dir = PROJECT_ROOT / "data" / category
        data_dir.mkdir(parents=True, exist_ok=True)

        target_path = data_dir / "train_data_month.json"

        # 기본 구조 생성
        if not target_path.exists():
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=4)

        # 입력 값 저장
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump({"dataLoad": value}, f, ensure_ascii=False, indent=4)

        return {"message": f"{category} 훈련 월 정보 저장 완료 (→ {target_path})"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{category} 저장 실패: {e}")


@router.post("/set_question_month")
def set_question_month(config: MonthData):
    return save_month_data("question", config.dataLoad)

@router.post("/set_emotion_month")
def set_emotion_month(config: MonthData):
    return save_month_data("emotion", config.dataLoad)

@router.post("/set_intent_month")
def set_intent_month(config: MonthData):
    return save_month_data("intent", config.dataLoad)

