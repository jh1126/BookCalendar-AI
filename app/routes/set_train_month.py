from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json

router = APIRouter()

class QuestionMonth(BaseModel):
    questionDataLoad: str

class IntentMonth(BaseModel):
    intentDataLoad: str

class EmotionMonth(BaseModel):
    emotionDataLoad: str

# 공통 저장 함수
def save_json(category: str, data: dict):
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        data_dir = PROJECT_ROOT / "data" / category
        data_dir.mkdir(parents=True, exist_ok=True)

        target_path = data_dir / "train_data_month.json"

        # 파일 없으면 빈 JSON 생성
        if not target_path.exists():
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=4)

        # 데이터 저장
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return {"message": f"{category} 훈련 월 정보 저장 완료 (→ {target_path})"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{category} 저장 실패: {e}")


@router.post("/set_question_month")
def set_question_month(config: QuestionMonth):
    return save_json("question", config.dict())

@router.post("/set_intent_month")
def set_intent_month(config: IntentMonth):
    return save_json("intent", config.dict())

@router.post("/set_emotion_month")
def set_emotion_month(config: EmotionMonth):
    return save_json("emotion", config.dict())

