from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json

router = APIRouter()

# 입력 데이터 구조
class ScoreUpdateRequest(BaseModel):
    model_name: str
    rouge_score: float

@router.post("/question/update_score")
def update_model_score(data: ScoreUpdateRequest):
    # 파일 경로 설정
    json_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "question", "question_model_metrics.json")

    try:
        # JSON 로드
        with open(json_path, "r", encoding="utf-8") as f:
            model_list = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics JSON 파일이 존재하지 않습니다.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Metrics JSON 형식 오류")

    # 모델 이름에 해당하는 항목 찾아서 점수 갱신
    updated = False
    for model in model_list:
        if model.get("model_name") == data.model_name:
            model["ROUGE Score"] = round(data.rouge_score, 4)
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail=f"모델 '{data.model_name}'을 찾을 수 없습니다.")

    # 덮어쓰기 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_list, f, ensure_ascii=False, indent=2)

    return {"message": f"모델 '{data.model_name}'의 ROUGE Score가 {data.rouge_score}로 갱신되었습니다."}
