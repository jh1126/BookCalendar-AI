from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPES = ["question", "intent", "emotion"]

# 자동학습 여부 설정(미완료)
MODEL_AUTO_FLAGS = {
    "question": 0,
    "intent": 0,
    "emotion": 1
    
}

#모든 버전 리스트 json
def load_metrics(model_type):
    metrics_path = os.path.join(
        BASE_DIR, "..", "..", "data", model_type, f"{model_type}_model_metrics.json"
    )
    if not os.path.exists(metrics_path):
        return []

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [entry["model_name"] for entry in data]

#사용중인 모델 json
def load_current_model(model_type): 
    run_path = os.path.join(
        BASE_DIR, "..", "models", model_type, f"{model_type}_model_run.json"
    )
    if not os.path.exists(run_path):
        return None

    with open(run_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data[0]["model_name"] if data else None

# FastAPI 엔드포인트 정의
@router.get("/modelRequire")
def model_inform():
    result = {}

    # model 리스트들 추가
    for model_type in MODEL_TYPES:
        result[f"{model_type}Model"] = load_metrics(model_type)

    # 현재 사용 중인 모델
    for model_type in MODEL_TYPES:
        result[f"{model_type}Loaded"] = load_current_model(model_type)

    # 자동학습 플래그
    for model_type in MODEL_TYPES:
        result[f"{model_type}ModelAuto"] = MODEL_AUTO_FLAGS.get(model_type, 0)

    print(result)
    return JSONResponse(content=result)
