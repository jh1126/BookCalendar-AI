from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPES = ["question", "intent", "emotion"]

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

# 자동학습 json
def auto_model(model_type): 
    run_path = os.path.join(
        BASE_DIR, "..", "..", "data", "auto_model.json"
    )
    if not os.path.exists(run_path):
        return 0  # 기본값: 자동 학습 안 함

    with open(run_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return 0

    # 키 이름 예: "questionAuto", "intentAuto", "emotionAuto"
    key_name = f"{model_type}Auto"
    return data[0].get(key_name, 0)
    
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
        result[f"{model_type}ModelAuto"] =  auto_model(model_type)
        
    return JSONResponse(content=result)
