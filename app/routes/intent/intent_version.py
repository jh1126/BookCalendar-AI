from fastapi import APIRouter
import os

# 모델 버전 선택 라우터
router = APIRouter()

from pydantic import BaseModel
class TextInput(BaseModel):
    intentModelLoad: str

import json

# 사용할 모델 이름 저장 
def save_model_metrics(model_name: str): 
    # JSON 파일 경로 설정
    METRICS_FILE = os.path.join(os.path.dirname(__file__), '..','..','models','intent','intent_model_run.json')
    
    # 새 기록만 저장
    metrics = [{
        "model_name": model_name
    }]

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4) 
        
@router.post("/set_intent")
def set_model_version(data: TextInput):
    save_model_metrics(data.intentModelLoad)
