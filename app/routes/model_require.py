from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

# 모델 기록 불러오기
def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return []
    
def load_inform():
    emotion_file = os.path.join(os.path.dirname(__file__), '..','..','..','data','emotion','emotion_model_metrics.json')
    intent_file = os.path.join(os.path.dirname(__file__), '..','..','..','data','intent','intent_model_metrics.json')

    
    

# FastAPI 엔드포인트 정의
@router.get("/modelRequire")
def model_inform():
    return JSONResponse(content={"questionModel": "abcd"})
