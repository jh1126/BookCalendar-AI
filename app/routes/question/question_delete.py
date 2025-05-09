from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json
import shutil

router = APIRouter()

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
METRICS_FILE = os.path.join(ROOT_DIR, 'data', 'question', 'question_model_metrics.json')
CURRENT_MODEL_FILE = os.path.join(ROOT_DIR, 'app', 'models', 'question', 'question_model_run.json')
MODEL_DIR = os.path.join(ROOT_DIR, 'models', 'question')

class DeleteRequest(BaseModel):
    deleteModelName: str

@router.post("/delete_question")
def delete_question_model(request: DeleteRequest):
    target_name = request.deleteModelName

    # 1. 메트릭 목록 수정
    if not os.path.exists(METRICS_FILE):
        raise HTTPException(status_code=500, detail="모델 메트릭 파일이 없습니다.")
    
    with open(METRICS_FILE, encoding="utf-8") as f:
        metrics = json.load(f)

    updated_metrics = [m for m in metrics if m.get("model_name") != target_name]
    
    if len(updated_metrics) == len(metrics):
        raise HTTPException(status_code=404, detail="모델이 메트릭 목록에 없습니다.")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_metrics, f, indent=2, ensure_ascii=False)

    # 2. 모델 폴더 삭제
    model_path = os.path.join(MODEL_DIR, target_name)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)

    # 3. 현재 모델이면 초기화
    if os.path.exists(CURRENT_MODEL_FILE):
        with open(CURRENT_MODEL_FILE, encoding="utf-8") as f:
            current = json.load(f)
        if isinstance(current, list) and current and current[0].get("model_name") == target_name:
            with open(CURRENT_MODEL_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

