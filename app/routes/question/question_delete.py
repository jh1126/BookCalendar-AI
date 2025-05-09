from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json
import shutil

router = APIRouter()

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(BASE_DIR, '..', '..', '..')
METRICS_FILE = os.path.join(ROOT_DIR, 'data', 'question', 'question_model_metrics.json')
CURRENT_FILE = os.path.join(ROOT_DIR, 'app', 'models', 'question', 'question_model_run.json')
MODEL_DIR = os.path.join(ROOT_DIR, 'models', 'question')

class DeleteRequest(BaseModel):
    deleteModelName: str

@router.post("/delete_question")
def delete_model(request: DeleteRequest):
    delete_name = request.deleteModelName

    # 모델 목록 열기
    if not os.path.exists(METRICS_FILE):
        raise HTTPException(status_code=500, detail="모델 목록 파일이 없습니다.")

    with open(METRICS_FILE, encoding="utf-8") as f:
        model_list = json.load(f)

    # 모델 제거
    new_list = [m for m in model_list if m.get("model_name") != delete_name]
    if len(new_list) == len(model_list):
        raise HTTPException(status_code=404, detail="해당 모델이 목록에 없습니다.")

    # 폴더 삭제
    model_path = os.path.join(MODEL_DIR, delete_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # 목록 갱신
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(new_list, f, indent=2)

    # 현재 사용 모델 초기화
    if os.path.exists(CURRENT_FILE):
        with open(CURRENT_FILE, encoding="utf-8") as f:
            current = json.load(f)
        if current.get("model_name") == delete_name:
            with open(CURRENT_FILE, "w", encoding="utf-8") as f:
                json.dump({"model_name": ""}, f, indent=2)
