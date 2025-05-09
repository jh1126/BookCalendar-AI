from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json

router = APIRouter()

# 요청 본문 구조
class TextInput(BaseModel):
    loadModelName: str

# 현재 사용 중인 question 모델 저장 파일 경로
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
QUESTION_MODEL_RUN_PATH = os.path.join(BASE_PATH, 'app', 'models', 'question', 'question_model_run.json')

# 모델 이름 저장 함수 (리스트 형태로 저장)
def save_question_model(model_name: str):
    data = [{ "model_name": model_name }]  # ✅ 리스트 안에 딕셔너리

    # 디렉토리 없으면 생성
    os.makedirs(os.path.dirname(QUESTION_MODEL_RUN_PATH), exist_ok=True)

    try:
        with open(QUESTION_MODEL_RUN_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 저장 실패: {e}")

# 모델 이름 불러오기 함수 (리스트 기준으로 파싱)
def load_question_model_name():
    try:
        with open(QUESTION_MODEL_RUN_PATH, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("model_name")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 불러오기 실패: {e}")
    raise HTTPException(status_code=404, detail="현재 설정된 모델이 없습니다.")

# POST 요청 엔드포인트 (모델 설정)
@router.post("/set_question")
def set_question_model(data: TextInput):
    save_question_model(data.loadModelName)
    return {"message": f"모델 '{data.loadModelName}'로 설정되었습니다."}

# GET 요청 엔드포인트 (현재 모델 확인)
@router.get("/get_question")
def get_question_model():
    model_name = load_question_model_name()
    return {"model_name": model_name}

