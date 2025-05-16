import os
import shutil
from fastapi import APIRouter, HTTPException
import json

router = APIRouter()

from pydantic import BaseModel
class TextInput(BaseModel):
    deleteModelName: str

def delete_model_info_from_json(model_name: str):
    """지정한 모델 이름을 가진 항목을 JSON 파일에서 삭제합니다."""

    # JSON 파일 경로 (필요 시 수정)
    json_path = os.path.join(os.path.dirname(__file__), '..','..','..','data','emotion','emotion_model_metrics.json')

    # JSON 불러오기
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            models = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("모델 정보 JSON 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        raise ValueError("JSON 파일 형식이 잘못되었습니다.")

    # 모델 삭제
    original_len = len(models)
    models = [m for m in models if m.get("model_name") != model_name]

    if len(models) == original_len:
        raise ValueError(f"모델 '{model_name}'을(를) JSON 파일에서 찾을 수 없습니다.")

    # JSON 저장
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(models, file, ensure_ascii=False, indent=4)
        
@router.post("/delete_emotion")
def delete_emotion_model(data: TextInput):
    """지정한 버전의 감정 모델 디렉토리 삭제"""
    model_name = data.deleteModelName
    model_dir = os.path.join(
        os.path.dirname(__file__), '..', '..','..', 'models', 'emotion', model_name)

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)  # 디렉터리 삭제
        delete_model_info_from_json(model_name) #json 정보 삭제
        


