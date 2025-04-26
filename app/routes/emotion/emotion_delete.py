import os
import shutil
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.delete("/delete_emtion/{version}")
def delete_emotion_model(version: str):
    """지정한 버전의 감정 모델 디렉토리 삭제"""
    
    model_dir = os.path.join(
        os.path.dirname(__file__), '..', '..','..', 'models', 'emotion', f"emotion_model_{version}"
    )

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)  # 디렉터리 삭제
        return {
            "message": f"{version} 버전 감정 모델이 삭제되었습니다.",
            "deleted_path": model_dir
        }
    else:
        raise HTTPException(status_code=404, detail=f"{version} 버전 모델이 존재하지 않습니다.")
