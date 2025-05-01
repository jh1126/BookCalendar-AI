import os
import shutil
from fastapi import APIRouter, HTTPException

router = APIRouter()

class TextInput(BaseModel):
    deleteModelName: str


@router.post("/emotion_delete")
def delete_emotion_model(data: TextInput):
    """지정한 버전의 감정 모델 디렉토리 삭제"""

    model_name = data.deleteModelName
    model_dir = os.path.join(
        os.path.dirname(__file__), '..', '..','..', 'models', 'emotion', f"emotion_model_{model_name}"
    )

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)  # 디렉터리 삭제
        
    else:
        raise HTTPException(status_code=404, detail=f"{model_name} 버전 모델이 존재하지 않습니다.")
