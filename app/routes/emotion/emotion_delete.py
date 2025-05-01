import os
import shutil
from fastapi import APIRouter, HTTPException

router = APIRouter()

from pydantic import BaseModel
class TextInput(BaseModel):
    deleteModelName: str


@router.post("/delete_emotion")
def delete_emotion_model(data: TextInput):
    """지정한 버전의 감정 모델 디렉토리 삭제"""
    return JSONResponse(content={"questionModel": "abcd"})

