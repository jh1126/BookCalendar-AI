from fastapi import APIRouter

router = APIRouter()

from pydantic import BaseModel

# FastAPI 엔드포인트 정의
@router.get("/modelRequire")
def model_inform():
    return
