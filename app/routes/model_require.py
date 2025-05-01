from fastapi import APIRouter

from pydantic import BaseModel
router = APIRouter()



# FastAPI 엔드포인트 정의
@router.get("/modelRequire")
def model_inform():
    return {"message": "AI 서버 작동 중"}
