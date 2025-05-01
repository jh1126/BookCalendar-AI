from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
router = APIRouter()



# FastAPI 엔드포인트 정의
@router.get("/modelRequire")
def model_inform():
    return JSONResponse(content={"questionModel": "abcd"})
