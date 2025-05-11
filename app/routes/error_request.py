from fastapi import APIRouter
import os

router = APIRouter()

ERROR_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "error_log.txt")

@router.get("/errorRequest")
def get_recent_errors():
    try:
        if not os.path.exists(ERROR_LOG_PATH):
            return {"errorLog": []}

        with open(ERROR_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 마지막 10개 줄만 가져오기
        recent_errors = [line.strip() for line in lines[-10:]]

        return {"errorLog": recent_errors}

    except Exception as e:
        return {"errorLog": [f"에러 로그를 불러오는 중 오류 발생: {str(e)}"]}
