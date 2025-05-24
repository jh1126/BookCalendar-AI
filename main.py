from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes import model_require
from app.routes import model_score
from app.routes import error_request

# 오류 로거 import
from data.log_error import log_error

# 자동화 관련 import
from app.routes import auto_train_settings
from apscheduler.schedulers.background import BackgroundScheduler
from app.routes.auto_job import run_auto_training
from app.routes.set_train_month import router as set_train_month_router

# Emotion
from app.routes.emotion import (
    emotion_predict,
    emotion_version,
    emotion_train,
    emotion_delete,
)

# Intent
from app.routes.intent import (
    intent_predict,
    intent_version,
    intent_train,
    intent_delete,
)

# Question
from app.routes.question import (
    question_predict,
    question_version,
    question_train,
    question_delete,
    train_question_auto,
    update_score,
)
from app.routes.question.preview_score import router as preview_score_router


# FastAPI 앱 생성
app = FastAPI(title="AI API", version="1.0")
app.include_router(update_score.router)
app.include_router(preview_score_router)

# 전역 예외 처리기 등록
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    log_error(f"[{request.method}] {request.url} - {str(exc)}")
    return JSONResponse(status_code=500, content={"message": "서버 내부 오류 발생"})

# 스케줄러 시작 함수
def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(
        run_auto_training,
        trigger='cron',
        day='1-7',
        day_of_week='mon',
        hour=3,
        minute=0,
        id="auto_train_job",
        replace_existing=True
    )
    scheduler.start()

# 자동화 관련 라우터 등록
app.include_router(set_train_month_router, prefix="", tags=["all"])

# 라우터 등록
app.include_router(model_require.router, prefix="", tags=["all"])
app.include_router(model_score.router, prefix="", tags=["all"])
app.include_router(auto_train_settings.router, prefix="", tags=["all"])
app.include_router(error_request.router, prefix="", tags=["all"])

app.include_router(emotion_train.router, prefix="/emotion", tags=["emotion"])
app.include_router(emotion_version.router, prefix="/emotion", tags=["emotion"])
app.include_router(emotion_delete.router, prefix="/emotion", tags=["emotion"])
app.include_router(emotion_predict.router, prefix="/emotion", tags=["emotion"])

app.include_router(intent_train.router, prefix="/intent", tags=["intent"])
app.include_router(intent_version.router, prefix="/intent", tags=["intent"])
app.include_router(intent_delete.router, prefix="/intent", tags=["intent"])
app.include_router(intent_predict.router, prefix="/intent", tags=["intent"])

app.include_router(question_train.router, prefix="/question", tags=["question"])
app.include_router(question_version.router, prefix="/question", tags=["question"])
app.include_router(question_delete.router, prefix="/question", tags=["question"])
app.include_router(question_predict.router, prefix="/question", tags=["question"])
app.include_router(train_question_auto.router, prefix="/question", tags=["question"])

# 스케줄러 실행
start_scheduler()


# 기본 경로 응답
@app.get("/")
def read_root():
    return {"message": "AI 감정 분석 서버 작동 중!"}
