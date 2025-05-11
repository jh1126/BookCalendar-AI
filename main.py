from fastapi import FastAPI

from app.routes import model_require
from app.routes import model_score
from app.routes import auto_train_settings
from app.routes import error_request

# 자동화 관련 import 추가
from apscheduler.schedulers.background import BackgroundScheduler
from app.routes.auto_job import run_auto_training
import datetime

# Emotion
from app.routes.emotion import (
    emotion_predict,
    emotion_version,
    emotion_train,
    emotion_delete,
)
# 자동화 시간 설정
def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Seoul")  # 한국 시간 설정

    # 매월 첫째 주 월요일 오전 3시에 실행
    scheduler.add_job(
        run_auto_training,
        trigger='cron',
        day='1-7',       # 첫째 주 (1~7일)
        day_of_week='mon',
        hour=3,
        minute=0,
        id="auto_train_job",
        replace_existing=True
    )

    scheduler.start()

# Intent
from app.routes.intent import (
    intent_predict,
    intent_version,
    intent_train,
    intent_delete,
)

#Question
from app.routes.question import (
    question_predict,
    question_version,
    question_train,
    question_delete,
)

app = FastAPI(title="AI API", version="1.0")

#/modelRequire ({모델 이름json 리턴, 서비스에 사용중인 모델, 자동학습여부)
app.include_router(model_require.router, prefix="", tags=["all"])
#/testModel(현재 사용중인 모델 성능지표 출력)
app.include_router(model_score.router, prefix="", tags=["all"])
#/autoTrain(자동학습)
app.include_router(auto_train_settings.router, prefix="", tags=["all"])
#/errorRequest(에러기록)
app.include_router(error_request.router, prefix="", tags=["all"])


# 감정 분석 관련 라우터 등록(관리자 서버)
app.include_router(emotion_train.router, prefix="/emotion", tags=["emotion"]) # 학습 & 검증
app.include_router(emotion_version.router, prefix="/emotion", tags=["emotion"]) # 모델 버전 선택 기능
app.include_router(emotion_delete.router, prefix="/emotion", tags=["emotion"]) # 모델 삭제 
#app.include_router(emotion_logs.router, prefix="/emotion", tags=["emotion"])# 장애 기록 정보 제공(미완료)
# 감정 분석 예측 (서비스 서버)
app.include_router(emotion_predict.router, prefix="/emotion", tags=["emotion"]) # 모델 예측

# 의도 분류 관련 라우터 등록(관리자 서버)
app.include_router(intent_train.router, prefix="/intent", tags=["intent"]) # 학습 & 검증
app.include_router(intent_version.router, prefix="/intent", tags=["intent"]) # 모델 버전 선택 기능
app.include_router(intent_delete.router, prefix="/intent", tags=["intent"]) # 모델 삭제 
#app.include_router(intent_logs.router, prefix="/intent", tags=["intent"]) # 장애 기록 정보 제공(미완료)
# 의도 분류 예측 (서비스 서버)
app.include_router(intent_predict.router, prefix="/intent", tags=["intent"]) # 모델 예측

#Question (관리자 서버)
app.include_router(question_train.router, prefix="/question", tags=["question"])
app.include_router(question_version.router, prefix="/question", tags=["question"])
app.include_router(question_delete.router, prefix="/question", tags=["question"])
# Question (서비스 서버)
app.include_router(question_predict.router, prefix="/question", tags=["question"])

start_scheduler()

@app.get("/")
def read_root():
    return {"message": "AI 감정 분석 서버 작동 중!"}
