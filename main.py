from fastapi import FastAPI
from fastapi import Request
from app.routes import model_require
from app.routes import model_score
from app.routes import auto_train_settings
from app.routes import error_request

# 자동화 관련 import 추가
from apscheduler.schedulers.background import BackgroundScheduler
from app.routes.auto_job import run_auto_training

app = FastAPI() 

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    log_error(f"[{request.method}] {request.url} - {str(exc)}")
    return JSONResponse(status_code=500, content={"message": "서버 내부 오류 발생"})

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
