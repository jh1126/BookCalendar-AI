from fastapi import FastAPI
from app.routes import (
    emotion_predict,
    emotion_model_version,
    emotion_settings
)

app = FastAPI(title="AI API", version="1.0")

# 감정 분석 관련 라우터 등록(관리자 서버)
app.include_router(emotion_train.router, prefix="/emotion", tags=["Emotion"]) # 학습 & 검증
app.include_router(emotion_model_set.router, prefix="/emotion", tags=["Emotion"]) # 모델 배포 기능
app.include_router(emotion_version_confirm.router, prefix="/emotion", tags=["Emotion"])# 모델 버전 확인
app.include_router(emotion_settings.router, prefix="/emotion", tags=["Emotion"]) # 모델 파라메터 수정
app.include_router(emotion_delete.router, prefix="/emotion", tags=["Emotion"]) # 모델 삭제 
app.include_router(emotion_logs.router, prefix="/emotion", tags=["Emotion"])# 장애 기록 정보 제공(미완료)
# 감정 분석 예측 (서비스 서버)
app.include_router(emotion_predict.router, prefix="/emotion", tags=["emotion"]) # 모델 예측

# 의도 분류 관련 라우터 등록(관리자 서버)
app.include_router(intent_train.router, prefix="/intent", tags=["intent"]) # 학습 & 검증
app.include_router(intent_model_set.router, prefix="/intent", tags=["intent"]) # 모델 배포 기능
app.include_router(intent_version_confirm.router, prefix="/intent", tags=["intent"])# 모델 버전 확인
app.include_router(intent_settings.router, prefix="/intent", tags=["intent"]) # 모델 파라메터 수정
app.include_router(intent_delete.router, prefix="/intent", tags=["intent"]) # 모델 삭제 
app.include_router(intent_logs.router, prefix="/intent", tags=["intent"]) # 장애 기록 정보 제공(미완료)
# 의도 분류 예측 (서비스 서버)
app.include_router(intent_predict.router, prefix="/intent", tags=["intent"]) # 모델 예측



@app.get("/")
def read_root():
    return {"message": "AI 감정 분석 서버 작동 중!"}
