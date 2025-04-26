from fastapi import FastAPI
from app.routes import (
    emotion_predict,
    emotion_model_version,
    emotion_settings
)

app = FastAPI(title="AI API", version="1.0")

# 감정 분석 관련 라우터 등록
app.include_router(emotion_train.router, prefix="/emotion", tags=["Emotion"]) # 학습 & 검증
app.include_router(emotion_model_version.router, prefix="/emotion", tags=["Emotion"]) # 모델 배포 기능
# 모델 버전 확인
app.include_router(emotion_settings.router, prefix="/emotion", tags=["Emotion"]) # 모델 파라메터 수정
app.include_router(emotion_settings.router, prefix="/emotion", tags=["Emotion"]) # 모델 삭제 
# 장애 기록 정보 제공
app.include_router(emotion_predict.router, prefix="/emotion", tags=["emotion"]) # 모델 예측

# 의도 분류 관련 라우터 등록
app.include_router(emotion_train.router, prefix="/emotion", tags=["intent"]) # 학습 & 검증
app.include_router(emotion_model_version.router, prefix="/emotion", tags=["intent"]) # 모델 배포 기능
# 모델 버전 확인
app.include_router(emotion_settings.router, prefix="/emotion", tags=["intent"]) # 모델 파라메터 수정
app.include_router(emotion_settings.router, prefix="/emotion", tags=["intent"]) # 모델 삭제 
# 장애 기록 정보 제공
app.include_router(emotion_predict.router, prefix="/emotion", tags=["intent"]) # 모델 예측

# 헬스 체크용 루트 엔드포인트
@app.get("/")
def read_root():
    return {"message": "AI 감정 분석 서버 작동 중!"}
