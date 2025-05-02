# app/routes/question/question_train.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os, json

router = APIRouter()

# 경로 설정
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..')
METRICS_FILE = os.path.join(BASE_PATH, 'data', 'question', 'question_model_metrics.json')
MODEL_DIR = os.path.join(BASE_PATH, 'models', 'question')

# 요청 데이터 구조 정의
class TrainRequest(BaseModel):
    newModelName: str
    epoch: int
    batchSize: int

@router.post("/train_question")
def train_question_model(data: TrainRequest):
    model_name = data.newModelName
    epoch = data.epoch
    batch_size = data.batchSize

    # 모델 저장 폴더 경로
    model_path = os.path.join(MODEL_DIR, model_name)

    # 이미 존재하면 오류 반환
    if os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="이미 존재하는 모델 이름입니다.")

    # 실제 훈련 로직은 여기에 연결할 수 있음
    # train_question_model_fn(model_name, epoch, batch_size)

    # 여기선 디렉토리만 생성
    os.makedirs(model_path, exist_ok=True)

    # 예시용 dummy ROUGE 점수
    dummy_rouge_score = 0.7324

    # 모델 성능 기록 항목 생성
    new_entry = {
        "modelName": model_name,
        "epoch": epoch,
        "batchSize": batch_size,
        "rouge_score": round(dummy_rouge_score, 4)
    }

    # 기존 metrics 읽기
    metrics = []
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, encoding="utf-8") as f:
            metrics = json.load(f)

    # 신규 항목 추가
    metrics.append(new_entry)

    # 메트릭 파일 저장
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return {
        "message": f"'{model_name}' 모델 훈련 및 등록 완료. (ROUGE: {new_entry['rouge_score']})"
    }
