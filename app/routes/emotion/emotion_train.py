# 필요한 라이브러리 Import
from transformers import (
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    create_optimizer,
)
import os
from transformers import DataCollatorWithPadding
from datasets import Dataset
import tensorflow as tf
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse

from fastapi import APIRouter, BackgroundTasks
import json


router = APIRouter()

from pydantic import BaseModel

# 요청 데이터를 받을 클래스
class ModelConfig(BaseModel):
    newModelName: str  # 사용할 모델 이름
    epoch: int         # 에폭 수
    dropOut: float    # 드롭아웃 비율

def load_hf_token(path):
    with open(path, "r") as file:
        token = file.read().strip()
    return token

def train_emotion_model(data: ModelConfig):
    # 훈련에 필요한 데이터셋(다중 감정 분류를 위한 데이터셋) 불러오기
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..','data', 'emotion', 'processed', 'emotion_all_data.csv')
    train_df = pd.read_csv(data_path)

    # 2. 레이블 인코딩
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['감정_대분류'])
    num_labels = len(label_encoder.classes_)
    train_df['encoded_label'] = np.asarray(label_encoder.transform(train_df['감정_대분류']), dtype=np.int32)

    # 3. 텍스트와 감정 라벨 분리
    train_texts = train_df["사람문장1"].to_list()
    train_labels = train_df["encoded_label"].to_list() # 감정 라벨(0,1,2,3,4)

    # 4. 훈련데이터(70%)와 검증데이터(15%)와 테스트데이터(15%)으로 분리
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1765, random_state=42, stratify=train_labels
    )
    # 5. 모델 준비 (Hugging Face)
    from transformers import AutoTokenizer
    model_name = "klue/bert-base"
    token_path = os.path.join(os.path.dirname(__file__), '..', '..','..', 'secret', 'hf_token.txt')
    token_path = os.path.abspath(token_path)
    
    from huggingface_hub import login
    login(token=load_hf_token(token_path))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 6. 토크나이징
    from transformers import BertTokenizerFast
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # 7. TensorFlow 데이터셋 준비
    import tensorflow as tf
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
    
    # 8. 모델 설정
    from tensorflow.keras.metrics import Metric
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    optimizer, schedule = create_optimizer(
        init_lr=3e-5,
        num_train_steps=(len(train_labels) // 32) * 5,  # num_train_steps 계산
        num_warmup_steps=756 #0
    )
    # 9. 모델 컴파일
    model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

    # 10. 모델 훈련
    model.fit(
        train_dataset.shuffle(1000).batch(32),
        validation_data=val_dataset.batch(32),
        epochs=5
    )
    # Change id2label, label2id in model.config
    import re

    id2labels = model.config.id2label
    model.config.id2label = {id : label_encoder.inverse_transform([int(re.sub('LABEL_', '', label))])[0]  for id, label in id2labels.items()}

    label2ids = model.config.label2id
    model.config.label2id = {label_encoder.inverse_transform([int(re.sub('LABEL_', '', label))])[0] : id   for id, label in id2labels.items()}

    #모델 저장 경로
    model_name2 = data.newModelName
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..','..', 'models','emotion',model_name2)

    # 디렉토리 없으면 생성
    os.makedirs(model_dir, exist_ok=True)

    # 모델과 토크나이저 저장
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # 11. 테스트 데이터로 성능 평가(scrips 에 있는 훈련코드와 다르게 함. 오류 발생시 scripts 코드로 변경해야 됨)
    from transformers import TFBertForSequenceClassification, BertTokenizerFast
    from transformers import TFBertForSequenceClassification, BertTokenizer
    from sklearn.metrics import classification_report, f1_score

    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

    predictions = model.predict(test_dataset.batch(64))
    pred_labels = np.argmax(predictions.logits, axis=-1)

    report = classification_report(test_labels, pred_labels, output_dict=True)

    f1_score = report['weighted avg']['f1-score']
    accuracy = report['accuracy']

    #모델 요구사항 저장(버전 이름, 성능지표f1_score)
    save_model_metrics(model_name2, accuracy)
    
    return accuracy, f1_score

import json

# JSON 파일 경로 설정
METRICS_FILE = os.path.join(os.path.dirname(__file__), '..','..','..','data','emotion','emotion_model_metrics.json')

# 모델 기록 불러오기
def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return []
    
# 성능 기록 추가 및 저장 (버전이름, f1_score)
def save_model_metrics(model_name: str, accuracy: float): 
    metrics = load_metrics()
    
    metrics.append({
        "model_name": model_name,
        "f1_score": accuracy
    })

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)  
    
# FastAPI 엔드포인트 정의
@router.post("/train_emotion")
def train_emotion(data: ModelConfig, background_tasks: BackgroundTasks):
    
    # 백그라운드 작업 등록
    background_tasks.add_task(train_emotion_model, data)
    
    return JSONResponse(content={"message": "훈련 시작하였습니다."})
