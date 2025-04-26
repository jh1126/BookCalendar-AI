# 필요한 라이브러리 Import
from transformers import (
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    create_optimizer,
)
from transformers import DataCollatorWithPadding
from datasets import Dataset
import tensorflow as tf
import numpy as np
import pandas as pd

# 훈련에 필요한 데이터셋(다중 감정 분류를 위한 데이터셋) 불러오기
import os
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'emotion', 'processed', 'emotion_all_data.csv')
train_df = pd.read_csv(data_path)

# Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(train_df['감정_대분류'])
num_labels = len(label_encoder.classes_)
train_df['encoded_label'] = np.asarray(label_encoder.transform(train_df['감정_대분류']), dtype=np.int32)

# 텍스트와 감정 라벨을 따로 분리
train_texts = train_df["사람문장1"].to_list() # Features (not-tokenized yet)
train_labels = train_df["감정_대분류"].to_list() # 감정 라벨(0,1,2,3,4)

# 훈련데이터(70%)와 검증데이터(15%)와 테스트데이터(15%)으로 분리
from sklearn.model_selection import train_test_split

# 훈련데이터와 테스트데이터
train_texts, test_texts, train_labels, test_labels = train_test_split(
    train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels
)

# 훈련데이터와 검증데이터
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1765, random_state=42, stratify=train_labels
)

# Klue/bert-base 모델 불러오기
from transformers import AutoTokenizer

# Hugging Face 로그인
import os
token_path = os.path.join(os.path.dirname(__file__), '..', '..', 'secret', 'hf_token.txt')
token_path = os.path.abspath(token_path)

def load_hf_token(path):
    with open(path, "r") as file:
        token = file.read().strip()
    return token

from huggingface_hub import login
login(token=load_hf_token(token_path))

# 모델 불러오기기
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 토크나이징
from transformers import BertTokenizerFast

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Tensorflow를 위한 datasets 변환
import tensorflow as tf

# 훈련 데이터셋
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

# 검증 데이터셋
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))

#모델 불러오기
from transformers import (
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    create_optimizer,
)
import tensorflow as tf
from tensorflow.keras.metrics import Metric

# Dropout 및 클래스 수 설정
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=5,
    hidden_dropout_prob=0.2,               # Dropout 설정
    attention_probs_dropout_prob=0.2
)

# 모델 로딩 (Dropout 포함된 설정 사용)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# 옵티마이저 설정 (Hugging Face 제공)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_train_steps=(80000 / 16) * 5,
    num_warmup_steps=0
)

# 모델 컴파일
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

# 모델 훈련
model.fit(
    train_dataset.shuffle(1000).batch(16),
    validation_data=val_dataset.batch(16),
    epochs=5
)

# Change id2label, label2id in model.config
import re

id2labels = model.config.id2label
model.config.id2label = {id : label_encoder.inverse_transform([int(re.sub('LABEL_', '', label))])[0]  for id, label in id2labels.items()}

label2ids = model.config.label2id
model.config.label2id = {label_encoder.inverse_transform([int(re.sub('LABEL_', '', label))])[0] : id   for id, label in id2labels.items()}

# F1-Score 성능평가
from transformers import TFBertForSequenceClassification, BertTokenizerFast
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report, f1_score

# 1. test_texts와 test_labels를 하나의 데이터프레임으로 합치기
test_df = pd.DataFrame({
    "사람문장1": test_texts,
    "감정_대분류": test_labels
})

test_df.loc[(test_df['감정_대분류'] == 0), '감정_대분류'] = '기쁨' # 기쁨 → 0
test_df.loc[(test_df['감정_대분류'] == 1), '감정_대분류'] = '당황'  # 당황 → 1
test_df.loc[(test_df['감정_대분류'] == 2), '감정_대분류'] = '분노'  # 분노 → 2
test_df.loc[(test_df['감정_대분류'] == 3), '감정_대분류'] = '불안'   #불안 → 3
test_df.loc[(test_df['감정_대분류'] == 4), '감정_대분류'] = '슬픔' # 슬픔 → 4

# 2. 입력 데이터 준비
texts = test_df['사람문장1'].tolist()       
true_labels = test_df['감정_대분류'].tolist()  # 정답 컬럼

# 3. 레이블 인코딩 (문자열을 숫자로 변환)
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)  # true_labels를 숫자로 변환

# 배치 설정
batch_size = 64
pred_labels = []

# 예측 실행
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]

    encodings = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='tf')

    outputs = model(encodings)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=1).numpy()
    preds = np.argmax(probs, axis=1)

    pred_labels.extend(preds)

# 모델 저장 경로
model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models','emotion','emotion_model_v1') # 버전 변경 v1,v2,v3 등등

# 모델 버전
model.save_pretrained(model_dir)     
tokenizer.save_pretrained(model_dir)
