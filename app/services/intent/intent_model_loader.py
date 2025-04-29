import os
from transformers import TFBertForSequenceClassification, BertTokenizer

MODEL_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..','..', 'models', 'intent')

# 모델 로딩/버전 관리 함수
def load_intent_model(version: str): # 예를 들어 version = intent_model_v1
    model_path = os.path.join(MODEL_BASE_PATH, version) 
    model = TFBertForSequenceClassification.from_pretrained(model_path) 
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer
