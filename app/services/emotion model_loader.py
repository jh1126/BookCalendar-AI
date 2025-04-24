import os
from transformers import TFBertForSequenceClassification, BertTokenizer

MODEL_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'intent')

def load_model(version: str):
    model_path = os.path.join(MODEL_BASE_PATH, version)
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer
