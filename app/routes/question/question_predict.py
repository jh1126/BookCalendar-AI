from fastapi import APIRouter, HTTPException
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast, BartForConditionalGeneration
)
from pydantic import BaseModel
import torch.nn.functional as F
import re, random, json, torch, os
from datetime import datetime
from database import get_connection
import traceback

from fastapi.responses import JSONResponse

router = APIRouter()

class TextInput(BaseModel):
    paragraph: str

okt = Okt()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "question", "processed", "question_data.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "app", "models", "question", "question_model_run.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "data", "question", "question_model_metrics.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "question")

with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_data = json.load(f)

category_keywords = { ... }  # 생략된 카테고리 내용 동일

def load_kobart_model_and_tokenizer():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        model_info = json.load(f)
    model_name = model_info[0]['model_name'] if isinstance(model_info, list) else model_info['model_name']
    model_path = os.path.join(MODELS_DIR, model_name)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device).eval()
    return tokenizer, model

model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sbert_model = AutoModel.from_pretrained(model_name)
sbert_model.to(device).eval()

def get_sbert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = sbert_model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return F.normalize(cls_emb, p=2, dim=1).squeeze(0).to(device)

template_texts_clean = [
    q["template"].replace("(키워드)", "").strip()
    for q in template_data.get("questions", [])
]
template_embeddings = torch.stack([
    get_sbert_embedding(text) for text in template_texts_clean
]).to(device)

kobart_tokenizer, kobart_model = load_kobart_model_and_tokenizer()

def summarize_kobart(text):
    input_ids = kobart_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_ids = kobart_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return kobart_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# 이하 나머지 함수들은 기존과 동일하되, torch 관련 연산들은 device를 일관되게 사용하도록 모두 수정
# 핵심은 get_sbert_embedding에서 device 고정, template_embeddings 생성 시 device 설정, generate_and_refine_questions 내 torch 연산 전부 device 일치시킴

# 최종 API 라우터도 변경 없음
@router.post("/predict_question")
def predict(input_data: TextInput):
    paragraph = input_data.paragraph.strip()
    if len(paragraph) < 30:
        raise HTTPException(status_code=422, detail="문장이 너무 짧습니다.")
    try:
        summary = summarize_kobart(paragraph)
        q_num = get_question_count()
        questions = generate_and_refine_questions(
            summary, template_data, template_embeddings, sbert_model, target_count=q_num
        )
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO questionData (date, input, output) VALUES (%s, %s, %s)"
                cursor.execute(sql, (datetime.now().strftime("%Y-%m-%d"), paragraph, summary))
            conn.commit()
        finally:
            conn.close()

        return JSONResponse(content={
            "summary": summary,
            **{f"question{i+1}": q for i, q in enumerate(questions)}
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"질문 생성 중 오류 발생: {e}")
