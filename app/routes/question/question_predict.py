from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from konlpy.tag import Okt
from collections import defaultdict, Counter
from datetime import datetime
import os, json, torch, re, traceback, random
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from database import get_connection

router = APIRouter()

# 입력 모델
class TextInput(BaseModel):
    paragraph: str

# 경로 설정
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
KOBART_PATH = os.path.join(PROJECT_ROOT, "models", "kobart_summary_model_v6")
SBERT_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "question", "processed", "question_data.json")

# 모델 불러오기
def load_model_and_tokenizer():
    model_path = "/home/t25101/v0.5/ai/BookCalendar-AI/models/kobart_summary_model_v6"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


sbert_tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL_NAME)
sbert_model = AutoModel.from_pretrained(SBERT_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert_model.to(device).eval()

# 템플릿 불러오기
with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_data = json.load(f)

template_texts_clean = [q["template"].replace("(키워드)", "").strip() for q in template_data["questions"]]

# 템플릿 임베딩
def get_sbert_embedding(text):
    inputs = sbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sbert_model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return F.normalize(cls_emb, p=2, dim=1).squeeze(0)

template_embeddings = torch.stack([
    get_sbert_embedding(text) for text in template_texts_clean
]).to(device)

# 기타 함수들
okt = Okt()

def summarize_kobart(text, tokenizer, model, device):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def adjust_postposition(keyword, template):
    def has_jongseong(char):
        code = ord(char)
        return (code - 0xAC00) % 28 != 0 if 0xAC00 <= code <= 0xD7A3 else False

    has_final = has_jongseong(keyword[-1])
    replacements = {
        r"\(이\)가": "이" if has_final else "가",
        r"\(을\)를": "을" if has_final else "를",
        r"\(은\)는": "은" if has_final else "는",
        r"\(과\)와": "과" if has_final else "와",
        r"\(이\)라고": "이라고" if has_final else "라고",
    }

    for pattern, repl in replacements.items():
        template = re.sub(pattern, repl, template)
    return template.replace("(키워드)", keyword).strip()

def clarify_subject(question, keyword):
    abstract_keywords = {"감정", "내면", "의미", "생각", "존재", "성찰", "느낌"}
    human_keywords = {"저자", "주인공", "인물", "사람"}

    replacement = None
    if any(phrase in question for phrase in ["하고 싶어", "만들고 싶어", "느끼고 싶어", "생각하나요", "중요한가요"]):
        if keyword in abstract_keywords:
            replacement = f"저자의 {keyword}"
        elif keyword in human_keywords:
            replacement = f"{keyword} 본인은"
    if replacement:
        question = re.sub(rf"\b{re.escape(keyword)}\b", replacement, count=1, string=question)
    if "무엇인가요?" in question and keyword in abstract_keywords:
        question = question.replace("무엇인가요?", "어떤 의미인가요?")
    return question

def extract_keywords(text, top_k=5):
    stopwords = {
        "것", "정말", "진짜", "그냥", "이런", "저런", "너무", "매우", "좀", "거의",
        "나", "너", "우리", "저", "그", "이", "위", "아래", "때", "중", "수", "등",
        "그리고", "그래서", "하지만", "그러나", "그때", "요즘", "오늘", "내일"
    }
    nouns = okt.nouns(text)
    filtered = [kw for kw in nouns if kw not in stopwords and len(kw) > 1]
    return [kw for kw, _ in Counter(filtered).most_common(top_k)]

def find_similar_templates_sbert(keyword):
    keyword_emb = get_sbert_embedding(keyword)
    sims = F.cosine_similarity(keyword_emb.unsqueeze(0), template_embeddings)
    top_indices = torch.topk(sims, k=5).indices.tolist()
    return [template_data["questions"][i]["template"] for i in top_indices]

def generate_questions(summary, target_count=5):
    keywords = extract_keywords(summary, top_k=5)
    questions = []
    used_templates = set()
    for kw in keywords:
        templates = find_similar_templates_sbert(kw)
        random.shuffle(templates)
        for tpl in templates:
            if tpl in used_templates or "(키워드)" not in tpl:
                continue
            question = clarify_subject(adjust_postposition(kw, tpl), kw)
            if question not in questions:
                questions.append(question)
                used_templates.add(tpl)
            if len(questions) >= target_count:
                break
        if len(questions) >= target_count:
            break
    return questions

def select_best_questions(summary, questions, top_k=2):
    summary_emb = get_sbert_embedding(summary)
    question_embs = [get_sbert_embedding(q) for q in questions]
    sims = [F.cosine_similarity(summary_emb, q_emb, dim=0).item() for q_emb in question_embs]
    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    return [questions[i] for i in top_indices]
    
@router.post("/predict_question")
def predict_question(input_data: TextInput):
    tokenizer, model, device = load_model_and_tokenizer()
    paragraph = input_data.paragraph.strip()
    summary = summarize_kobart(paragraph, tokenizer, model, device)
    if len(paragraph) < 30:
        raise HTTPException(status_code=422, detail="문장이 너무 짧습니다.")
    try:
        raw_questions = generate_questions(summary, target_count=5)
        best_questions = select_best_questions(summary, raw_questions, top_k=2)

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
            **{f"question{i+1}": q for i, q in enumerate(best_questions)}
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"질문 생성 중 오류 발생: {str(e)}")


