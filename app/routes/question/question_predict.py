from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from konlpy.tag import Okt
from fastapi.responses import JSONResponse
import os, json, torch, re, random, traceback
from collections import defaultdict
from datetime import datetime
from database import get_connection

router = APIRouter()

class TextInput(BaseModel):
    paragraph: str

okt = Okt()

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "question", "processed", "question_data.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "app", "models", "question", "question_model_run.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "data", "question", "question_model_metrics.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "question")

with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_data = json.load(f)

category_keywords = {
    "감정탐구": ["감정", "느낌", "마음", "상실", "기쁨", "분노"],
    "관점전환": ["시선", "반대", "다름", "차이", "경계"],
    "메타인지": ["깨달음", "성찰", "되돌아봄", "인지", "생각"],
    "비판적사고": ["갈등", "문제", "관습", "편견", "논리", "현실"],
    "상상력발휘": ["우주", "상상", "꿈", "미래", "창의"],
    "시대와맥락": ["과거", "역사", "시대", "문화", "맥락"],
    "심층주제파고들기": ["본질", "핵심", "중심", "주제", "의미"],
    "연결성찾기": ["관계", "연결", "관련", "비교", "유사"],
    "윤리적고민": ["윤리", "선악", "선택", "가치", "판단"],
    "인물심층분석": ["인물", "성격", "행동", "동기", "성장"],
    "장르분석": ["판타지", "추리", "로맨스", "SF", "서사"],
    "창의적재해석": ["재해석", "다시보기", "의외성", "반전", "창조"],
    "핵심가치": ["자유", "사랑", "존중", "책임", "공감"],
    "행동유도": ["실천", "행동", "도전", "참여", "변화"]
}

def load_model_and_tokenizer():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        model_info = json.load(f)
    model_name = model_info[0]['model_name'] if isinstance(model_info, list) else model_info['model_name']
    model_path = os.path.join(MODELS_DIR, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device

def summarize_kobart(text):
    tokenizer, model, device = load_model_and_tokenizer()
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def adjust_postposition(keyword, template):
    code = ord(keyword[-1])
    has_final = (code - 0xAC00) % 28 != 0 if 0xAC00 <= code <= 0xD7A3 else False
    replacements = {
        r"\(이\)가": "이" if has_final else "가",
        r"\(을\)를": "을" if has_final else "를",
        r"\(은\)는": "은" if has_final else "는",
        r"\(과\)와": "과" if has_final else "와",
        r"\(이\)란": "이란" if has_final else "란"
    }
    for pattern, repl in replacements.items():
        template = re.sub(pattern, repl, template)
    return template.replace("(키워드)", keyword).strip()

def extract_keywords_okt(text, top_k=5):
    raw_nouns = okt.nouns(text)
    stopwords = {"것", "정말", "진짜", "그냥", "이런", "저런", "너무", "매우", "좀", "거의", "등", "수", "때"}
    return [kw for kw in raw_nouns if kw not in stopwords and len(kw) > 1][:top_k]

def find_similar_templates_for_keyword(keyword):
    for category, sub_keywords in category_keywords.items():
        if keyword in sub_keywords:
            return [q['template'] for q in template_data['questions'] if q['category'] == category][:5]
    return []

def get_question_count():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(METRICS_PATH):
        return 2
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            model_info = json.load(f)
        current_model = model_info[0]['model_name'] if isinstance(model_info, list) else model_info['model_name']
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        for entry in metrics:
            if entry['model_name'] == current_model and 'q_num' in entry:
                return entry['q_num']
    except:
        pass
    return 2

def generate_questions(summary, target_count=5):
    keywords = extract_keywords_okt(summary, top_k=5)
    questions = []
    used_templates = set()
    for kw in keywords:
        templates = find_similar_templates_for_keyword(kw)
        if not templates:
            templates = ["(키워드)은 당신에게 어떤 의미인가요?"]
        for tpl in templates:
            if tpl in used_templates:
                continue
            question = adjust_postposition(kw, tpl)
            if question not in questions:
                questions.append(question)
                used_templates.add(tpl)
            if len(questions) >= target_count:
                return questions
    while len(questions) < target_count:
        questions.append("(자연스러운 질문 생성 실패)")
    return questions

@router.post("/predict_question")
def predict(input_data: TextInput):
    paragraph = input_data.paragraph.strip()
    if len(paragraph) < 30:
        raise HTTPException(status_code=422, detail="문장이 너무 짧습니다.")
    try:
        summary = summarize_kobart(paragraph)
        q_num = get_question_count()
        questions = generate_questions(summary, target_count=q_num)

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

