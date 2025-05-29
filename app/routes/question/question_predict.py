from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
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

tokenizer_sbert = AutoTokenizer.from_pretrained("jhgan/ko-sbert-sts")
model_sbert = AutoModel.from_pretrained("jhgan/ko-sbert-sts")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_sbert.to(device)
model_sbert.eval()

def get_embedding(text):
    inputs = tokenizer_sbert(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model_sbert(**inputs)
        return output.last_hidden_state[:, 0, :].cpu().numpy()

template_texts_clean = [q["template"].replace("(키워드)", "").strip() for q in template_data["questions"]]
template_embeddings = [get_embedding(t)[0] for t in template_texts_clean]

def load_model_and_tokenizer():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        model_info = json.load(f)
    model_name = model_info[0]['model_name'] if isinstance(model_info, list) else model_info['model_name']
    model_path = os.path.join(MODELS_DIR, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device).eval()
    return tokenizer, model

def summarize_kobart(text):
    tokenizer, model = load_model_and_tokenizer()
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
    input_text = f"{keyword}는 어떤 의미를 가지고 있나요?"
    kw_emb = get_embedding(input_text)[0].reshape(1, -1)
    similarities = cosine_similarity(kw_emb, template_embeddings)[0]
    scored_templates = [
        (template_data["questions"][i]["template"], score)
        for i, score in enumerate(similarities) if score >= 0.4
    ]
    random.shuffle(scored_templates)
    return [tpl for tpl, _ in scored_templates[:5]]

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

