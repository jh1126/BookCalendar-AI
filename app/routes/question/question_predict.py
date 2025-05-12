from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from konlpy.tag import Okt
import os, json, torch, re, random

router = APIRouter()

# 요청 데이터 포맷
class ParagraphRequest(BaseModel):
    paragraph: str

# 형태소 분석기 및 키워드 추출기 초기화
okt = Okt()

# 템플릿 로딩
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'question', 'processed', 'question_data.json')
TEMPLATE_PATH = os.path.abspath(TEMPLATE_PATH)

with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_data = json.load(f)["questions"]

# 모델 불러오기
CONFIG_PATH = os.path.join(BASE_DIR, 'data', 'question', 'question_model_run.json')
with open(CONFIG_PATH, encoding='utf-8') as f:
    model_info = json.load(f)
model_name = model_info[0]['model_name'] if isinstance(model_info, list) else model_info['model_name']
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'question', model_name)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Stopwords 제거
def clean_keywords(keywords):
    stopwords = {
        "것", "정말", "진짜", "그냥", "이런", "저런", "너무", "매우", "좀", "거의",
        "나", "너", "우리", "저", "그", "이", "위", "아래", "때", "중", "수", "등",
        "그리고", "그래서", "하지만", "그러나", "그때", "요즘", "오늘", "내일"
    }
    return [kw.strip() for kw in keywords if kw.strip() not in stopwords and len(kw.strip()) > 1]

# 요약 함수
def summarize_kobart(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    output_ids = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 키워드 추출 함수
def extract_keywords_okt(text, top_k=5):
    nouns = okt.nouns(text)
    cleaned = clean_keywords(nouns)
    return cleaned[:top_k]

# 조사 보정 템플릿 적용 함수
def adjust_postposition(keyword, template):
    last_char = keyword[-1]
    try:
        code = ord(last_char)
        has_final = (code - 44032) % 28 != 0 if 0xAC00 <= code <= 0xD7A3 else False
    except:
        has_final = False

    postpositions = {
        r"\(이\)가": "이" if has_final else "가",
        r"\(을\)를": "을" if has_final else "를",
        r"\(은\)는": "은" if has_final else "는",
        r"\(과\)와": "과" if has_final else "와",
        r"\(이\)란": "이란" if has_final else "란",
        r"\(에\)": "에"
    }
    for pattern, replacement in postpositions.items():
        template = re.sub(pattern, replacement, template)

    return template.replace("(키워드)", keyword).replace("  ", " ").strip()

# 질문 생성 함수
def generate_questions_from_template(paragraph):
    summary = summarize_kobart(paragraph)
    keywords = extract_keywords_okt(summary)

    candidates = [q['template'] for q in template_data if "(키워드)" in q['template']]
    random.shuffle(keywords)
    questions = []

    for kw in keywords:
        if not candidates:
            break
        template = random.choice(candidates)
        questions.append(adjust_postposition(kw, template))
        if len(questions) == 2:
            break

    if len(questions) < 2:
        raise HTTPException(status_code=500, detail="질문이 2개 생성되지 않았습니다.")

    return questions

# FastAPI 라우터
@router.post("/predict_question")
def predict(input_data: ParagraphRequest):
    paragraph = " ".join(input_data.paragraph.split())
    try:
        questions = generate_questions_from_template(paragraph)
        return {"question1": questions[0], "question2": questions[1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
