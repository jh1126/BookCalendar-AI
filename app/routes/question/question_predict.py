from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from konlpy.tag import Okt
import os, json, torch, re, random
from datetime import datetime
from .database import get_connection

router = APIRouter()

# 요청 데이터 포맷
class TextInput(BaseModel):
    paragraph: str

# 형태소 분석기 초기화
okt = Okt()

# 경로 기준: 프로젝트 루트
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "question", "processed", "question_data.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "app", "models", "question", "question_model_run.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "data", "question", "question_model_metrics.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "question")

# 템플릿 로딩
with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_data = json.load(f)  # 리스트 구조에 맞게

# 모델 로딩 (요청 시점에 로딩되도록 처리)
def load_model_and_tokenizer():
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=500, detail="question_model_run.json 파일이 없습니다.")

    with open(CONFIG_PATH, encoding='utf-8') as f:
        model_info = json.load(f)

    model_name = model_info[0]['model_name'] if isinstance(model_info, list) else model_info['model_name']
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise HTTPException(status_code=500, detail=f"모델 디렉토리 {model_path}에 HuggingFace 모델이 없습니다.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    return tokenizer, model

#질문 수 조절
def get_question_count():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(METRICS_PATH):
        return 2  # 기본값
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
    tokenizer, model = load_model_and_tokenizer()
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

# 질문 생성 함수 (개수 조절 포함)
def generate_questions_from_template(summary, target_count=2):
    keywords = extract_keywords_okt(summary)

    candidates = [q['template'] for q in template_data["questions"] if "(키워드)" in q['template']]
    random.shuffle(keywords)
    questions = []

    for kw in keywords:
        if not candidates:
            break
        template = random.choice(candidates)
        questions.append(adjust_postposition(kw, template))
        if len(questions) >= target_count:
            break

    if len(questions) < target_count:
        raise HTTPException(status_code=500, detail=f"질문이 {target_count}개 생성되지 않았습니다.")

    return questions

# FastAPI 라우터
@router.post("/predict_question")
def predict(input_data: TextInput):
    paragraph = " ".join(input_data.paragraph.split())
    try:
        q_num = get_question_count()

        summary = summarize_kobart(paragraph)
        questions = generate_questions_from_template(summary, target_count=q_num)

        today = datetime.utcnow().strftime("%Y-%m-%d")
        output_text = " / ".join(questions)
        sql = "INSERT INTO questionData (date, input, output) VALUES (%s, %s, %s)"
        values = (today, paragraph, output_text)

        try:
            conn = get_connection()
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, values)
        except Exception as db_err:
            raise HTTPException(status_code=500, detail=f"DB 저장 실패: {db_err}")
        
        return {f"question{i+1}": q for i, q in enumerate(questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

