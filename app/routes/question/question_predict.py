from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
from konlpy.tag import Okt
import os, json, torch, re, random
from datetime import datetime
from database import get_connection
from fastapi.responses import JSONResponse
import traceback

router = APIRouter()

# 요청 데이터 포맷
class TextInput(BaseModel):
    paragraph: str

# 형태소 분석기
okt = Okt()

# 경로 설정
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "question", "processed", "question_data.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "app", "models", "question", "question_model_run.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "data", "question", "question_model_metrics.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "question")

# 템플릿 로드
with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_data = json.load(f)

# 모델 로딩
def load_model_and_tokenizer():
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=500, detail="question_model_run.json 파일이 없습니다.")
    with open(CONFIG_PATH, encoding='utf-8') as f:
        model_info = json.load(f)
    model_name = model_info[0]['model_name'] if isinstance(model_info, list) else model_info['model_name']
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise HTTPException(status_code=500, detail=f"모델 디렉토리 {model_path}에 HuggingFace 모델이 없습니다.")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

# 질문 수 조절
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

# 불용어 제거
def clean_keywords(keywords):
    stopwords = {
        "것", "정말", "진짜", "그냥", "이런", "저런", "너무", "매우", "좀", "거의",
        "나", "너", "우리", "저", "그", "이", "위", "아래", "때", "중", "수", "등",
        "그리고", "그래서", "하지만", "그러나", "그때", "요즘", "오늘", "내일"
    }
    return [kw.strip() for kw in keywords if kw.strip() not in stopwords and len(kw.strip()) > 1]

# ✅ 요약 함수 (출력 포함)
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
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ✅ 요약 결과 출력
    print(f"\n📘 요약 결과:\n{summary}\n")
    return summary

# 키워드 추출
def extract_keywords_okt(text, top_k=5):
    nouns = okt.nouns(text)
    cleaned = clean_keywords(nouns)
    return cleaned[:top_k]

# 조사 보정
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

# 질문 생성
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

# FastAPI 라우트
@router.post("/predict_question")
def predict(input_data: TextInput):
    paragraph = " ".join(input_data.paragraph.split())
    if len(paragraph) <= 30:
        raise HTTPException(status_code=422, detail="문장이 너무 적습니다.")
    try:
        q_num = get_question_count()
        summary = summarize_kobart(paragraph)
        questions = generate_questions_from_template(summary, target_count=q_num)

        date_str = datetime.now().strftime("%Y-%m-%d")
        output_text = " / ".join(questions)

        # DB 저장
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO questionData (date, input, output) VALUES (%s, %s, %s)"
                cursor.execute(sql, (date_str, paragraph, summary))
            conn.commit()
        finally:
            conn.close()

        # ✅ summary도 응답에 포함하고 싶다면 아래로 변경 가능
        return JSONResponse(content={
            "summary": summary,
            **{f"question{i+1}": q for i, q in enumerate(questions)}
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"질문 생성 중 오류 발생: {e}")
