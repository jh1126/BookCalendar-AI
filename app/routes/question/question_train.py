from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os, json, torch, random, re
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from konlpy.tag import Okt

router = APIRouter()
okt = Okt()

# 요청 형식 정의
class ParagraphRequest(BaseModel):
    paragraph: str

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
QUESTION_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, 'data', 'question', 'processed', 'question_data.json')
SUMMARY_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'summary', 'kobart_summary_model_v6')  # 모델 경로는 수정 가능

# 요약 모델 로딩
tokenizer = PreTrainedTokenizerFast.from_pretrained(SUMMARY_MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(SUMMARY_MODEL_PATH)

# 템플릿 불러오기
with open(QUESTION_TEMPLATE_PATH, encoding="utf-8") as f:
    question_templates = json.load(f)["questions"]

# 불용어 및 조사 제거 함수
def clean_keywords(keywords):
    stopwords = {"것", "정말", "진짜", "그냥", "너무", "매우", "좀", "거의", "나", "너", "우리", "저", "그", "이", "때", "중", "수", "등"}
    cleaned = []
    for kw in keywords:
        kw = re.sub(r"(의|에|에서|으로|로|와|과|는|은|가|이|를|을)$", "", kw)
        if kw not in stopwords and len(kw) > 1:
            cleaned.append(kw)
    return cleaned

# 조사 보정 함수
def adjust_postposition(keyword, template):
    last_char = keyword[-1]
    has_final = (ord(last_char) - 44032) % 28 != 0
    postpositions = {
        "(이)가": "이" if has_final else "가",
        "(을)를": "을" if has_final else "를",
        "(은)는": "은" if has_final else "는",
        "(과)와": "과" if has_final else "와",
        "(이)란": "이란" if has_final else "란",
        "(에)": "에"
    }
    for token, replacement in postpositions.items():
        template = template.replace(token, replacement)
    return template.replace("(키워드)", keyword).strip()

# 요약 함수
def summarize_kobart(text: str) -> str:
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 키워드 추출 (OKT + 필터링)
def extract_keywords(text: str, top_k=5):
    nouns = okt.nouns(text)
    cleaned = clean_keywords(nouns)
    return cleaned[:top_k]

# 질문 생성
def generate_questions(summary: str):
    keywords = extract_keywords(summary)
    random.shuffle(keywords)

    results = []
    for kw in keywords:
        templates = [q["template"] for q in question_templates if "(키워드)" in q["template"]]
        if not templates:
            continue
        template = random.choice(templates)
        adjusted = adjust_postposition(kw, template)
        results.append(adjusted)
        if len(results) == 2:
            break

    if len(results) < 2:
        raise HTTPException(status_code=500, detail="질문이 충분히 생성되지 않았습니다.")

    return results

# 엔드포인트
@router.post("/predict_question")
def predict(input_data: ParagraphRequest):
    try:
        summary = summarize_kobart(input_data.paragraph)
        questions = generate_questions(summary)
        return {
            "summary": summary,
            "question1": questions[0],
            "question2": questions[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 오류: {e}")
