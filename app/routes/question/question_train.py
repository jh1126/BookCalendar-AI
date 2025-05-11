from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import os, json, torch, random, re
from konlpy.tag import Okt

router = APIRouter()
okt = Okt()

# 요청 형식
class ParagraphRequest(BaseModel):
    paragraph: str

# 경로
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'question_model_run.json')
TEMPLATE_PATH = os.path.join(ROOT_DIR, 'data', 'question', 'processed', 'question_data.json')
MODEL_BASE_PATH = os.path.join(ROOT_DIR, 'models', 'question')  # 실제 모델들이 있는 디렉토리

# 현재 설정된 요약 모델 이름 불러오기
def get_current_summary_model_name():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("model_name")
            elif isinstance(data, dict):
                return data.get("model_name")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 설정 파일 오류: {e}")
    raise HTTPException(status_code=404, detail="현재 설정된 요약 모델이 없습니다.")

# 불용어 + 조사 제거
def clean_keywords(keywords):
    stopwords = {
        "것", "정말", "진짜", "그냥", "너무", "매우", "좀", "거의",
        "나", "너", "우리", "저", "그", "이", "때", "중", "수", "등",
        "그리고", "그래서", "하지만", "그러나", "그때", "요즘", "오늘", "내일"
    }
    cleaned = []
    for kw in keywords:
        kw = re.sub(r"(의|에|에서|으로|로|와|과|는|은|가|이|를|을)$", "", kw)
        if kw not in stopwords and len(kw) > 1:
            cleaned.append(kw)
    return cleaned

# 조사 보정
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

# 템플릿 로딩
with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_data = json.load(f)["questions"]

# 요약 + 질문 생성 엔드포인트
@router.post("/predict_question")
def predict(input_data: ParagraphRequest):
    try:
        # 1. 모델 이름 로딩
        model_name = get_current_summary_model_name()
        model_path = os.path.join(MODEL_BASE_PATH, model_name)

        # 2. 요약 모델 로딩
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.eval()

        # 3. 요약 수행
        input_ids = tokenizer.encode(input_data.paragraph, return_tensors="pt", truncation=True, max_length=512)
        summary_output = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_output[0], skip_special_tokens=True)

        # 4. 키워드 추출
        nouns = okt.nouns(summary)
        keywords = clean_keywords(nouns)[:5]

        # 5. 질문 생성
        random.shuffle(keywords)
        questions = []
        for kw in keywords:
            candidates = [q["template"] for q in template_data if "(키워드)" in q["template"]]
            if not candidates:
                continue
            template = random.choice(candidates)
            adjusted = adjust_postposition(kw, template)
            questions.append(adjusted)
            if len(questions) == 2:
                break

        if len(questions) < 2:
            raise HTTPException(status_code=500, detail="질문이 충분히 생성되지 않았습니다.")

        return {
            "question1": questions[0],
            "question2": questions[1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {e}")
