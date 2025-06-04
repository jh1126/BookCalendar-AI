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

#독후감 input 데이터
class TextInput(BaseModel):
    paragraph: str

okt = Okt()

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "question", "processed", "question_data.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "app", "models", "question", "question_model_run.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "data", "question", "question_model_metrics.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "question")



#질문 템플릿 로딩
with open(TEMPLATE_PATH, encoding="utf-8") as f:
    template_data = json.load(f)


#디바이스 설정 - cuda를 하드코딩 x, device 변수에 통일 처리
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#카테고리-키워드 사전 ( 질문 생성 시 키워드 의미적 분류 기준 '상실'-> 감정탐구 )
category_keywords = {
    "감정탐구": [
        "감정", "느낌", "마음", "상실", "기쁨", "분노", "두려움", "외로움", "설렘", "불안", 
        "안도", "희망", "절망", "사랑", "증오", "우울", "열정", "수치심", "혐오", "놀람"
    ],
    "관점전환": [
        "시선", "반대", "다름", "차이", "경계", "관점", "입장", "바라보기", "시야", 
        "비유", "편견해소", "다각도", "비판적시각", "다른생각", "생각의유연성"
    ],
    "메타인지": [
        "깨달음", "성찰", "되돌아봄", "인지", "생각", "자각", "내면", "반성", "정신적성장", 
        "객관화", "의식", "자기이해", "사고의과정", "스스로", "지각변화"
    ],
    "비판적사고": [
        "갈등", "문제", "관습", "편견", "논리", "현실", "의문", "의심", "반박", 
        "분석", "논쟁", "비판", "허점", "추론", "사실확인"
    ],
    "상상력발휘": [
        "우주", "상상", "꿈", "미래", "창의", "환상", "가상", "비현실", "이상향", 
        "기발함", "모험", "변형", "발명", "다른세계", "공상"
    ],
    "시대와맥락": [
        "과거", "역사", "시대", "문화", "맥락", "전통", "세대", "시기", "배경", 
        "진보", "보수", "시대정신", "사회변화", "흐름", "시대상"
    ],
    "심층주제파고들기": [
        "본질", "핵심", "중심", "주제", "의미", "내포", "근원", "의도", "구조", 
        "깊이", "중요성", "주제의식", "핵심질문", "핵심사상", "초점"
    ],
    "연결성찾기": [
        "관계", "연결", "관련", "비교", "유사", "대조", "상관", "접점", "유추", 
        "맥락", "연속성", "유기성", "함의", "상호작용", "원인결과"
    ],
    "윤리적고민": [
        "윤리", "선악", "선택", "가치", "판단", "책임", "양심", "도덕", "규범", 
        "옳고그름", "갈등상황", "정의", "인간다움", "사회규범", "공정성"
    ],
    "인물심층분석": [
        "인물", "성격", "행동", "동기", "성장", "변화", "배경", "관계", "말투", 
        "심리", "트라우마", "욕망", "결정", "역할", "한계"
    ],
    "장르분석": [
        "판타지", "추리", "로맨스", "SF", "서사", "형식", "플롯", "클리셰", "장르특성", 
        "장르규칙", "서사방식", "서브장르", "문체", "분위기", "장르해체"
    ],
    "창의적재해석": [
        "재해석", "다시보기", "의외성", "반전", "창조", "재구성", "해체", "새로운시각", 
        "기존틀깨기", "응용", "독창성", "탈구조", "발명", "다시쓰기", "전복"
    ],
    "핵심가치": [
        "자유", "사랑", "존중", "책임", "공감", "연대", "배려", "용기", "평등", 
        "자아", "자율", "정의", "정직", "진실", "신뢰"
    ],
    "행동유도": [
        "실천", "행동", "도전", "참여", "변화", "결단", "해결", "계획", "습관", 
        "리더십", "영향력", "자기주도", "사회참여", "의지", "실행"
    ]
}





#KoBART 모델 로딩
def load_kobart_model_and_tokenizer():
    # 현재 사용 중인 모델 이름 가져오기
    with open(CONFIG_PATH, encoding='utf-8') as f:
        model_info = json.load(f)
    model_name = model_info[0]['model_name'] if isinstance(model_info, list) else model_info['model_name']

    # 모델 경로 구성
    model_path = os.path.join(MODELS_DIR, model_name)

    # run.json 모델의 Tokenizer, Model 로드(.eval()을 통해 추론 모드)
    #Fastapi 재시작 없이 모델 교체 가능
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(model_path, "tokenizer.json"),
        bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>'
    )
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device).eval()

    return tokenizer, model


#SBERT 모델 로딩 및 설정 - 질문 템플릿 유사도 비교에 이용
#템플릿과 키워드 간 문장 임베딩 기반 유사도 측정 지원
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sbert_model = AutoModel.from_pretrained(model_name)
sbert_model.to("cuda")
sbert_model.eval()


#SBERT 임베딩 추출 함수
#TEXT를 토크나이징 -> sbert에 입력 -> 토큰 백터 추출
def get_sbert_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)  # ✅ 하드코딩된 "cuda" → device 변수로 통일

    with torch.no_grad():
        outputs = sbert_model(**inputs)

    #normalize를 적용해 코사인 유사도에 적합한 단위 벡터로 변환
    cls_emb = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_dim)
    return F.normalize(cls_emb, p=2, dim=1).squeeze(0).to(device)  # ✅ .to(device) 추가로 명시



# 1. (키워드) 제거한 템플릿 문장 리스트
template_texts_clean = [
    q["template"].replace("(키워드)", "").strip()
    for q in template_data.get("questions", [])
]

# 2. SBERT 임베딩 추출 후 스택 (GPU로 이동)
#template_embaddings에 저장 돼 한 번만 임베딩하고, 키워드 임베딩, 코사인 유사도만 계산
template_embeddings = torch.stack([
    get_sbert_embedding(text) for text in template_texts_clean
]).to("cuda" if torch.cuda.is_available() else "cpu")

#요약 전리 (줄바꿈 제거, 공백 정리)
def preprocess_paragraph(text):
    return ' '.join(text.strip().split())


#요약 함수
kobart_tokenizer, kobart_model = load_kobart_model_and_tokenizer()
def summarize_kobart(text):
    input_ids = kobart_tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    output_ids = kobart_model.generate(
        input_ids,
        max_length=100,
        num_beams=4, #다양한 후보를 봄
        early_stopping=True
    )
    #디코딩 : 숫자 ID를 다시 한글 문장으로 바꿈
    return kobart_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

#조사 보정 함수
def adjust_postposition(keyword, template):
    #keyword 마지막 글자 받침 유무 판별
    def has_jongseong(char):
        code = ord(char)
        return (code - 0xAC00) % 28 != 0 if 0xAC00 <= code <= 0xD7A3 else False

    #판별한 내용을 바탕으로 템플릿 내 조사 자동 조정
    has_final = has_jongseong(keyword[-1])
    replacements = {
        r"\(이\)가": "이" if has_final else "가",
        r"\(을\)를": "을" if has_final else "를",
        r"\(은\)는": "은" if has_final else "는",
        r"\(과\)와": "과" if has_final else "와",
        r"\(이\)라고": "이라고" if has_final else "라고",
    }

    #키워드를 실제 키워드로 다체 후 반환
    for pattern, repl in replacements.items():
        template = re.sub(pattern, repl, template)
    return template.replace("(키워드)", keyword).strip()




# 주어 보정 함수
def clarify_subject(question, keyword):
    abstract_keywords = {"감정", "내면", "의미", "생각", "존재", "성찰", "느낌"}
    human_keywords = {"저자", "주인공", "인물", "사람"}

    replacement = None

    #문장 내 특정 문구가 있을 때 보완
    if any(phrase in question for phrase in ["하고 싶어", "만들고 싶어", "느끼고 싶어", "생각하나요", "중요한가요"]):
        if keyword in abstract_keywords:
            replacement = f"저자의 {keyword}"
        elif keyword in human_keywords:
            replacement = f"{keyword} 본인은"

    # 단어가 포함된 경우, 처음 한 번만 대체
    if replacement:
        question = re.sub(rf"\b{re.escape(keyword)}\b", replacement, question, count=1)

    # "무엇인가요?" → "어떤 의미인가요?" 로 더 자연스럽게
    if "무엇인가요?" in question and keyword in abstract_keywords:
        question = question.replace("무엇인가요?", "어떤 의미인가요?")

    return question


# 불필요한 키워드 제거
def clean_keywords(keywords):
    stopwords = {
        "것", "정말", "진짜", "그냥", "이런", "저런", "너무", "매우", "좀", "거의",
        "나", "너", "우리", "저", "그", "이", "위", "아래", "때", "중", "수", "등",
        "그리고", "그래서", "하지만", "그러나", "그때", "요즘", "오늘", "내일", "현실", "모든", "일상", "송두리째"
    }
    return [kw.strip() for kw in keywords if kw.strip() not in stopwords and len(kw.strip()) > 1]



#핵심 키워드 추출
def extract_keywords_okt_with_filter(text, sbert_model=None, top_k=3, threshold=0.35, verbose=True):
    #명사 추출
    raw_nouns = okt.nouns(text)
    stopwords = {
        "것", "정말", "진짜", "그냥", "이런", "저런", "너무", "매우", "좀", "거의", "등", "수", "때",
        "나", "너", "우리", "저", "그", "이", "위", "아래", "중", "그리고", "그래서", "하지만", "그러나",
        "요즘", "오늘", "내일", "무엇", "낯선"
    }
    abstract_keywords = {
        "사이", "통해", "과정", "모습", "생각", "이야기", "상황", "사실", "시간", "장면",
        "경험", "부분", "사람", "사회", "자신", "의미", "존재", "내용", "중심", "주인공"
    }
    
    filtered_nouns = [
        kw for kw in raw_nouns
        #불용어, 추상 키워드 제거
        if kw not in stopwords and kw not in abstract_keywords and len(kw.strip()) > 1
    ]
    
    freq_sorted = Counter(filtered_nouns).most_common()

    #키워드 벡터 - 카테고리 벡터 간 최대 유사도 계산
    #특정 기준(threshold)이상이면 키워드 사용
    if sbert_model:
        result = []
        #키워드 별 임베딩 생성
        for kw, _ in freq_sorted:
            kw_emb = get_sbert_embedding(kw)  # kw_emb는 GPU

            #카테고리 키워드 유사도 측정
            scores = []
            for ref_list in category_keywords.values():
                ref_embs = [get_sbert_embedding(r).to(kw_emb.device) for r in ref_list]
                ref_tensor = torch.stack(ref_embs)

                #벡터간 유사도 계산
                sim = torch.matmul(kw_emb, ref_tensor.T).max().item()
                scores.append(sim)

            #임계값 이상 키워드만 선택
            if max(scores) >= threshold:#0.35 이상이 되도록 설정
                result.append((kw, max(scores)))

        #상위 top_k개 선택 -> 의미 있는 키워들만 존재
        result = sorted(result, key=lambda x: x[1], reverse=True)[:top_k]
        final_keywords = [kw for kw, _ in result[:top_k]]

    else:
        final_keywords = [kw for kw, _ in freq_sorted[:top_k]]
    #키워드 확인
    if verbose:
        print("필터링된 키워드:", final_keywords)
    
    return final_keywords


#질문 수 받기
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


#키워드와 의미적으로 비슷한 템플릿 5개 찾기
def find_similar_templates_sbert(keyword, template_data, template_embeddings, sbert_model):
    keyword_emb = get_sbert_embedding(keyword)
    #키워드 - 템플릿 임베딩들과 코사인 유사도 계산
    sims = torch.nn.functional.cosine_similarity(keyword_emb.unsqueeze(0), template_embeddings)
    top_indices = torch.topk(sims, k=5).indices.tolist()
    #상위 5개 가져오기
    return [template_data["questions"][i]["template"] for i in top_indices]

#어색한 질문 수
def is_template_suitable(keyword, question, template):
    # 예: 갈등을 읽는다 → 어색
    if f"{keyword}을 읽" in question or f"{keyword}를 읽" in question:
        return False
    #특정 키워드 두 번 반복 어색
    if question.count(keyword) > 1:
        return False
    #이미 템플릿에 키워드가 들어있는 경우
    template_base = re.sub(r"\(키워드\)", "", template)
    if keyword in template_base:
        return False
    return True


#질문 생성 함수(최종 함수)
def generate_and_refine_questions(summary, template_data, template_embeddings, sbert_model, target_count=None, verbose=True):
    if target_count is None:
        target_count = get_question_count()
    #Okt + SBERT 기반 키워드 5개
    keywords = extract_keywords_okt_with_filter(summary, top_k=3, verbose=verbose)
    questions = []
    used_templates = set()
    keyword_usage = defaultdict(int)
    max_attempts = target_count * 10
    attempt = 0
    MAX_USE_PER_KEYWORD = 1


    # 👇 2. 커스텀 키워드 템플릿 정의
    custom_templates = {
        "닭장": "작품 속 닭장은 작품 속에서 어떤 상징을 띄나요?",
        "잎싹": "잎싹의 선택은 오늘날 우리에게 어떤 메시지를 전달하나요?",
        "마당": "마당은 작품 속에서 어떤 역할을 했다고 볼 수 있나요?"
    }

    # 👇 3. 커스텀 키워드 우선 적용
    filtered_keywords = []
    for kw in keywords:
        if kw in custom_templates and len(questions) < target_count:
            questions.append(custom_templates[kw])
            keyword_usage[kw] += 1
        else:
            filtered_keywords.append(kw)

    weights = [0.4, 0.3, 0.1, 0.1, 0.1]
    weighted_keywords = list(zip(keywords, weights))
    random.shuffle(weighted_keywords)

    while len(questions) < target_count and attempt < max_attempts:
        attempt += 1
        available_keywords = [kw for kw, _ in weighted_keywords if keyword_usage[kw] < MAX_USE_PER_KEYWORD]
        kw = random.choice(available_keywords) if available_keywords else random.choices(keywords, weights=weights, k=1)[0]
        if keyword_usage[kw] >= MAX_USE_PER_KEYWORD:
            continue
        templates = find_similar_templates_sbert(kw, template_data, template_embeddings, sbert_model)
        if not templates:
            templates = [
                "(키워드)은 당신에게 어떤 의미인가요?",
                "(키워드)을 통해 무엇을 느꼈나요?",
                "(키워드)을 다시 바라본다면 어떤 점이 보이나요?"
            ]
            force_use_fallback = True
        else:
            force_use_fallback = False
        random.shuffle(templates)
        for template in templates:
            if not force_use_fallback and template in used_templates:
                continue
            if "(키워드)" not in template:
                continue
            question_raw = adjust_postposition(kw, template)
            question = clarify_subject(question_raw, kw)
            if question.count(kw) != 1:
                continue
            if not is_template_suitable(kw, question, template):
                continue
            if not force_use_fallback and question in questions:
                continue
            questions.append(question)
            used_templates.add(template)
            keyword_usage[kw] += 1
            break


    while len(questions) < target_count:
        available_fallback_kws = [kw for kw in keywords if keyword_usage[kw] < MAX_USE_PER_KEYWORD]
        if not available_fallback_kws:
            break
        fallback_kw = random.choice(available_fallback_kws)
        fallback_template = "(키워드)은 당신에게 어떤 의미인가요?"
        question_raw = adjust_postposition(fallback_kw, fallback_template)
        question = clarify_subject(question_raw, fallback_kw)
        if question not in questions:
            questions.append(question)
            keyword_usage[fallback_kw] += 1

    return questions


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
        raise HTTPException(status_code=500, detail=f"질문 생성 중 오류 발생 : {e}")
