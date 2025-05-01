from fastapi import APIRouter
from app.services.intent.intent_model_loader import load_intent_model
from app.models import intent_model

# 모델 버전 선택 라우터
router = APIRouter()

class TextInput(BaseModel):
    intentModelLoad: str
    
@router.post("/set_intent")
def set_model_version(data: TextInput):
    model, tokenizer = load_intent_model(data.intentModelLoad)
    intent_model.current_model = model
    intent_model.current_tokenizer = tokenizer
    intent_model.current_version = version
    return {"message": f"질문 의도 분류 모델 버전 {version}이 로드되었습니다."}
