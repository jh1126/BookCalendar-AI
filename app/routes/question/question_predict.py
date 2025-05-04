from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.question.question_predictor import predict_question  # 여기도 수정 필요

router = APIRouter()

class ParagraphRequest(BaseModel):
    paragraph: str

@router.post("/predict_question")
def predict(request: ParagraphRequest):
    paragraph = request.paragraph

    try:
        # 버전 없이 predict_question 함수 호출
        questions = predict_question(paragraph)

        # 질문이 2개 이상이어야 함
        if not isinstance(questions, list) or len(questions) < 2:
            raise ValueError("질문이 2개 이상 생성되지 않았습니다.")

        return {
            "question1": questions[0],
            "question2": questions[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
