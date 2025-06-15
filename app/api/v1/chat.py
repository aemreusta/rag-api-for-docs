from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import verify_api_key
from app.core.query_engine import query_engine
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, _: str = Depends(verify_api_key)) -> ChatResponse:
    try:
        response = await query_engine.query(
            query_text=request.message, chat_history=request.chat_history
        )
        return ChatResponse(message=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
