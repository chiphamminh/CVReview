from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.rag.candidate.candidate_graph import candidate_chatbot
from app.services.recruitment_api import recruitment_api

router = APIRouter()

class SessionCreateRequest(BaseModel):
    user_id: str
    position_id: Optional[int] = None
    
class SessionResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    query: str
    candidate_id: str
    cv_id: Optional[int] = None
    
class ChatResponse(BaseModel):
    answer: str
    metadata: Dict[str, Any]

@router.post("/candidate/session", response_model=SessionResponse, tags=["Candidate Chatbot"])
async def create_candidate_session(request: SessionCreateRequest):
    """
    Tạo session mới cho Candidate. Gọi API của recruitment-service.
    """
    try:
        res = await recruitment_api.create_session(user_id=request.user_id, chatbot_type="CANDIDATE", position_id=request.position_id)
        return SessionResponse(session_id=res["sessionId"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/candidate/chat", response_model=ChatResponse, tags=["Candidate Chatbot"])
async def candidate_chat(request: ChatRequest):
    """
    Candidate send message.
    """
    try:
        response = await candidate_chatbot.chat(
            query=request.query,
            session_id=request.session_id,
            candidate_id=request.candidate_id,
            cv_id=request.cv_id
        )
        return ChatResponse(
            answer=response["answer"],
            metadata=response["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
