"""
HR Chatbot routes.
FE calls these endpoints (via API Gateway) to create HR sessions and send messages.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal, Dict, Any

from app.rag.hr.hr_graph import hr_chatbot
from app.services.recruitment_api import recruitment_api

router = APIRouter()


class HRSessionCreateRequest(BaseModel):
    hr_id: str
    position_id: int
    mode: Literal["HR_MODE", "CANDIDATE_MODE"]


class HRSessionResponse(BaseModel):
    session_id: str


class HRChatRequest(BaseModel):
    session_id: str
    query: str
    hr_id: str
    position_id: int
    mode: Literal["HR_MODE", "CANDIDATE_MODE"]


class HRChatResponse(BaseModel):
    answer: str
    metadata: Dict[str, Any]


@router.post("/hr/session", response_model=HRSessionResponse, tags=["HR Chatbot"])
async def create_hr_session(request: HRSessionCreateRequest):
    """
    Khởi tạo session mới cho HR chatbot.
    FE gọi endpoint này khi HR chọn một Position và Mode để bắt đầu phiên chat.
    """
    try:
        res = await recruitment_api.create_session(
            user_id=request.hr_id,
            chatbot_type="HR",
            position_id=request.position_id,
            mode=request.mode
        )
        return HRSessionResponse(session_id=res["sessionId"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hr/chat", response_model=HRChatResponse, tags=["HR Chatbot"])
async def hr_chat(request: HRChatRequest):
    """
    Gửi message từ HR và nhận phản hồi từ chatbot.
    session_id, position_id và mode phải khớp với session đã tạo.
    """
    try:
        response = await hr_chatbot.chat(
            query=request.query,
            session_id=request.session_id,
            hr_id=request.hr_id,
            position_id=request.position_id,
            mode=request.mode
        )
        return HRChatResponse(
            answer=response["answer"],
            metadata=response["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
