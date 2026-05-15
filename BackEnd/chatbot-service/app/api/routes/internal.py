from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.services import position_score_cache

settings = get_settings()
router = APIRouter()


class UpdateScoreRequest(BaseModel):
    score: float


@router.put("/internal/positions/{position_id}/minimum-fit-score")
async def update_position_score(
    position_id: int,
    body: UpdateScoreRequest,
    x_internal_service: str = Header(None),
) -> dict:
    if x_internal_service != settings.INTERNAL_SERVICE_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    position_score_cache.set_score(position_id, body.score)
    return {"status": "ok"}
