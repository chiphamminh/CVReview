from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.services.qdrant import qdrant_service
from app.config import get_settings

settings = get_settings()
router = APIRouter()


class JDMetadataUpdate(BaseModel):
    positionTitle: str
    seniority: str


@router.put("/{position_id}/metadata")
async def update_jd_metadata(position_id: int, body: JDMetadataUpdate):
    """Update positionTitle and seniority payload for all JD chunks of a position."""
    success = qdrant_service.update_jd_metadata(
        collection_name=settings.JD_COLLECTION_NAME,
        position_id=position_id,
        position_title=body.positionTitle,
        seniority=body.seniority,
    )
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update metadata for JD {position_id}",
        )
    return {"message": f"Updated metadata for JD {position_id}", "positionId": position_id}


@router.delete("/{position_id}")
async def delete_jd_embeddings(position_id: int):
    """
    Delete all embeddings for a JD (all versions)
    
    Args:
        position_id: Position/JD ID to delete
        
    Returns:
        Success message
    """
    try:
        # Delete all points with this positionId
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="positionId",
                    match=MatchValue(value=position_id)
                )
            ]
        )
        
        success = qdrant_service.delete_by_filter(
            collection_name=settings.JD_COLLECTION_NAME,
            filters=filter_condition
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete embeddings for JD {position_id}"
            )
        
        return {
            "message": f"Successfully deleted all embeddings for JD {position_id}",
            "positionId": position_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting JD {position_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting JD embeddings: {str(e)}"
        )