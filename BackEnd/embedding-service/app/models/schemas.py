from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# CV Models
class CVChunk(BaseModel):
    """Single chunk from CV (matches Java ChunkPayload)"""
    candidateId: Optional[str] = None
    hrId: Optional[str] = None
    positionId: Optional[int] = None
    position: Optional[str] = None
    section: str
    chunkIndex: int
    chunkText: str
    words: int
    tokensEstimate: int
    email: Optional[str] = None
    cvId: int
    cvStatus: Optional[str] = None
    sourceType: Optional[str] = None
    createdAt: Optional[str] = None
    
    # Metadata fields populated by Java
    skills: List[str] = []
    experienceYears: Optional[int] = 0
    seniorityLevel: str = "Unknown"
    companies: List[str] = []
    degrees: List[str] = []
    dateRanges: List[str] = []

class CVChunkedEvent(BaseModel):
    """Payload received from cv.embed.queue"""
    cvId: int
    candidateId: Optional[str] = None
    hrId: Optional[str] = None
    position: Optional[str] = None
    chunks: List[CVChunk]
    totalChunks: int
    totalTokens: int

class EmbedReplyEvent(BaseModel):
    """Reply payload sent to cv.embed.reply.queue"""
    cvId: int
    batchId: Optional[str] = None
    success: bool
    errorMessage: Optional[str] = None
    technicalScore: Optional[int] = None
    experienceScore: Optional[int] = None
    overallStatus: Optional[str] = None


# ===== JD Models =====
class JDEmbeddingRequest(BaseModel):
    """Request to embed Job Description"""
    hrId: str
    position: str


class JDEmbeddingResponse(BaseModel):
    """Response after embedding JD"""
    jdId: int
    version: int
    message: str
    processingTime: float


# ===== Search Models =====
class SearchRequest(BaseModel):
    """Search request"""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Single search result"""
    id: str
    score: float
    payload: dict


class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: List[SearchResult]
    total: int