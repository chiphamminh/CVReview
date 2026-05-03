from typing import TypedDict, Literal, Optional, List, Dict, Any


class CandidateChatState(TypedDict):
    """State passed between every node in the Candidate graph."""

    # Input — set once at entry point
    session_id: str
    query: str
    candidate_id: str
    cv_id: Optional[int]
    jd_id: Optional[int]

    # Session
    conversation_history: List[Dict[str, Any]]
    active_position_ids: List[int]
    position_ref_map: Dict[int, str]

    # Processing
    intent: Literal["jd_search", "jd_analysis", "cv_analysis", "general"]
    intent_confidence: float
    domain: str
    is_apply_intent: bool

    # Retrieved context
    cv_context: List[Dict[str, Any]]
    jd_context: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]

    # Multi-dimensional pre-screening scores
    scored_jobs: Optional[List[Dict[str, Any]]]

    # LLM pipeline
    system_prompt: str
    user_prompt: str
    llm_response: str
    function_calls: Optional[List[Dict[str, Any]]]

    # Output
    final_answer: str
    metadata: Dict[str, Any]
