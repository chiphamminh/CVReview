"""
State schema for the Candidate Chatbot graph.
"""

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
    active_position_ids: Optional[List[int]]
    position_ref_map: Dict[int, str]

    # Intent routing (Sprint 2 — replaces legacy intent/domain/confidence fields)
    pipeline_strategy: str   # JD_SEARCH | JD_ANALYSIS | CV_ANALYSIS | APPLY | STATUS_CHECK | JD_CONVERSE
    query_intent: str        # mirrors pipeline_strategy; kept for metadata output
    query_entities: Dict     # skill_keywords, top_n extracted by router
    expanded_query: Optional[str]   # output of LLM expansion (JD_SEARCH only)
    skill_variants: List[str]       # expanded skill synonyms for keyword search

    # Legacy fields — preserved for backward compat with scoring_node / prompts / session
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