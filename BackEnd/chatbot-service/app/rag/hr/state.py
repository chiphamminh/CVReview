from typing import TypedDict, Literal, Optional, List, Dict, Any

class HRChatState(TypedDict):
    """State passed between every node in the HR graph."""

    # Inputs — set once at entry point
    query: str
    hr_id: str
    session_id: str
    position_id: int
    mode: Literal["HR_MODE", "CANDIDATE_MODE"]

    # Tầng 1 hard-rule flags
    is_cv_count_query: bool

    # ISSUE-01: track the cvIds discussed in prior turns so follow-up queries
    # (COMPARE / DETAIL) can pin-fetch them without re-running full reranking.
    active_cv_ids: List[int]
    hr_query_intent: str  # RANK | COMPARE | DETAIL | GENERAL

    # Phase 3: candidate_ids is REMOVED from Qdrant query path.
    # Qdrant filters by applied_position_ids directly (see retriever.py).
    # sql_metadata is still fetched for name/email/score display only.
    sql_metadata: List[Dict[str, Any]]
    cv_id_to_meta: Dict[int, Dict[str, Any]]  # fast lookup: cvId → metadata record

    # Email confirmation flow state
    pending_emails: Optional[List[Dict[str, Any]]]  # email args pending HR confirmation

    # Scoring flow state
    pending_scoring_candidates: Optional[List[Dict[str, Any]]]  # candidates queued for scoring

    # Session memory
    conversation_history: List[Dict[str, Any]]

    # RAG context
    cv_context: List[Dict[str, Any]]
    jd_context: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]

    # LLM pipeline
    system_prompt: str
    user_prompt: str
    llm_response: str
    function_calls: Optional[List[Dict[str, Any]]]

    # Output
    final_answer: str
    metadata: Dict[str, Any]

    # --- NEW FIELDS FOR REFACTOR SPRINT 1 ---
    pipeline_strategy: str          # "SEMANTIC"|"COMPARE"|"FILTER"|"ACTION"|"AGGREGATE"|"FIND_MORE"
    query_intent: str               # "RANK"|"COMPARE"|"DETAIL"|"ACTION"|"AGGREGATE"|"FILTER"|"FIND_MORE"
    query_entities: Dict            # extracted entities: skill_keywords, score_threshold, top_n, candidate_names
    expanded_query: Optional[str]   # output của LLM expansion
    skill_variants: List[str]       # expanded skill list cho keyword search
