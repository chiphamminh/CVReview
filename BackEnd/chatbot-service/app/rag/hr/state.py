from typing import TypedDict, Literal, Optional, List, Dict, Any

class HRChatState(TypedDict):
    """State passed between every node in the HR graph."""

    # Inputs — set once at entry point
    query: str
    hr_id: str
    session_id: str
    position_id: int
    position_name: str
    mode: Literal["INTERNAL", "EXTERNAL"]

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

    # Routing & expansion
    pipeline_strategy: str          # RANK|FILTER|FIND_MORE|COMPARE|DETAIL|ACTION|AGGREGATE
    query_intent: str
    query_entities: Dict            # skill_keywords, score_threshold, top_n, resolved_cv_id
    expanded_query: Optional[str]
    skill_variants: List[str]

    # F12 — Conversation state machine
    conv_state: str                         # "IDLE" | "AWAITING_CONFIRM"
    pending_action: Optional[str]           # e.g. "SEND_INTERVIEW" when awaiting confirm

    # F13 — Ranked candidate list for ordinal resolution ("người thứ 2" → cvId)
    ranked_cv_list: Optional[List[Dict[str, Any]]]  # [{"rank":1,"cvId":12,"name":"Nguyen A"}]
