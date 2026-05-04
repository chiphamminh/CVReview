"""
Hard-rule Intent Router for Candidate Chatbot.

Priority order (first match wins — no subsequent evaluation):

  Priority 1 — Session state shortcuts:
    scored_jobs exist + APPLY pattern → APPLY (skip Qdrant, direct tool call)
    is_apply_intent flag set          → APPLY (Tầng 1 override from session node)

  Priority 2 — Hard keyword patterns:
    STATUS_CHECK patterns → STATUS_CHECK (SQL only, no Qdrant)
    CV_ANALYSIS patterns  → CV_ANALYSIS  (CV chunks only)
    JD_CONVERSE patterns  → JD_CONVERSE  (JD chunks for specific JD conversation)
    JD_ANALYSIS patterns  → JD_ANALYSIS  (JD chunks, no expansion)

  Priority 3 — Default semantic:
    JD_SEARCH (default)   → JD_SEARCH    (CV + JD hybrid retrieval + expansion)

Intent → legacy `intent` field mapping (for backward compat with scoring/prompts):
  JD_SEARCH   → "jd_search"
  JD_ANALYSIS → "jd_analysis"
  CV_ANALYSIS → "cv_analysis"
  APPLY       → "jd_search"   (needs scored_jobs context)
  STATUS_CHECK→ "general"
  JD_CONVERSE → "jd_analysis"

Output: pipeline_strategy + query_intent + query_entities + intent written to CandidateChatState.
"""

import re
from typing import Dict, Any, List

from app.rag.candidate.state import CandidateChatState

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_APPLY_PATTERN = re.compile(
    r"\b(apply|nộp đơn|nop don|ứng tuyển|ung tuyen|finalize|submit.*application"
    r"|i want to apply|giúp tôi apply|help me apply|đăng ký|dang ky)\b",
    re.IGNORECASE,
)

_STATUS_CHECK_PATTERN = re.compile(
    r"\b(đã apply|da apply|đã nộp|da nop|trạng thái|trang thai|status|đơn của tôi"
    r"|don cua toi|đã ứng tuyển|da ung tuyen|kiểm tra đơn|kiem tra don"
    r"|application status|my application|đơn ứng tuyển)\b",
    re.IGNORECASE,
)

_CV_ANALYSIS_PATTERN = re.compile(
    r"\b(cv của tôi|cv cua toi|kỹ năng của tôi|ky nang cua toi|profile của tôi"
    r"|ho so cua toi|hồ sơ của tôi|điểm mạnh của tôi|diem manh cua toi"
    r"|kinh nghiệm của tôi|kinh nghiem cua toi|my cv|my profile|my skills"
    r"|my experience|my background|phân tích cv|phan tich cv|review cv của)\b",
    re.IGNORECASE,
)

_JD_CONVERSE_PATTERN = re.compile(
    r"\b(lương|luong|salary|benefit|phúc lợi|phuc loi|quy trình|quy trinh"
    r"|culture|văn hóa|van hoa|remote|hybrid|work from home|làm từ xa|lam tu xa"
    r"|môi trường làm việc|moi truong lam viec|chế độ|che do|bonuses|thưởng|thuong"
    r"|bảo hiểm|bao hiem|insurance|nghỉ phép|nghi phep|leave|overtime|tăng ca)\b",
    re.IGNORECASE,
)

_JD_ANALYSIS_PATTERN = re.compile(
    r"\b(vị trí này yêu cầu|vi tri nay yeu cau|yêu cầu vị trí|yeu cau vi tri"
    r"|job require|position require|phân tích jd|phan tich jd|jd này|jd nay"
    r"|job description|mô tả công việc|mo ta cong viec|vị trí này cần|vi tri nay can"
    r"|requirements for|what does.*require|vị trí yêu cầu gì)\b",
    re.IGNORECASE,
)

# Skill keyword extraction — same pattern as HR router for consistency
_SKILL_KEYWORD_PATTERN = re.compile(
    r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*|"     # TitleCase phrases: Spring Boot
    r"[A-Z]{2,}(?:\+{1,2}|#)?|"               # Acronyms: AWS, C++, C#
    r"java|python|react|node\.?js|typescript|javascript|"
    r"spring|django|fastapi|docker|kubernetes|kafka|redis|"
    r"sql|mysql|postgresql|mongodb|elasticsearch|"
    r"git|ci\/cd|devops|agile|scrum)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Mapping: pipeline_strategy → legacy intent field value
# scoring_node and prompts.py key off the `intent` field, so we must keep it.
# ---------------------------------------------------------------------------

_STRATEGY_TO_LEGACY_INTENT: Dict[str, str] = {
    "JD_SEARCH":   "jd_search",
    "JD_ANALYSIS": "jd_analysis",
    "CV_ANALYSIS": "cv_analysis",
    "APPLY":       "jd_search",   # Needs scored_jobs → same retrieval path as jd_search
    "STATUS_CHECK":"general",
    "JD_CONVERSE": "jd_analysis",
}


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

def _extract_skill_keywords(query: str) -> List[str]:
    return list({m.lower() for m in _SKILL_KEYWORD_PATTERN.findall(query)})


def _extract_query_entities(query: str) -> Dict[str, Any]:
    """Extract structured entities from the candidate's raw query."""
    return {
        "skill_keywords": _extract_skill_keywords(query),
    }


# ---------------------------------------------------------------------------
# Router node — pure function, no side effects, no API calls
# ---------------------------------------------------------------------------

def route_candidate_intent_node(state: CandidateChatState) -> CandidateChatState:
    """
    Classify candidate query into a pipeline_strategy and extract entities.

    Writes to state:
      - pipeline_strategy: routing decision
      - query_intent:      same value (kept for metadata)
      - query_entities:    extracted skill_keywords
      - intent:            legacy field for backward compat with scoring/prompts nodes
      - intent_confidence: set to 1.0 (hard-rule, always certain)
      - domain:            set to "candidate"
    """
    query         = state["query"]
    is_apply      = state.get("is_apply_intent", False)
    scored_jobs   = state.get("scored_jobs")

    # Priority 1a — Tầng 1 apply flag set by session node OR apply keyword detected
    if is_apply or _APPLY_PATTERN.search(query):
        strategy = "APPLY"
        print(f"[Candidate Router] P1 → APPLY | tầng1_flag={is_apply}")

    # Priority 2a — Status check (SQL only)
    elif _STATUS_CHECK_PATTERN.search(query):
        strategy = "STATUS_CHECK"
        print("[Candidate Router] P2a → STATUS_CHECK")

    # Priority 2b — CV self-analysis
    elif _CV_ANALYSIS_PATTERN.search(query):
        strategy = "CV_ANALYSIS"
        print("[Candidate Router] P2b → CV_ANALYSIS")

    # Priority 2c — Specific JD conversation (benefits, culture, process)
    elif _JD_CONVERSE_PATTERN.search(query):
        strategy = "JD_CONVERSE"
        print("[Candidate Router] P2c → JD_CONVERSE")

    # Priority 2d — JD requirements analysis
    elif _JD_ANALYSIS_PATTERN.search(query):
        strategy = "JD_ANALYSIS"
        print("[Candidate Router] P2d → JD_ANALYSIS")

    # Priority 3 — Default: full JD search + scoring pipeline
    else:
        strategy = "JD_SEARCH"
        print("[Candidate Router] P3 → JD_SEARCH (default)")

    legacy_intent = _STRATEGY_TO_LEGACY_INTENT[strategy]

    state["pipeline_strategy"]  = strategy
    state["query_intent"]       = strategy
    state["query_entities"]     = _extract_query_entities(query)
    # Backward compat: scoring_node and build_prompts_node key off `intent`
    state["intent"]             = legacy_intent
    state["intent_confidence"]  = 1.0
    state["domain"]             = "candidate"

    return state
