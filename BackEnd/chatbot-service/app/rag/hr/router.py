"""
Hard-rule Intent Router for HR Chatbot.

Priority order (first match wins — no subsequent evaluation):
  Priority 1 — Session state shortcuts:
    pending_emails + confirm phrase → ACTION(confirm_email)
    active_cv_ids + COMPARE pattern → COMPARE
    active_cv_ids + DETAIL pattern  → DETAIL

  Priority 2 — Hard keyword patterns:
    AGGREGATE patterns → AGGREGATE (skip Qdrant)
    ACTION patterns    → ACTION (skip Qdrant)
    FIND_MORE patterns → FIND_MORE (Qdrant + exclude active_cv_ids)

  Priority 3 — Default semantic:
    FILTER patterns    → FILTER (Hybrid + SQL filter)
    fallback           → RANK (Hybrid full)

Output: pipeline_strategy written to HRChatState.
"""

import re
from typing import Dict, Any, List

from app.rag.hr.state import HRChatState

# ---------------------------------------------------------------------------
# Compiled regex patterns — ordered from specific to broad
# ---------------------------------------------------------------------------

_EMAIL_CONFIRM_PATTERN = re.compile(
    r"\b(đồng ý|đồng y|xác nhận|xac nhan|confirm|yes|gửi đi|gui di|ok|okay|send it|chắc chắn)\b",
    re.IGNORECASE,
)

_COMPARE_PATTERN = re.compile(
    r"\b(so sánh|so sanh|compare|đối chiếu|doi chieu|khác nhau|khac nhau|điểm khác|diem khac|versus|vs\.?)\b",
    re.IGNORECASE,
)

_DETAIL_PATTERN = re.compile(
    r"\b(chi tiết|chi tiet|detail|thông tin về|thong tin ve|tell me about|nói về|noi ve|profile của|ho so cua|hồ sơ của)\b",
    re.IGNORECASE,
)

_AGGREGATE_PATTERN = re.compile(
    r"\b(bao nhiêu|bao nhieu|how many|tổng số|tong so|số lượng|so luong|"
    r"thống kê|thong ke|statistics|stat|total|count|cv count|cv stat)\b",
    re.IGNORECASE,
)

_ACTION_PATTERN = re.compile(
    r"\b(gửi email|gui email|send email|phỏng vấn|phong van|invite|"
    r"từ chối|tu choi|reject|offer|thông báo|thong bao|liên hệ|lien he)\b",
    re.IGNORECASE,
)

_FIND_MORE_PATTERN = re.compile(
    r"\b(còn ai|con ai|ai khác|ai khac|thêm|them|khác|khac|next|more|tiếp|tiep|"
    r"tìm thêm|tim them|find more|còn ứng viên|con ung vien|ứng viên khác)\b",
    re.IGNORECASE,
)

_FILTER_PATTERN = re.compile(
    r"\b(có skill|co skill|biết|biet|kỹ năng|ky nang|điểm|diem|score|"
    r"lọc|loc|filter|có kinh nghiệm|co kinh nghiem|năm kinh nghiệm|nam kinh nghiem|"
    r"yêu cầu|yeu cau|require)\b",
    re.IGNORECASE,
)

# Skill keyword extraction — broad catch for common tech/soft skills
_SKILL_KEYWORD_PATTERN = re.compile(
    r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*|"     # TitleCase phrases: Spring Boot, React Native
    r"[A-Z]{2,}(?:\+{1,2}|#)?|"                # Acronyms: AWS, C++, C#
    r"java|python|react|node\.?js|typescript|javascript|"
    r"spring|django|fastapi|docker|kubernetes|kafka|redis|"
    r"sql|mysql|postgresql|mongodb|elasticsearch|"
    r"git|ci\/cd|devops|agile|scrum)\b",
    re.IGNORECASE,
)

# Top-N extraction
_TOP_N_PATTERN = re.compile(r"\b(?:top\s*)?(\d+)\b", re.IGNORECASE)
_HR_DEFAULT_TOP_N = 10

# Score threshold extraction (e.g. "điểm >= 75", "score > 80")
_SCORE_THRESHOLD_PATTERN = re.compile(r"(?:điểm|score|diem)\s*[>>=]+\s*(\d+)", re.IGNORECASE)

# Candidate name extraction from COMPARE/DETAIL (e.g. "so sánh A và B")
_NAME_EXTRACTION_PATTERN = re.compile(
    r"(?:so sánh|compare|chi tiết về|thông tin về|tell me about)\s+(.+?)(?:\s+và\s+|\s+and\s+|$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

def _extract_skill_keywords(query: str) -> List[str]:
    return list({m.lower() for m in _SKILL_KEYWORD_PATTERN.findall(query)})


def _extract_top_n(query: str) -> int:
    matches = _TOP_N_PATTERN.findall(query)
    for m in matches:
        n = int(m)
        if n > 0:
            return min(n, 50)
    return _HR_DEFAULT_TOP_N


def _extract_score_threshold(query: str) -> int | None:
    m = _SCORE_THRESHOLD_PATTERN.search(query)
    return int(m.group(1)) if m else None


def _extract_query_entities(query: str) -> Dict[str, Any]:
    """Extract all structured entities from the raw query string."""
    return {
        "skill_keywords":    _extract_skill_keywords(query),
        "top_n":             _extract_top_n(query),
        "score_threshold":   _extract_score_threshold(query),
    }


# ---------------------------------------------------------------------------
# Router node — pure function, no side effects, no API calls
# ---------------------------------------------------------------------------

def route_hr_intent_node(state: HRChatState) -> HRChatState:
    """
    Classify HR query into a pipeline_strategy and extract structured entities.
    Writes `pipeline_strategy`, `query_intent`, and `query_entities` to state.
    """
    query         = state["query"]
    active_cv_ids = state.get("active_cv_ids") or []
    pending_emails = state.get("pending_emails")

    # Priority 1a — Confirm pending email (session state shortcut)
    if pending_emails and _EMAIL_CONFIRM_PATTERN.search(query):
        strategy = "ACTION"
        intent   = "ACTION"
        print(f"[Router] P1a → ACTION(confirm_email) | pending={len(pending_emails)}")

    # Priority 1b — Compare previously surfaced candidates
    elif active_cv_ids and _COMPARE_PATTERN.search(query):
        strategy = "COMPARE"
        intent   = "COMPARE"
        print(f"[Router] P1b → COMPARE | active_cv_ids={active_cv_ids}")

    # Priority 1c — Detail on previously surfaced candidate
    elif active_cv_ids and _DETAIL_PATTERN.search(query):
        strategy = "DETAIL"
        intent   = "DETAIL"
        print(f"[Router] P1c → DETAIL | active_cv_ids={active_cv_ids}")

    # Priority 2a — Aggregate / statistics queries
    elif _AGGREGATE_PATTERN.search(query):
        strategy = "AGGREGATE"
        intent   = "AGGREGATE"
        print(f"[Router] P2a → AGGREGATE")

    # Priority 2b — Action queries (send email, invite, reject)
    elif _ACTION_PATTERN.search(query):
        strategy = "ACTION"
        intent   = "ACTION"
        print(f"[Router] P2b → ACTION")

    # Priority 2c — Find more candidates (exclude currently active)
    elif _FIND_MORE_PATTERN.search(query):
        strategy = "FIND_MORE"
        intent   = "FIND_MORE"
        print(f"[Router] P2c → FIND_MORE | will exclude {len(active_cv_ids)} id(s)")

    # Priority 3a — Filter by skill / score
    elif _FILTER_PATTERN.search(query):
        strategy = "FILTER"
        intent   = "FILTER"
        print(f"[Router] P3a → FILTER")

    # Priority 3b — Default: semantic ranking
    else:
        strategy = "RANK"
        intent   = "RANK"
        print(f"[Router] P3b → RANK (default)")

    state["pipeline_strategy"] = strategy
    state["query_intent"]      = intent
    state["query_entities"]    = _extract_query_entities(query)

    return state
