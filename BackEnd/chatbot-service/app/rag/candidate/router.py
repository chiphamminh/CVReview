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

_GENERAL_PATTERN = re.compile(
    r"^(xin chào|chào|hello|hi|hey|cảm ơn|camon|ok|được rồi|.{1,15})$",
    re.IGNORECASE
)

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

_JD_SEARCH_PATTERN = re.compile(
    r"\b(tìm việc|tim viec|tìm công việc|tim cong viec|tìm vị trí|tim vi tri|"
    r"công việc phù hợp|cong viec phu hop|job search|find a job|find jobs|"
    r"có việc nào|co viec nao|việc làm|viec lam|gợi ý việc|goi y viec)\b",
    re.IGNORECASE,
)

_CV_ANALYSIS_PATTERN = re.compile(
    r"\b(kỹ năng của tôi|ky nang cua toi|"
    r"điểm mạnh của tôi|diem manh cua toi|"
    r"kinh nghiệm của tôi|kinh nghiem cua toi|my skills|"
    r"my experience|my background|phân tích cv|phan tich cv|review cv của|"
    r"nhận xét cv|nhan xet cv|đánh giá cv|danh gia cv)\b",
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

# Skill keyword extraction — broad catch for common tech/soft skills
_TECH_SKILL_WHITELIST = frozenset({
    # Languages
    "java", "python", "javascript", "typescript", "kotlin", "swift",
    "golang", "go", "rust", "c++", "c#", "php", "ruby", "scala",
    # Frameworks
    "spring", "spring boot", "spring mvc", "spring framework",
    "django", "fastapi", "flask", "express", "nestjs",
    "react", "angular", "vue", "next.js", "nuxt",
    "node.js", "nodejs",
    # Data / DB
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "oracle", "sqlite", "dynamodb",
    # DevOps / Infra
    "docker", "kubernetes", "k8s", "kafka", "rabbitmq",
    "aws", "gcp", "azure", "terraform", "ansible",
    "ci/cd", "cicd", "jenkins", "github actions", "gitlab ci",
    # Tools & Practices
    "git", "devops", "agile", "scrum", "microservices", "restful", "graphql",
})

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
    "GENERAL":     "general",
}


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

def _extract_skill_keywords(query: str) -> List[str]:
    """
    Whitelist-based skill extraction.
    Uses regex word boundaries to match whitelist directly against the query.
    """
    query_lower = query.lower()
    found: set = set()
    for skill in _TECH_SKILL_WHITELIST:
        if re.search(rf'\b{re.escape(skill)}\b', query_lower):
            found.add(skill)
    return list(found)


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

    # Priority 2b — Status check (SQL only)
    elif _STATUS_CHECK_PATTERN.search(query):
        strategy = "STATUS_CHECK"
        print("[Candidate Router] P2b → STATUS_CHECK")

    # Priority 2c — Explicit Job Search
    elif _JD_SEARCH_PATTERN.search(query):
        strategy = "JD_SEARCH"
        print("[Candidate Router] P2c → JD_SEARCH (explicit keyword)")

    # Priority 2d — CV self-analysis
    elif _CV_ANALYSIS_PATTERN.search(query):
        strategy = "CV_ANALYSIS"
        print("[Candidate Router] P2d → CV_ANALYSIS")

    # Priority 2e — Specific JD conversation (benefits, culture, process)
    elif _JD_CONVERSE_PATTERN.search(query):
        strategy = "JD_CONVERSE"
        print("[Candidate Router] P2e → JD_CONVERSE")

    # Priority 2f — JD requirements analysis
    elif _JD_ANALYSIS_PATTERN.search(query):
        strategy = "JD_ANALYSIS"
        print("[Candidate Router] P2f → JD_ANALYSIS")

    # Priority 2g — General short/greeting query
    elif _GENERAL_PATTERN.match(query.strip()):
        strategy = "GENERAL"
        print("[Candidate Router] P2g → GENERAL (short/greeting query)")

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
