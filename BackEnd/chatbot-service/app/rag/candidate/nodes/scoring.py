"""
Node 2.5 — Multi-dimensional CV-JD scoring (parallel, one LLM call per JD).

Runs only for `jd_search` intent on Turn 1 (no scoring cache).

Model tier:
  APPLY strategy  → Pro model (guardrail accuracy is critical)
  All others      → Flash model (browsing; cost/speed tradeoff)

MatchStatus thresholds:
  EXCELLENT_MATCH  technicalScore >= 85 AND experienceScore >= 80
  GOOD_MATCH       technicalScore >= 70 AND experienceScore >= 65
  POTENTIAL        technicalScore >= 55 OR  experienceScore >= 55 (but not GOOD_MATCH)
  POOR_FIT         All other cases
"""

import asyncio
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.rag.candidate.state import CandidateChatState
from app.rag.prompts import build_cv_context
from app.config import get_settings

settings = get_settings()


def _get_scoring_model(pipeline_strategy: str) -> str:
    """APPLY guardrail needs Pro accuracy; all browsing intents use Flash."""
    if pipeline_strategy == "APPLY":
        return settings.SCORING_GEMINI_MODEL
    return settings.GEMINI_MODEL


_SCORING_PROMPT_SINGLE = """You are an experienced HR scoring system. Score this CV against the provided JD.

SCORING SYSTEM:

1. Technical Score (0–100):
   - 85-100: Covers all required skills with depth and relevant project evidence.
   - 65-84:  Covers most required skills; minor gaps in secondary tools.
   - 45-64:  Covers core skills but missing some important stack items.
   - 0-44:   Significant gaps in required technical skills.

2. Experience Score (0–100):
   - 85-100: Led architectural decisions, measurable business impact, system design ownership.
   - 65-84:  Mid-level, implemented features with rationale, understands trade-offs.
   - 40-64:  Junior/basic CRUD projects, limited scale or architectural thinking.
   - 0-39:   No relevant professional experience.

3. Overall Status (derive from scores above):
   - EXCELLENT_MATCH: technicalScore >= 85 AND experienceScore >= 80
   - GOOD_MATCH:      technicalScore >= 70 AND experienceScore >= 65
   - POTENTIAL:       technicalScore >= 55 OR experienceScore >= 55 (but not GOOD_MATCH)
   - POOR_FIT:        All other cases

RULES:
- Evidence-based: score what is in the CV, but allow reasonable inference from project context.
- NO HALLUCINATION: Do not invent skills not mentioned anywhere in the CV.
- HIERARCHICAL SKILL INFERENCE: If candidate has advanced skills that imply basics (e.g., Spring Boot implies Java), do NOT deduct for missing basic keywords. If overqualified for a lower-level role, score highly and note "Overqualified" in aiAssessment.

CV:
{cv_profile}

JD to score:
{jd_block}

Return EXACTLY this JSON object (no markdown, no preamble):
{{
  "positionId": <integer>,
  "technicalScore": <0-100>,
  "experienceScore": <0-100>,
  "overallStatus": "<EXCELLENT_MATCH|GOOD_MATCH|POTENTIAL|POOR_FIT>",
  "aiAssessment": "<1-2 concise HR-tone sentences summarising key matched skills and critical gaps>"
}}"""


def _build_jd_block(jd: dict) -> str:
    payload = jd.get("payload", {})
    jd_id = payload.get("positionId", "unknown")
    jd_title = " ".join(
        filter(
            None,
            [
                payload.get("positionTitle"),
                payload.get("seniority"),
            ],
        )
    ) or payload.get("positionTitle", "Unknown Position")
    jd_text = payload.get("jdText", "")
    return f"\n[JD ID: {jd_id} | Title: {jd_title}]\n{jd_text}\n"


async def _score_single_jd(
    cv_profile: str,
    jd: dict,
    llm,
) -> dict | None:
    """Score one JD against the candidate CV. Returns None on parse failure so
    asyncio.gather can safely skip it (graceful degradation).
    Accepts a shared LLM instance — one instance handles concurrent ainvoke calls
    via its internal async HTTP connection pool (mirrors HR scoring pattern)."""
    payload = jd.get("payload", {})
    jd_id = payload.get("positionId", "unknown")

    prompt = _SCORING_PROMPT_SINGLE.format(
        cv_profile=cv_profile,
        jd_block=_build_jd_block(jd),
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[Scoring Error] JD {jd_id}: {e}")
        return None


async def scoring_node(state: CandidateChatState) -> CandidateChatState:
    """Score CV against each retrieved JD in parallel.

    Skips if intent is not `jd_search`, context is missing, or a cache hit exists.
    Failed individual JD calls are skipped (graceful degradation).
    """
    if (
        state["intent"] != "jd_search"
        or not state["jd_context"]
        or not state["cv_context"]
    ):
        state["scored_jobs"] = state.get("scored_jobs")
        return state

    if state.get("scored_jobs"):
        print("[Scoring] Cache hit — bypassing LLM scoring call.")
        return state

    scoring_model = _get_scoring_model(state.get("pipeline_strategy", "JD_SEARCH"))
    jd_count = len(state["jd_context"])
    print(
        f"[Scoring] Parallel scoring {jd_count} positions with model: {scoring_model}..."
    )

    llm = ChatGoogleGenerativeAI(
        model=scoring_model,
        temperature=0.0,
        max_output_tokens=settings.GEMINI_MAX_TOKENS,
        google_api_key=settings.GEMINI_API_KEY,
        thinking_budget=0,
    )

    cv_profile = build_cv_context(state["cv_context"])

    results = await asyncio.gather(
        *[
            _score_single_jd(cv_profile, jd, llm)
            for jd in state["jd_context"]
        ],
    )

    scored_jobs = [r for r in results if r is not None]
    state["scored_jobs"] = scored_jobs if scored_jobs else None
    print(f"[Scoring] Scored {len(scored_jobs)}/{jd_count} positions (parallel).")

    return state
