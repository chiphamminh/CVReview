"""
Node 2.5 — Multi-dimensional CV-JD scoring.

Runs only for `jd_search` intent on Turn 1 (no scoring cache).
Uses a dedicated Pro model for higher accuracy.

MatchStatus thresholds:
  EXCELLENT_MATCH  technicalScore >= 85 AND experienceScore >= 80
  GOOD_MATCH       technicalScore >= 70 AND experienceScore >= 65
  POTENTIAL        technicalScore >= 55 OR  experienceScore >= 55 (but not GOOD_MATCH)
  POOR_FIT         All other cases
"""

import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.rag.candidate.state import CandidateChatState
from app.rag.prompts import build_cv_context
from app.config import get_settings

settings = get_settings()

_SCORING_PROMPT_TEMPLATE = """You are a strict HR scoring system. Score each CV against the provided JDs.

SCORING SYSTEM:

1. Technical Score (0–100):
   - Start: 100 pts.
   - Deduct 8 pts for each missing REQUIRED skill explicitly stated in JD.
   - Deduct 12 pts if candidate has < 70% of required tech stack overall.

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
- STRICT: If a skill is not explicitly stated in CV, assume candidate does NOT have it.
- NO HALLUCINATION: Do not infer skills from project context alone.
- HIERARCHICAL SKILL INFERENCE: If candidate is overqualified for a lower-level role, do NOT penalize. Score highly. Set feedback to "Overqualified".
- learningPath: Provide ONLY for POTENTIAL or POOR_FIT. For POOR_FIT, this is MANDATORY. For EXCELLENT_MATCH or GOOD_MATCH, set to null.

CV:
{cv_profile}

JDs to score:
{jds_block}

Return EXACTLY this JSON array (no markdown, no preamble):
[
  {{
    "positionId": <integer>,
    "technicalScore": <0-100>,
    "experienceScore": <0-100>,
    "overallStatus": "<EXCELLENT_MATCH|GOOD_MATCH|POTENTIAL|POOR_FIT>",
    "feedback": "<1 concise HR-tone sentence>",
    "skillMatch": ["<skill>", "<skill>"],
    "skillMiss": ["<skill>", "<skill>"],
    "learningPath": "<90-day roadmap string or null>"
  }}
]"""


def _build_jds_block(jd_context: list) -> str:
    """Format JD context list into a single text block for the scoring prompt."""
    block = ""
    for jd in jd_context:
        payload  = jd.get("payload", {})
        jd_id    = payload.get("positionId", "unknown")
        jd_title = " ".join(filter(None, [
            payload.get("positionName"),
            payload.get("language"),
            payload.get("level"),
        ])) or payload.get("positionName", "Unknown Position")
        jd_text  = payload.get("jdText", "")
        block += f"\n[JD ID: {jd_id} | Title: {jd_title}]\n{jd_text}\n"
    return block


async def scoring_node(state: CandidateChatState) -> CandidateChatState:
    """Score CV against retrieved JDs using a dedicated Pro model.

    Skips if intent is not `jd_search`, context is missing, or a cache hit exists.
    """
    if state["intent"] != "jd_search" or not state["jd_context"] or not state["cv_context"]:
        state["scored_jobs"] = state.get("scored_jobs")
        return state

    if state.get("scored_jobs"):
        print("[Scoring] Cache hit — bypassing LLM scoring call.")
        return state

    print(f"[Scoring] Running multi-dimensional scoring with model: {settings.SCORING_GEMINI_MODEL}...")

    prompt = _SCORING_PROMPT_TEMPLATE.format(
        cv_profile=build_cv_context(state["cv_context"]),
        jds_block=_build_jds_block(state["jd_context"]),
    )

    llm = ChatGoogleGenerativeAI(
        model=settings.SCORING_GEMINI_MODEL,
        temperature=0.0,
        max_output_tokens=1500,
        google_api_key=settings.GEMINI_API_KEY,
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        state["scored_jobs"] = json.loads(raw)
        print(f"[Scoring] Scored {len(state['scored_jobs'])} positions (multi-dimensional).")
    except Exception as e:
        print(f"[Scoring Error] JSON parse failed: {e} — proceeding without scores.")
        state["scored_jobs"] = None

    return state
