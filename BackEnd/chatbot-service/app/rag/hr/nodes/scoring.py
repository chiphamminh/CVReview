import asyncio
import json as _json
from langchain_core.messages import HumanMessage
from app.rag.hr.state import HRChatState
from app.services.recruitment_api import recruitment_api
from app.services.retriever import get_chunk_text
from app.rag.hr.nodes.reasoning import _build_llm, _extract_llm_text

_STATUS_ICON = {
    "EXCELLENT_MATCH": "🌟",
    "GOOD_MATCH": "✅",
    "POTENTIAL": "🟡",
    "POOR_FIT": "❌",
}


async def _score_one(
    candidate: dict,
    jd_text: str,
    cv_context: list,
    position_id: int,
    session_id: str,
    llm,
) -> dict:
    """Score a single candidate against the JD. Designed to run concurrently via asyncio.gather."""
    cv_id = candidate.get("cvId")
    app_cv_id = candidate.get("appCvId")
    name = candidate.get("candidateName", f"CV-{cv_id}")

    cv_chunks = [c for c in cv_context if c.get("payload", {}).get("cvId") == cv_id]
    cv_text = (
        "\n\n".join(
            f"[{c.get('payload', {}).get('section', '?')}]\n{get_chunk_text(c.get('payload', {}))}"
            for c in cv_chunks
        ).strip()
        or "(CV content not available)"
    )

    scoring_prompt = f"""You are a senior technical recruiter performing a structured CV evaluation.

## Job Description:
{jd_text}

## Candidate CV ({name}):
{cv_text}

Evaluate this candidate and respond with ONLY a JSON object (no markdown) in this exact format:
{{
  "technicalScore": <integer 0-100>,
  "experienceScore": <integer 0-100>,
  "overallStatus": "<EXCELLENT_MATCH|GOOD_MATCH|POTENTIAL|POOR_FIT>",
  "aiAssessment": "<2-3 sentence summary including matched skills, missing skills, and overall recommendation>"
}}
NOTE: Do NOT include learningPath. This is an HR tool for recruiter decisions, not candidate coaching.
- Consolidate skill match/miss observations and overall feedback into the single aiAssessment field."""

    try:
        response = await llm.ainvoke([HumanMessage(content=scoring_prompt)])
        raw = _extract_llm_text(response.content).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        score_data = _json.loads(raw)
        saved = False

        if app_cv_id:
            try:
                await recruitment_api.evaluate_application(
                    app_cv_id=app_cv_id,
                    position_id=position_id,
                    technical_score=score_data.get("technicalScore", 0),
                    experience_score=score_data.get("experienceScore", 0),
                    overall_status=score_data.get("overallStatus", "POOR_FIT"),
                    ai_assessment=score_data.get("aiAssessment", ""),
                    learning_path=None,
                    session_id=session_id,
                )
                saved = True
            except Exception as save_err:
                print(f"[Scoring] Failed to save score for {name}: {save_err}")

        status_icon = _STATUS_ICON.get(score_data.get("overallStatus", ""), "•")
        tech_score = score_data.get("technicalScore", 0)
        exp_score = score_data.get("experienceScore", 0)
        match_percent = (tech_score + exp_score) // 2
        feedback = str(score_data.get("aiAssessment", "")).replace("\n", " ").strip()

        return {
            "row": f"| **{name}** | {match_percent}% {status_icon} | {feedback} |",
            "saved": saved,
            "match_percent": match_percent,
        }

    except Exception as e:
        print(f"[Scoring] Error scoring {name}: {e}")
        return {
            "row": f"| **{name}** | Lỗi | Lỗi khi chấm điểm — {str(e)} |",
            "saved": False,
            "match_percent": -1,
        }


async def hr_scoring_node(state: HRChatState) -> HRChatState:
    """
    Scores all candidates in state['pending_scoring_candidates'] against the JD in parallel.
    """
    candidates = state.get("pending_scoring_candidates") or []
    jd_context = state.get("jd_context", [])

    if not candidates:
        state["llm_response"] = (
            "Không có ứng viên nào trong ngữ cảnh hiện tại để chấm điểm. Vui lòng thực hiện tìm kiếm trước."
        )
        state["pending_scoring_candidates"] = None
        return state

    jd_text = (
        "\n\n".join(get_chunk_text(c.get("payload", {})) for c in jd_context).strip()
        or "(JD not available)"
    )

    llm = _build_llm(temperature=0.1)

    results = await asyncio.gather(
        *[
            _score_one(
                c,
                jd_text,
                state.get("cv_context", []),
                state["position_id"],
                state["session_id"],
                llm,
            )
            for c in candidates
        ]
    )

    results_sorted = sorted(results, key=lambda r: r["match_percent"], reverse=True)
    scoring_rows = [r["row"] for r in results_sorted]
    saved_count = sum(1 for r in results_sorted if r["saved"])

    summary_header = (
        f"\n\n📊 **Kết quả đánh giá {len(scoring_rows)} ứng viên** "
        f"(đã lưu {saved_count}/{len(scoring_rows)} vào hệ thống):\n\n"
        f"| Tên ứng viên | Độ phù hợp | Điểm nổi bật |\n"
        f"|---|---|---|\n"
    )

    new_table = summary_header + "\n".join(scoring_rows)
    state["llm_response"] = new_table.strip()

    state["pending_scoring_candidates"] = None
    state["function_calls"] = [
        {
            "name": "hr_scoring",
            "candidates_scored": len(scoring_rows),
            "saved": saved_count,
        }
    ]
    # Track scored cvIds so FILTER turns don't re-score the same candidates
    scored_ids: set = set(state.get("scored_cv_ids") or [])
    scored_ids.update(c["cvId"] for c in candidates if c.get("cvId") is not None)
    state["scored_cv_ids"] = list(scored_ids)
    return state
