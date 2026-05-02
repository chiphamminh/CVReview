import json as _json
from langchain_core.messages import HumanMessage
from app.rag.hr.state import HRChatState
from app.services.recruitment_api import recruitment_api
from app.services.retriever import get_chunk_text
from app.rag.hr.nodes.reasoning import _build_llm, _extract_llm_text

async def hr_scoring_node(state: HRChatState) -> HRChatState:
    """
    Scores each candidate in state['pending_scoring_candidates'] against the JD.
    """
    candidates = state.get("pending_scoring_candidates") or []
    jd_context  = state.get("jd_context", [])

    if not candidates:
        state["llm_response"] = "Không có ứng viên nào trong ngữ cảnh hiện tại để chấm điểm. Vui lòng thực hiện tìm kiếm trước."
        state["pending_scoring_candidates"] = None
        return state

    jd_text = "\n\n".join(
        get_chunk_text(c.get("payload", {})) for c in jd_context
    ).strip() or "(JD not available)"

    llm = _build_llm(temperature=0.1)
    scoring_results = []
    saved_count = 0

    for candidate in candidates:
        cv_id   = candidate.get("cvId")
        app_cv_id = candidate.get("appCvId")
        name    = candidate.get("candidateName", f"CV-{cv_id}")

        cv_chunks = [
            c for c in state.get("cv_context", [])
            if c.get("payload", {}).get("cvId") == cv_id
        ]
        cv_text = "\n\n".join(
            f"[{c.get('payload',{}).get('section','?')}]\n{get_chunk_text(c.get('payload', {}))}"
            for c in cv_chunks
        ).strip() or "(CV content not available)"

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
  "feedback": "<2-3 sentence summary>",
  "skillMatch": "<comma-separated matched skills>",
  "skillMiss": "<comma-separated missing skills or 'None'>",
  "learningPath": "<recommended path if POTENTIAL/POOR_FIT, else null>"
}}"""

        print("HR scoring_prompt: ", scoring_prompt)

        try:
            response = await llm.ainvoke([HumanMessage(content=scoring_prompt)])
            raw = _extract_llm_text(response.content).strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            score_data = _json.loads(raw)

            if app_cv_id:
                try:
                    await recruitment_api.evaluate_application(
                        app_cv_id=app_cv_id,
                        position_id=state["position_id"],
                        technical_score=score_data.get("technicalScore", 0),
                        experience_score=score_data.get("experienceScore", 0),
                        overall_status=score_data.get("overallStatus", "POOR_FIT"),
                        feedback=score_data.get("feedback", ""),
                        skill_match=score_data.get("skillMatch", ""),
                        skill_miss=score_data.get("skillMiss", ""),
                        learning_path=score_data.get("learningPath"),
                        session_id=state["session_id"],
                    )
                    saved_count += 1
                except Exception as save_err:
                    print(f"[Scoring] Failed to save score for {name}: {save_err}")

            status_icon = {
                "EXCELLENT_MATCH": "🌟",
                "GOOD_MATCH": "✅",
                "POTENTIAL": "🟡",
                "POOR_FIT": "❌",
            }.get(score_data.get("overallStatus", ""), "•")

            scoring_results.append(
                f"**{name}** {status_icon}\n"
                f"• Kỹ thuật: {score_data.get('technicalScore')}/100 | "
                f"Kinh nghiệm: {score_data.get('experienceScore')}/100\n"
                f"• Nhận xét: {score_data.get('feedback')}\n"
                f"• Phù hợp: {score_data.get('skillMatch')}\n"
                f"• Còn thiếu: {score_data.get('skillMiss')}"
            )
        except Exception as e:
            scoring_results.append(f"**{name}**: Lỗi khi chấm điểm — {str(e)}")
            print(f"[Scoring] Error scoring {name}: {e}")

    summary_header = (
        f"📊 **Kết quả chấm điểm {len(scoring_results)} ứng viên** "
        f"(đã lưu {saved_count}/{len(scoring_results)} vào hệ thống):\n"
    )
    state["llm_response"] = summary_header + "\n\n".join(scoring_results)
    state["pending_scoring_candidates"] = None
    state["function_calls"] = [{"name": "hr_scoring", "candidates_scored": len(scoring_results), "saved": saved_count}]
    return state
