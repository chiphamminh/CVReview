from app.rag.hr.state import HRChatState
from app.rag.hr.helpers.history_formatter import _format_history
from app.rag.hr.helpers.context_formatter import _format_cv_context, _format_jd_context, _format_sql_metadata

_ADAPTIVE_INSTRUCTION = """
Before responding, silently determine what HR actually needs:
- COUNT/STATS question → answer with exact numbers from provided data ONLY; never estimate
- SALARY/BENEFITS question → extract only from JD text; never invent numbers
- EMAIL/ACTION request → NEVER send email directly; always confirm recipient first
- CANDIDATE COMPARISON → reference specific CV data for each candidate; compare side by side
- If required data is NOT in context → state explicitly what is missing

Every factual claim MUST trace to a CV chunk, SQL record, or JD section shown above.
NEVER fabricate candidate details, scores, or contact information.
NEVER expose raw UUIDs or database IDs in your response.
"""

# Strategy-specific instructions injected into the system prompt
_STRATEGY_HINTS: dict[str, str] = {
    "COMPARE":   "\n\nINSTRUCTION: Compare the candidates side by side. Use a structured table or bullet comparison. Reference each candidate by name, not by index.",
    "DETAIL":    "\n\nINSTRUCTION: Provide a detailed profile summary for the requested candidate. Cover technical skills, experience, education, and notable projects.",
    "ACTION":    "\n\nINSTRUCTION: Focus on the action requested (email, status update, etc.). Do NOT fetch additional candidates. Use only the metadata provided.",
    "AGGREGATE": "\n\nINSTRUCTION: Answer with exact numbers from the statistics data provided. Do NOT estimate or fabricate counts.",
    "FIND_MORE": "\n\nINSTRUCTION: These are NEW candidates not previously shown. Introduce them freshly without referencing prior candidates.",
    "FILTER":    "\n\nINSTRUCTION: Filter and rank candidates according to the specified criteria. Only list candidates that match.",
    "RANK":      "",
}


def build_hr_prompts_node(state: HRChatState) -> HRChatState:
    """Assemble system + user prompts for the HR LLM call, adapted by pipeline_strategy."""
    strategy     = state.get("pipeline_strategy", "RANK")
    mode_label   = "HR Mode (sourced CVs)" if state["mode"] == "HR_MODE" else "Candidate Mode (inbound applications)"
    history_text = _format_history(state.get("conversation_history", []))
    cv_text      = _format_cv_context(state.get("cv_context", []), state.get("cv_id_to_meta", {}))
    jd_text      = _format_jd_context(state.get("jd_context", []))
    sql_text     = _format_sql_metadata(state.get("sql_metadata", []), state["mode"])

    position_name = "Unknown Position"
    if state.get("jd_context"):
        for chunk in state["jd_context"]:
            name = chunk.get("payload", {}).get("positionName")
            if name:
                position_name = name
                break

    pending_emails = state.get("pending_emails")
    pending_note   = ""
    if pending_emails:
        lines = ["PENDING EMAIL CONFIRMATION — HR must confirm before sending:"]
        for pe in pending_emails:
            lines.append(
                f"  • {pe.get('candidate_name')} ({pe.get('candidate_email')}) "
                f"— Type: {pe.get('email_type')}"
            )
        lines.append("If HR confirms, call send_interview_email with the stored args.")
        pending_note = "\n\n" + "\n".join(lines)

    strategy_hint = _STRATEGY_HINTS.get(strategy, "")

    system_prompt = (
        f"You are a Senior HR Talent Acquisition Specialist assisting with recruitment decisions.\n"
        f"Current mode: {mode_label} | Position: {position_name} (ID: {state['position_id']})\n"
        f"\nGuidelines:\n"
        f"- Respond concisely and professionally — as an experienced recruiter, not a chatbot.\n"
        f"- Base all assessments strictly on the CV data and scores provided — never fabricate candidate details.\n"
        f"- When HR requests to send an email: FIRST confirm the recipient name and email address before invoking send_interview_email.\n"
        f"- When HR requests detailed candidate information, invoke the `get_candidate_details` tool.\n"
        f"- When HR asks about CV count or statistics, invoke the `get_cv_summary` tool.\n"
        f"- When HR wants to filter/rank candidates, invoke the `search_candidates_by_criteria` tool.\n"
        f"- When HR requests to score/evaluate/chấm điểm candidates, invoke the `evaluate_candidates` tool.\n"
        f"- NEVER reveal raw UUIDs or system IDs to HR in your response.\n"
        f"{_ADAPTIVE_INSTRUCTION}{strategy_hint}{pending_note}"
    )

    # Build user prompt — only inject JD block when it has content (saves tokens on COMPARE/DETAIL/ACTION)
    if state.get("jd_context"):
        user_prompt = (
            f"## Conversation History:\n{history_text or '(New session)'}\n\n"
            f"## Job Description Context:\n{jd_text}\n\n"
            f"## CV Data Retrieved from System:\n{cv_text}\n"
        )
    else:
        user_prompt = (
            f"## Conversation History:\n{history_text or '(New session)'}\n\n"
            f"## CV Data Retrieved from System:\n{cv_text}\n"
        )

    if sql_text:
        user_prompt += f"\n## Application Records from Database:\n{sql_text}\n"

    user_prompt += f"\n## HR Question:\n{state['query']}"

    state["system_prompt"] = system_prompt
    state["user_prompt"]   = user_prompt
    return state
