from app.rag.hr.state import HRChatState
from app.rag.hr.helpers.history_formatter import _format_history
from app.rag.hr.helpers.context_formatter import (
    _format_cv_context,
    _format_jd_context,
    _format_sql_metadata,
)

_ADAPTIVE_INSTRUCTION = """
Before responding, silently determine what HR actually needs:
- COUNT/STATS question → answer with exact numbers from provided data ONLY; never estimate
- SALARY/BENEFITS question → extract only from JD text; never invent numbers
- EMAIL/ACTION request → call send_interview_email tool immediately for each named recipient; system will confirm automatically
- CANDIDATE COMPARISON → reference specific CV data for each candidate; compare side by side
- If required data is NOT in context → state explicitly what is missing

Every factual claim MUST trace to a CV chunk, SQL record, or JD section shown above.
NEVER fabricate candidate details, scores, or contact information.
NEVER expose raw UUIDs or database IDs in your response.
"""

# Strategy-specific instructions injected into the system prompt
_STRATEGY_HINTS: dict[str, str] = {
    "COMPARE": "\n\nINSTRUCTION: Compare the candidates side by side. Use a structured table or bullet comparison. Reference each candidate by name, not by index.",
    "DETAIL": "\n\nINSTRUCTION: Provide a detailed profile summary for the requested candidate. Cover technical skills, experience, education, and notable projects.",
    "ACTION": "\n\nINSTRUCTION: Focus on the action requested (email, position_name, status update, interview_date ISO format YYYY-MM-DD HH:MM, etc.). Do NOT fetch additional candidates. Use only the metadata provided.",
    "AGGREGATE": "\n\nINSTRUCTION: Answer with exact numbers from the statistics data provided. Do NOT estimate or fabricate counts.",
    "FIND_MORE": (
        "\n\nINSTRUCTION: These are NEW candidates not previously shown. "
        "Do NOT write a detailed analysis — a structured scoring table will be generated automatically. "
        "Respond with ONE brief sentence acknowledging new candidates found, then stop."
    ),
    "FILTER": "\n\nINSTRUCTION: Filter and rank candidates according to the specified criteria. Only list candidates that match.",
    "RANK": "\n\nINSTRUCTION: Do NOT write a detailed analysis or list candidates individually — a structured scoring table will be generated automatically. Respond with ONE brief sentence acknowledging the request, then stop.",
}


def _build_candidate_action_instruction(strategy: str) -> str:
    """
    Return the strategy-specific instruction line for candidate search/rank/filter.

    RANK: LLM reads CV chunks from context and answers directly — no DB tool call needed.
          Scoring is auto-triggered by the graph after reasoning completes.
    FILTER: HR is filtering by score/skill/name criteria stored in MySQL, not Qdrant,
            so search_candidates_by_criteria must query the DB.
    Others: No ranking-related instruction injected.
    """
    if strategy in ("RANK", "FIND_MORE"):
        return "- Do NOT call `search_candidates_by_criteria` — AI scoring will run automatically after your response.\n"
    if strategy == "FILTER":
        return (
            "- To filter candidates by score, skill, or name criteria: invoke `search_candidates_by_criteria` "
            "with the appropriate parameters.\n"
        )
    return ""


def build_hr_prompts_node(state: HRChatState) -> HRChatState:
    """Assemble system + user prompts for the HR LLM call, adapted by pipeline_strategy."""
    strategy = state.get("pipeline_strategy", "RANK")
    mode_label = (
        "Internal Mode (HR-sourced CVs)"
        if state["mode"] == "INTERNAL"
        else "External Mode (inbound applications)"
    )
    history_text = _format_history(state.get("conversation_history", []))
    cv_text = (
        _format_cv_context(state.get("cv_context", []), state.get("cv_id_to_meta", {}))
        if strategy not in ("ACTION", "AGGREGATE")
        else ""
    )
    jd_text = (
        _format_jd_context(state.get("jd_context", []))
        if strategy not in ("ACTION", "AGGREGATE")
        else ""
    )
    sql_text = _format_sql_metadata(state.get("sql_metadata", []), state["mode"])

    position_name = state.get("position_name", "Unknown Position")

    if state.get("jd_context"):
        for chunk in state["jd_context"]:
            name = chunk.get("payload", {}).get("positionName")
            if name:
                position_name = name
                break

    if position_name == "Unknown Position" and state.get("sql_metadata"):
        first = state["sql_metadata"][0]
        position_name = first.get("positionName") or first.get(
            "position", "Unknown Position"
        )

    pending_emails = state.get("pending_emails")
    pending_note = ""
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

    _candidate_action_instruction = _build_candidate_action_instruction(strategy)

    system_prompt = (
        f"You are a Senior HR Talent Acquisition Specialist assisting with recruitment decisions.\n"
        f"Current mode: {mode_label} | Position: {position_name} (ID: {state['position_id']})\n"
        f"\nGuidelines:\n"
        f"- Respond concisely and professionally — as an experienced recruiter, not a chatbot.\n"
        f"- Base all assessments strictly on the CV data and scores provided — never fabricate candidate details.\n"
        f"- When HR requests to send an email: invoke `send_interview_email` IMMEDIATELY for each recipient. The system handles confirmation automatically — do NOT generate manual confirmation text.\n"
        f"- When HR requests detailed candidate information, invoke the `get_candidate_details` tool.\n"
        f"- When HR asks about CV count or statistics, invoke the `get_cv_summary` tool.\n"
        f"{_candidate_action_instruction}"
        f"- NEVER reveal raw UUIDs or system IDs to HR in your response.\n"
        f"{_ADAPTIVE_INSTRUCTION}{strategy_hint}{pending_note}"
    )

    # Build user prompt — omit CV/JD blocks entirely for ACTION and AGGREGATE to save tokens
    user_prompt = f"## Conversation History:\n{history_text or '(New session)'}\n\n"

    if strategy not in ("ACTION", "AGGREGATE"):
        if state.get("jd_context"):
            user_prompt += f"## Job Description Context:\n{jd_text}\n\n"
        user_prompt += f"## CV Data Retrieved from System:\n{cv_text}\n"

    if sql_text:
        user_prompt += f"\n## Application Records from Database:\n{sql_text}\n"

    # F13: inject ranked candidate reference so LLM can resolve "người thứ 2"
    ranked_cv_list = state.get("ranked_cv_list") or []
    if ranked_cv_list:
        ordinal_lines = "\n".join(
            f"  - Rank {r['rank']}: {r['name']} (cvId={r['cvId']})"
            for r in ranked_cv_list
        )
        user_prompt += f"\n## Ranked Candidates (use for ordinal references like 'người thứ 2'):\n{ordinal_lines}\n"

    user_prompt += f"\n## HR Question:\n{state['query']}"

    state["system_prompt"] = system_prompt
    state["user_prompt"] = user_prompt
    return state
