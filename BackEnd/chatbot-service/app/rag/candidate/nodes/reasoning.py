"""
Node 4 — LLM reasoning with tool support.

Handles three tool calls:
  finalize_application  — application submission with POOR_FIT guardrail
  evaluate_cv_fit       — marker tool; routes scored_jobs cache back to LLM
  check_application_status — status lookup via recruitment-service API
"""

import json
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from app.rag.candidate.state import CandidateChatState
from app.rag.candidate.candidate_tools import CANDIDATE_TOOLS
from app.config import get_settings

settings = get_settings()

# MatchStatus values that allow application submission
_APPLY_ALLOWED_STATUSES = {"EXCELLENT_MATCH", "GOOD_MATCH", "POTENTIAL"}


def _extract_llm_text(content: Any) -> str:
    """Normalise LLM response content regardless of whether it is a str or a list of blocks."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return str(content)


def _get_temperature(intent: str) -> float:
    temperatures = {
        "cv_analysis": 0.2,
        "jd_analysis": 0.25,
        "jd_search":   0.3,
        "general":     0.4,
    }
    return temperatures.get(intent, 0.3)


async def llm_reasoning_node(state: CandidateChatState) -> CandidateChatState:
    """Single LLM call that generates the full candidate-facing response."""
    intent      = state["intent"]
    temperature = _get_temperature(intent)
    tool_map    = {t.name: t for t in CANDIDATE_TOOLS}

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=temperature,
        max_output_tokens=settings.GEMINI_MAX_TOKENS,
        google_api_key=settings.GEMINI_API_KEY,
    ).bind_tools(CANDIDATE_TOOLS)

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=state["user_prompt"]),
    ]

    response = await llm.ainvoke(messages)
    state["llm_response"]   = _extract_llm_text(response.content)
    state["function_calls"] = None

    if not response.tool_calls:
        return state

    print("[LLM] Tool calls detected:", response.tool_calls)
    state["function_calls"] = []
    messages.append(response)

    finalized_positions = []

    for call in response.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        if tool_name == "finalize_application":
            state, finalized = await _handle_finalize_application(
                state, call, tool_args, tool_map, messages
            )
            if finalized:
                finalized_positions.append(finalized)

        elif tool_name == "evaluate_cv_fit":
            scored_summary = json.dumps(state.get("scored_jobs") or [], ensure_ascii=False)
            state["function_calls"].append({"name": tool_name, "arguments": tool_args})
            messages.append(ToolMessage(content=scored_summary, tool_call_id=call["id"]))

        elif tool_name == "check_application_status":
            enriched_args = {**tool_args, "candidate_id": state["candidate_id"]}
            try:
                tool_res = await tool_map["check_application_status"].ainvoke(enriched_args)
                state["function_calls"].append({"name": tool_name, "arguments": enriched_args, "result": tool_res})
                messages.append(ToolMessage(content=str(tool_res), tool_call_id=call["id"]))
            except Exception as e:
                messages.append(ToolMessage(content=f"Status check error: {str(e)}", tool_call_id=call["id"]))

    if finalized_positions:
        pos_list_str = ", ".join(f"**{name}**" for name in finalized_positions)
        state["llm_response"] = (
            f"I have successfully submitted your application for the following position(s): {pos_list_str}. "
            "Please wait for our HR response!"
        )
        return state

    if state["function_calls"]:
        llm_no_tools = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=temperature,
            max_output_tokens=settings.GEMINI_MAX_TOKENS,
            google_api_key=settings.GEMINI_API_KEY,
        )
        second_response = await llm_no_tools.ainvoke(messages)
        state["llm_response"] = _extract_llm_text(second_response.content)

    return state


async def _handle_finalize_application(
    state: CandidateChatState,
    call: dict,
    tool_args: dict,
    tool_map: dict,
    messages: list,
) -> tuple[CandidateChatState, str | None]:
    """Handle finalize_application tool call with POOR_FIT guardrail and position resolution.

    Returns the updated state and the position name if finalization succeeded, else None.
    """
    ref_map  = state.get("position_ref_map", {})
    pos_id   = tool_args.get("position_id")

    # Auto-resolve position_id from scored_jobs when the LLM omits it.
    if not pos_id and state.get("scored_jobs"):
        allowed = [j for j in state["scored_jobs"] if j.get("overallStatus") in _APPLY_ALLOWED_STATUSES]
        if allowed:
            best   = max(allowed, key=lambda j: j.get("technicalScore", 0) + j.get("experienceScore", 0))
            pos_id = best.get("positionId")
            tool_args = {**tool_args, "position_id": pos_id}
            print(f"[Tầng 1] Resolved position_id={pos_id} from scored_jobs cache.")

    applied_position_name = ref_map.get(pos_id, f"position #{pos_id}")

    # Guardrail: block POOR_FIT applications.
    matched_job = next(
        (j for j in (state.get("scored_jobs") or []) if j.get("positionId") == pos_id),
        None,
    )
    if matched_job and matched_job.get("overallStatus") not in _APPLY_ALLOWED_STATUSES:
        skill_miss    = matched_job.get("skillMiss", [])
        learning_path = matched_job.get("learningPath", "")
        block_msg = (
            f"Không thể nộp đơn vào vị trí **{applied_position_name}** "
            f"(Trạng thái: **POOR_FIT**).\n\n"
            f"**Kỹ năng còn thiếu:** {', '.join(skill_miss) if skill_miss else 'N/A'}\n\n"
            f"**Lộ trình cải thiện:** {learning_path or 'Chưa có gợi ý cụ thể.'}"
        )
        messages.append(ToolMessage(content=block_msg, tool_call_id=call["id"]))
        state["function_calls"].append({"name": "finalize_application", "arguments": tool_args, "result": block_msg})
        return state, None

    try:
        # ISSUE-10: Normalize list→str for skill fields to prevent
        # LLM serializing List[str] as Python repr in API payload.
        normalized_args = {**tool_args}
        for skill_field in ("skill_match", "skill_miss"):
            val = normalized_args.get(skill_field)
            if isinstance(val, list):
                normalized_args[skill_field] = ", ".join(val)
            elif val is None:
                normalized_args[skill_field] = ""

        tool_res = await tool_map["finalize_application"].ainvoke({
            **normalized_args,
            "candidate_id": state["candidate_id"],
            "session_id":   state["session_id"],
        })
        state["function_calls"].append({"name": "finalize_application", "arguments": tool_args, "result": tool_res})
        messages.append(ToolMessage(content=str(tool_res), tool_call_id=call["id"]))

        if "thành công" in str(tool_res).lower() or "success" in str(tool_res).lower():
            return state, applied_position_name
    except Exception as e:
        messages.append(ToolMessage(content=f"Application error: {str(e)}", tool_call_id=call["id"]))

    return state, None
