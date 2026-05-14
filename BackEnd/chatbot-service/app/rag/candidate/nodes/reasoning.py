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
from app.services import position_score_cache
from app.config import get_settings

settings = get_settings()


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


def _get_max_output_tokens(intent: str) -> int:
    """JD_SEARCH generates multi-position analysis with learning paths — needs more headroom."""
    if intent == "jd_search":
        return 4096
    return settings.GEMINI_MAX_TOKENS


async def llm_reasoning_node(state: CandidateChatState) -> CandidateChatState:
    """Single LLM call that generates the full candidate-facing response."""
    intent          = state["intent"]
    temperature     = _get_temperature(intent)
    max_out_tokens  = _get_max_output_tokens(intent)
    tool_map        = {t.name: t for t in CANDIDATE_TOOLS}

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=temperature,
        max_output_tokens=max_out_tokens,
        google_api_key=settings.GEMINI_API_KEY,
        model_kwargs={"thinking_config": {"thinking_budget": 0}},
    ).bind_tools(CANDIDATE_TOOLS)

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=state["user_prompt"]),
    ]

    # ainvoke (not astream) — astream inside astream_events causes LangGraph
    # callback interference that truncates the generator after a few chunks.
    response = await llm.ainvoke(messages)

    state["llm_response"]   = _extract_llm_text(response.content) if response else ""
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
            if finalized == "BLOCKED":
                return state  # llm_response already set, skip second LLM
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
            f"Bạn đã nộp đơn ứng tuyển thành công cho vị trí: {pos_list_str}. "
            "Hãy chờ phản hồi từ phía HR nhé!"
        )
        return state

    if state["function_calls"]:
        llm_no_tools = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=temperature,
            max_output_tokens=max_out_tokens,
            google_api_key=settings.GEMINI_API_KEY,
            model_kwargs={"thinking_config": {"thinking_budget": 0}},
        )
        second_response = await llm_no_tools.ainvoke(messages)
        state["llm_response"] = _extract_llm_text(second_response.content) if second_response else ""

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
        allowed = [
            j for j in state["scored_jobs"]
            if (j.get("technicalScore", 0) + j.get("experienceScore", 0)) / 2
               >= position_score_cache.get(j.get("positionId"), 70.0)
        ]
        if allowed:
            best   = max(allowed, key=lambda j: j.get("technicalScore", 0) + j.get("experienceScore", 0))
            pos_id = best.get("positionId")
            tool_args = {**tool_args, "position_id": pos_id}
            print(f"[Tầng 1] Resolved position_id={pos_id} from scored_jobs cache.")

    applied_position_name = ref_map.get(pos_id, f"position #{pos_id}")

    # Guardrail: block applications below minimumFitScore threshold.
    # Always inject authoritative scores from state — never rely on LLM-provided values.
    matched_job = next(
        (j for j in (state.get("scored_jobs") or []) if j.get("positionId") == pos_id),
        None,
    )
    if matched_job:
        tool_args = {
            **tool_args,
            "technical_score":  matched_job.get("technicalScore", 0),
            "experience_score": matched_job.get("experienceScore", 0),
            "overall_status":   matched_job.get("overallStatus", tool_args.get("overall_status", "")),
            "ai_assessment":    matched_job.get("aiAssessment", tool_args.get("ai_assessment", "")),
        }
        avg_score = (matched_job.get("technicalScore", 0) + matched_job.get("experienceScore", 0)) / 2
        min_score = position_score_cache.get(pos_id, 70.0)
        if avg_score < min_score:
            state["llm_response"] = (
                f"Rất tiếc, bạn chưa đủ điều kiện để ứng tuyển vào vị trí **{applied_position_name}**.\n\n"
                f"Điểm phù hợp của bạn là **{avg_score:.0f}/100**, thấp hơn ngưỡng tối thiểu yêu cầu "
                f"(**{min_score:.0f}/100**) của vị trí này.\n\n"
                f"Hãy tiếp tục phát triển kỹ năng và thử lại sau khi đã cải thiện các điểm còn thiếu."
            )
            messages.append(ToolMessage(content="blocked: score below minimum", tool_call_id=call["id"]))
            state["function_calls"].append({"name": "finalize_application", "arguments": tool_args, "result": "blocked"})
            return state, "BLOCKED"

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
