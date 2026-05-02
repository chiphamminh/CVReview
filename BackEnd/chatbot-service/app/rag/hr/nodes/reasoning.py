from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.hr.state import HRChatState
from app.rag.hr_tools import HR_TOOLS
from app.config import get_settings
from app.rag.hr.helpers.candidate_resolver import _resolve_candidates_by_name

settings = get_settings()


def _extract_llm_text(content) -> str:
    """Normalise LLM response content regardless of str vs list-of-blocks format."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return str(content)


def _build_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=temperature,
        max_output_tokens=settings.GEMINI_MAX_TOKENS,
        google_api_key=settings.GEMINI_API_KEY
    )


async def _execute_pending_email_confirmation(
    state: HRChatState,
    pending_emails: list,
) -> HRChatState:
    """
    Execute send_interview_email for each pending email in a single confirmed batch.
    Router has already verified the query is a confirmation phrase before calling this.
    """
    tool_map   = {t.name: t for t in HR_TOOLS}
    email_tool = tool_map.get("send_interview_email")
    results    = []
    func_calls = []

    if email_tool:
        for pe in pending_emails:
            try:
                res = await email_tool.ainvoke(pe)
                results.append(str(res))
                func_calls.append({"name": "send_interview_email", "arguments": pe, "result": res})
            except Exception as e:
                err_msg = f"Lỗi khi gửi email tới {pe.get('candidate_name')}: {str(e)}"
                results.append(err_msg)
                func_calls.append({"name": "send_interview_email", "arguments": pe, "result": err_msg})

    state["llm_response"]   = "\n".join(results)
    state["function_calls"] = func_calls
    state["pending_emails"] = None
    print(f"[Email Confirm] Sent {len(func_calls)} email(s).")
    return state


async def llm_hr_reasoning_node(state: HRChatState) -> HRChatState:
    """
    Execute the LLM reasoning step with HR_TOOLS bound.

    The router (route_hr_intent_node) has already classified intent and written
    pipeline_strategy to state, so this node no longer needs to detect intent.

    ACTION strategy with pending_emails → immediately execute email confirms.
    All other strategies → standard LLM call + tool loop.
    """
    strategy       = state.get("pipeline_strategy", "RANK")
    pending_emails = state.get("pending_emails")

    # --- Fast path: confirmed email send (router detected ACTION + confirm phrase) ---
    if strategy == "ACTION" and pending_emails:
        print(f"[Reasoning] ACTION + pending_emails → executing {len(pending_emails)} confirmation(s)")
        return await _execute_pending_email_confirmation(state, pending_emails)

    # --- Standard LLM call ---
    llm = _build_llm(temperature=0.3).bind_tools(HR_TOOLS)
    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=state["user_prompt"])
    ]

    response = await llm.ainvoke(messages)
    state["function_calls"] = None

    # No tool calls → plain text answer
    if not response.tool_calls:
        state["llm_response"] = _extract_llm_text(response.content)
        # Clear stale pending_emails if HR didn't confirm (says something else under ACTION)
        if strategy == "ACTION" and pending_emails:
            print("[Reasoning] ACTION but no confirmation phrase detected — clearing pending_emails.")
            state["pending_emails"] = None
        return state

    # --- Tool execution loop ---
    executed_calls     = []
    new_pending_emails = []
    tool_map           = {t.name: t for t in HR_TOOLS}
    messages.append(response)

    for call in response.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"].copy()

        # Auto-inject context params that LLM tends to omit
        if tool_name in ("send_interview_email", "get_candidate_details", "get_cv_summary", "search_candidates_by_criteria"):
            if not tool_args.get("position_id"):
                tool_args["position_id"] = state["position_id"]
                print(f"[Tool Inject] position_id → {tool_name}")
            if tool_name in ("search_candidates_by_criteria", "get_cv_summary"):
                tool_args["mode"] = state["mode"]

        # Intercept send_interview_email → mandatory confirmation flow
        if tool_name == "send_interview_email":
            candidate_name = tool_args.get("candidate_name", "")
            matches = _resolve_candidates_by_name(
                candidate_name, state.get("sql_metadata", []), state["mode"]
            )

            if len(matches) == 0:
                state["llm_response"] = (
                    f"Tôi không tìm thấy ứng viên nào tên '{candidate_name}' "
                    f"trong danh sách vị trí này. Vui lòng kiểm tra lại tên."
                )
                return state

            if len(matches) > 1:
                lines = [f"Tôi tìm thấy {len(matches)} ứng viên tên '{candidate_name}':"]
                for idx, m in enumerate(matches, 1):
                    lines.append(f"{idx}. {m.get('candidateName')} — {m.get('candidateEmail')}")
                lines.append("\nBạn muốn gửi email cho ai? Vui lòng chỉ rõ địa chỉ email.")
                state["llm_response"] = "\n".join(lines)
                return state

            matched = matches[0]
            tool_args["app_cv_id"]       = int(matched.get("appCvId") or 0)
            tool_args["candidate_id"]    = str(matched.get("candidateId") or matched.get("appCvId", ""))
            tool_args["candidate_email"] = tool_args.get("candidate_email") or matched.get("candidateEmail", "")
            tool_args["candidate_name"]  = matched.get("candidateName", candidate_name)

            new_pending_emails.append(tool_args)
            continue  # Do NOT execute yet — await HR confirmation next turn

        # Intercept evaluate_candidates → delegate to hr_scoring_node
        if tool_name == "evaluate_candidates":
            cv_id_to_meta  = state.get("cv_id_to_meta", {})
            cv_ids_in_ctx: set = {
                chunk.get("payload", {}).get("cvId")
                for chunk in state.get("cv_context", [])
                if chunk.get("payload", {}).get("cvId") is not None
            }

            state["pending_scoring_candidates"] = [
                {
                    "cvId":          cv_id,
                    "appCvId":       cv_id_to_meta.get(cv_id, {}).get("appCvId"),
                    "candidateName": cv_id_to_meta.get(cv_id, {}).get("candidateName", f"CV-{cv_id}"),
                }
                for cv_id in cv_ids_in_ctx
            ]
            print(f"[Scoring] Intercepted evaluate_candidates → {len(state['pending_scoring_candidates'])} candidate(s) queued")
            continue

        # All other tools — execute normally
        tool_instance = tool_map.get(tool_name)
        if tool_instance is None:
            tool_result = f"Tool '{tool_name}' does not exist."
        else:
            try:
                tool_result = await tool_instance.ainvoke(tool_args)
            except Exception as e:
                tool_result = f"Error executing tool {tool_name}: {str(e)}"

        executed_calls.append({"name": tool_name, "arguments": tool_args, "result": tool_result})
        messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))

    # Build confirmation prompt if LLM wants to send email(s)
    if new_pending_emails:
        state["pending_emails"] = new_pending_emails
        lines = ["Bạn có chắc muốn gửi email tới các ứng viên sau không?"]
        for pe in new_pending_emails:
            lines.append(f"- **{pe.get('candidate_name')}** ({pe.get('candidate_email')})")
        lines.append("\nGõ 'Đồng ý' để xác nhận hoặc 'Huỷ' để bỏ qua.")
        state["llm_response"] = "\n".join(lines)
        state["function_calls"] = None
        print(f"[Email Confirm] Pending confirmation for {len(new_pending_emails)} candidate(s)")
        return state

    state["function_calls"] = executed_calls

    # Second LLM call to synthesise tool results into a natural language response
    llm_no_tools   = _build_llm(temperature=0.3)
    final_response = await llm_no_tools.ainvoke(messages)
    state["llm_response"] = _extract_llm_text(final_response.content)

    return state
