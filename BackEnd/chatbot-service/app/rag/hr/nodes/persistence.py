import json
from typing import Optional
from app.rag.hr.state import HRChatState
from app.services.recruitment_api import recruitment_api

async def save_hr_turn_node(state: HRChatState) -> HRChatState:
    """Persist user + assistant messages to recruitment-service (MySQL)."""
    try:
        await recruitment_api.save_message(
            session_id=state["session_id"],
            role="USER",
            content=state["query"],
        )
        raw_calls      = state.get("function_calls")
        pending_emails = state.get("pending_emails")
        active_cv_ids  = state.get("active_cv_ids") or []

        if raw_calls:
            # ISSUE-01: embed active_cv_ids into the function_call payload so it
            # survives session reload and can be restored on the next turn.
            payload_dict = (
                raw_calls if isinstance(raw_calls, dict)
                else {"calls": raw_calls}
            )
            payload_dict["active_cv_ids"] = active_cv_ids
            function_call_payload: Optional[str] = json.dumps(payload_dict, ensure_ascii=False)
        elif pending_emails:
            import uuid
            # ISSUE-05: Tạo action_id cho mỗi lần có pending emails
            action_id = str(uuid.uuid4())
            for pe in pending_emails:
                pe["_action_id"] = action_id
            
            cache: dict = {"pending_emails": pending_emails}
            if active_cv_ids:
                cache["active_cv_ids"] = active_cv_ids
            function_call_payload = json.dumps(cache, ensure_ascii=False)
        elif active_cv_ids:
            function_call_payload = json.dumps(
                {"active_cv_ids": active_cv_ids}, ensure_ascii=False
            )
        else:
            function_call_payload = None

        await recruitment_api.save_message(
            session_id=state["session_id"],
            role="ASSISTANT",
            content=state["llm_response"],
            function_call=function_call_payload,
        )
    except Exception as e:
        print(f"[HR Graph] Could not save turn: {e}")

    return state
