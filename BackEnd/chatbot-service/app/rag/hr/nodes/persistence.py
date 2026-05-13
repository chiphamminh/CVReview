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
        ranked_cv_list = state.get("ranked_cv_list") or []
        conv_state     = state.get("conv_state", "IDLE")
        pending_action = state.get("pending_action")

        # Build the session cache dict — always written so session node can restore
        # conv_state, ranked_cv_list, and active_cv_ids on the next turn.
        scored_cv_ids  = state.get("scored_cv_ids") or []

        def _base_cache() -> dict:
            cache: dict = {"conv_state": conv_state}
            if active_cv_ids:
                cache["active_cv_ids"] = active_cv_ids
            if ranked_cv_list:
                cache["ranked_cv_list"] = ranked_cv_list
            if pending_action:
                cache["pending_action"] = pending_action
            if scored_cv_ids:
                cache["scored_cv_ids"] = scored_cv_ids
            return cache

        if raw_calls:
            payload_dict = (
                raw_calls if isinstance(raw_calls, dict)
                else {"calls": raw_calls}
            )
            payload_dict.update(_base_cache())
            function_call_payload: Optional[str] = json.dumps(payload_dict, ensure_ascii=False)
        elif pending_emails:
            import uuid
            action_id = str(uuid.uuid4())
            for pe in pending_emails:
                pe["_action_id"] = action_id
            cache = _base_cache()
            cache["pending_emails"] = pending_emails
            function_call_payload = json.dumps(cache, ensure_ascii=False)
        else:
            cache = _base_cache()
            function_call_payload = json.dumps(cache, ensure_ascii=False) if cache else None

        await recruitment_api.save_message(
            session_id=state["session_id"],
            role="ASSISTANT",
            content=state["llm_response"],
            function_call=function_call_payload,
        )
    except Exception as e:
        print(f"[HR Graph] Could not save turn: {e}")

    return state
