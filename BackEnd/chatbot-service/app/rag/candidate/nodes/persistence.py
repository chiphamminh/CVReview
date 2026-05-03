"""Node 4.5 — Persist user + assistant messages to recruitment-service (MySQL)."""

import json
from typing import Optional

from app.rag.candidate.state import CandidateChatState
from app.services.recruitment_api import recruitment_api


async def save_turn_node(state: CandidateChatState) -> CandidateChatState:
    """Save user + assistant messages to recruitment-service via internal API."""
    try:
        await recruitment_api.save_message(
            session_id=state["session_id"],
            role="USER",
            content=state["query"],
        )

        function_call_payload: Optional[str] = None
        raw_calls = state.get("function_calls")
        if raw_calls:
            function_call_payload = json.dumps(raw_calls, ensure_ascii=False)
        elif state.get("scored_jobs"):
            # Persist scored_jobs so the next turn can restore the cache.
            function_call_payload = json.dumps(
                {"scored_jobs": state["scored_jobs"]}, ensure_ascii=False
            )

        await recruitment_api.save_message(
            session_id=state["session_id"],
            role="ASSISTANT",
            content=state["llm_response"],
            function_call=function_call_payload,
        )
    except Exception as e:
        print(f"[API Error] Could not save turn: {e}")

    return state
