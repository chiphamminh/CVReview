"""Node 3 — Build system + user prompts for the LLM reasoning step."""

from app.rag.candidate.state import CandidateChatState
from app.rag.prompts import get_prompt_for_intent


def build_prompts_node(state: CandidateChatState) -> CandidateChatState:
    """Assemble the system + user prompts, including multi-dimensional scores for jd_search."""
    system_prompt, user_prompt = get_prompt_for_intent(
        intent=state["intent"],
        query=state["query"],
        cv_context=state.get("cv_context", []),
        jd_context=state.get("jd_context", []),
        conversation_history=state.get("conversation_history", []),
        scored_jobs=state.get("scored_jobs"),
    )
    state["system_prompt"] = system_prompt
    state["user_prompt"]   = user_prompt
    return state
