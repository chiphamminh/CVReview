"""Node 5 — Package the final answer and metadata for the API layer."""

from app.rag.candidate.state import CandidateChatState
from app.rag.candidate.nodes.reasoning import _get_temperature


def format_response_node(state: CandidateChatState) -> CandidateChatState:
    """Assemble the API response envelope from the final graph state."""
    state["final_answer"] = state["llm_response"]
    state["metadata"] = {
        "intent":            state["intent"],
        "intent_confidence": state["intent_confidence"],
        "domain":            state["domain"],
        "is_apply_intent":   state.get("is_apply_intent", False),
        "cv_chunks_used":    len(state.get("cv_context", [])),
        "jd_docs_used":      len(state.get("jd_context", [])),
        "temperature_used":  _get_temperature(state["intent"]),
        "function_calls":    state.get("function_calls"),
        "scored_jobs":       state.get("scored_jobs"),
    }
    return state
