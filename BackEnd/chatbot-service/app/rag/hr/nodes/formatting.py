from app.rag.hr.state import HRChatState

def format_hr_response_node(state: HRChatState) -> HRChatState:
    """Package the final answer and metadata for the API layer."""
    state["final_answer"] = state["llm_response"]
    state["metadata"] = {
        "mode":              state["mode"],
        "position_id":       state["position_id"],
        "is_cv_count_query": state.get("is_cv_count_query", False),
        "cv_chunks_used":    len(state.get("cv_context", [])),
        "sql_records_count": len(state.get("sql_metadata", [])),
        "function_calls":    state.get("function_calls"),
        "retrieval_stats":   state.get("retrieval_stats", {}),
        "pending_emails":    state.get("pending_emails"),
    }
    return state
