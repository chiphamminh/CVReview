"""
LangGraph workflow for the HR chatbot.

Graph topology (Sprint 1 — Intent-Aware Routing):

  load_hr_session_history
          │
  load_candidate_scope
          │
  route_hr_intent          ← Hard-rule router, pure function, no API calls
          │
          ├─ ACTION / AGGREGATE ──────────────────────────────────┐
          │    (bypass Qdrant; retrieval node sets cv_context=[])  │
          │                                                         │
          ├─ COMPARE / DETAIL ──────────────────────────────────── ┤
          │    (pinned fetch from active_cv_ids)                    │
          │                                                         │
          └─ RANK / FILTER / FIND_MORE ─────────────────────────── ┤
               (full retrieval pipeline)                            │
                                                                    ▼
                                                         retrieve_hr_context
                                                                    │
                                                          build_hr_prompts
                                                                    │
                                                          llm_hr_reasoning
                                                           │            │
                                                     hr_scoring    save_hr_turn
                                                           └────────────┘
                                                                    │
                                                        format_hr_response → END
"""

from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END

from app.rag.hr.state import HRChatState
from app.rag.hr.router import route_hr_intent_node
from app.rag.hr.nodes.session import load_hr_session_history_node
from app.rag.hr.nodes.scope import load_candidate_scope_node
from app.rag.hr.nodes.retrieval import retrieve_hr_context_node
from app.rag.hr.nodes.prompts import build_hr_prompts_node
from app.rag.hr.nodes.reasoning import llm_hr_reasoning_node
from app.rag.hr.nodes.scoring import hr_scoring_node
from app.rag.hr.nodes.persistence import save_hr_turn_node
from app.rag.hr.nodes.formatting import format_hr_response_node


def _route_after_reasoning(state: HRChatState) -> str:
    """Branch to hr_scoring if evaluate_candidates was intercepted, else persist."""
    if state.get("pending_scoring_candidates"):
        return "hr_scoring"
    return "save_hr_turn"


def create_hr_graph():
    workflow = StateGraph(HRChatState)

    # Register nodes
    workflow.add_node("load_hr_session_history", load_hr_session_history_node)
    workflow.add_node("load_candidate_scope",    load_candidate_scope_node)
    workflow.add_node("route_hr_intent",         route_hr_intent_node)
    workflow.add_node("retrieve_hr_context",     retrieve_hr_context_node)
    workflow.add_node("build_hr_prompts",        build_hr_prompts_node)
    workflow.add_node("llm_hr_reasoning",        llm_hr_reasoning_node)
    workflow.add_node("hr_scoring",              hr_scoring_node)
    workflow.add_node("save_hr_turn",            save_hr_turn_node)
    workflow.add_node("format_hr_response",      format_hr_response_node)

    # Linear head: session → scope → router
    workflow.set_entry_point("load_hr_session_history")
    workflow.add_edge("load_hr_session_history", "load_candidate_scope")
    workflow.add_edge("load_candidate_scope",    "route_hr_intent")

    # Router always flows into retrieve_hr_context (which handles bypassing internally)
    workflow.add_edge("route_hr_intent", "retrieve_hr_context")

    # Linear body: retrieval → prompts → LLM
    workflow.add_edge("retrieve_hr_context", "build_hr_prompts")
    workflow.add_edge("build_hr_prompts",    "llm_hr_reasoning")

    # After reasoning: optionally score, then persist
    workflow.add_conditional_edges(
        "llm_hr_reasoning",
        _route_after_reasoning,
        {"hr_scoring": "hr_scoring", "save_hr_turn": "save_hr_turn"},
    )
    workflow.add_edge("hr_scoring",       "save_hr_turn")
    workflow.add_edge("save_hr_turn",     "format_hr_response")
    workflow.add_edge("format_hr_response", END)

    return workflow.compile()


class HRChatbot:
    def __init__(self):
        self.graph = create_hr_graph()

    async def chat(
        self,
        query: str,
        session_id: str,
        hr_id: str,
        position_id: int,
        mode: Literal["HR_MODE", "CANDIDATE_MODE"],
    ) -> Dict[str, Any]:
        initial_state: HRChatState = {
            "query":                      query,
            "hr_id":                      hr_id,
            "session_id":                 session_id,
            "position_id":                position_id,
            "mode":                       mode,
            # Legacy flag kept for backward-compat with session node
            "is_cv_count_query":          False,
            "sql_metadata":               [],
            "cv_id_to_meta":              {},
            "pending_emails":             None,
            "pending_scoring_candidates": None,
            "conversation_history":       [],
            "active_cv_ids":              [],
            "hr_query_intent":            "RANK",  # kept for session.py compat
            "cv_context":                 [],
            "jd_context":                 [],
            "retrieval_stats":            {},
            "system_prompt":              "",
            "user_prompt":                "",
            "llm_response":               "",
            "function_calls":             None,
            "final_answer":               "",
            "metadata":                   {},
            # Sprint 1 — Router fields
            "pipeline_strategy":          "",
            "query_intent":               "",
            "query_entities":             {},
            "expanded_query":             None,
            "skill_variants":             [],
        }

        final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 50})

        return {
            "answer":   final_state["final_answer"],
            "metadata": final_state["metadata"],
        }


hr_chatbot = HRChatbot()
