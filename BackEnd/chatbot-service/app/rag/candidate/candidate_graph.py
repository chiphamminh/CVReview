"""
LangGraph workflow for the Candidate chatbot.

Graph topology (Sprint 1 — legacy intent routing, to be replaced by
candidate/router.py in Sprint 2):

  load_session_history
          │
  classify_intent
          │
    ┌─────┴──────┐
    ▼            ▼
retrieve_context  build_prompts (skip_retrieve path)
    │            │
  scoring        │
    └────────────┘
          │
    build_prompts
          │
    llm_reasoning
          │
      save_turn
          │
   format_response → END

Phase 4 features preserved:
  - Multi-dimensional scoring (scoring_node) with POOR_FIT guardrail.
  - Tầng 1 hard-rule apply-intent detection (session_node).
  - Dual-mode JD retrieval (Mode A full-JD / Mode B chunk cache).
  - scored_jobs cache persisted via functionCall field for cross-turn restore.
"""

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END

from app.rag.candidate.state import CandidateChatState
from app.rag.candidate.nodes.session import load_session_history_node
from app.rag.candidate.nodes.intent import classify_intent_node, should_retrieve
from app.rag.candidate.nodes.retrieval import retrieve_context_node
from app.rag.candidate.nodes.scoring import scoring_node
from app.rag.candidate.nodes.prompts import build_prompts_node
from app.rag.candidate.nodes.reasoning import llm_reasoning_node
from app.rag.candidate.nodes.persistence import save_turn_node
from app.rag.candidate.nodes.formatting import format_response_node


def create_candidate_graph():
    workflow = StateGraph(CandidateChatState)

    workflow.add_node("load_session_history", load_session_history_node)
    workflow.add_node("classify_intent",      classify_intent_node)
    workflow.add_node("retrieve_context",     retrieve_context_node)
    workflow.add_node("scoring",              scoring_node)
    workflow.add_node("build_prompts",        build_prompts_node)
    workflow.add_node("llm_reasoning",        llm_reasoning_node)
    workflow.add_node("save_turn",            save_turn_node)
    workflow.add_node("format_response",      format_response_node)

    workflow.set_entry_point("load_session_history")
    workflow.add_edge("load_session_history", "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        should_retrieve,
        {"retrieve": "retrieve_context", "skip_retrieve": "build_prompts"},
    )

    workflow.add_edge("retrieve_context", "scoring")
    workflow.add_edge("scoring",          "build_prompts")
    workflow.add_edge("build_prompts",    "llm_reasoning")
    workflow.add_edge("llm_reasoning",    "save_turn")
    workflow.add_edge("save_turn",        "format_response")
    workflow.add_edge("format_response",  END)

    return workflow.compile()


class CandidateChatbot:
    def __init__(self):
        self.graph = create_candidate_graph()

    async def chat(
        self,
        query: str,
        session_id: str,
        candidate_id: str,
        cv_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        initial_state: CandidateChatState = {
            "query":                query,
            "session_id":           session_id,
            "candidate_id":         candidate_id,
            "cv_id":                cv_id,
            "jd_id":                None,
            "conversation_history": [],
            "active_position_ids":  [],
            "position_ref_map":     {},
            "cv_context":           [],
            "jd_context":           [],
            "retrieval_stats":      {},
            "scored_jobs":          None,
            "is_apply_intent":      False,
            "system_prompt":        "",
            "user_prompt":          "",
            "llm_response":         "",
            "function_calls":       None,
            "final_answer":         "",
            "metadata":             {},
        }

        final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 50})

        return {
            "answer":   final_state["final_answer"],
            "metadata": final_state["metadata"],
        }


candidate_chatbot = CandidateChatbot()
