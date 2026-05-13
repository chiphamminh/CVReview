"""
LangGraph workflow for the Candidate chatbot.

Graph topology (Sprint 2 — Intent-Aware Routing + Hybrid Retrieval):

  load_session_history
          │
  route_candidate_intent      ← Hard-rule router (pure function, zero LLM cost)
          │
          ├─ STATUS_CHECK / APPLY ─────────────────────────────────┐
          │    (bypass Qdrant; retrieval node sets cv_context=[])   │
          │                                                          │
          ├─ CV_ANALYSIS ──────────────────────────────────────────┤
          │    (CV chunks only, no expansion)                        │
          │                                                          │
          ├─ JD_CONVERSE / JD_ANALYSIS ────────────────────────────┤
          │    (JD chunks only, no expansion)                        │
          │                                                          │
          └─ JD_SEARCH ──────────────────────────────────────────── ┤
               │                                                     │
          query_expansion   ← LLM Flash (synonym expand, ~50-100ms) │
               │                                                     │
          retrieve_context ←───────────────────────────────────────┘
               │
             scoring      (JD_SEARCH Turn 1 only; cache skips)
               │
          build_prompts
               │
          llm_reasoning
               │
           save_turn
               │
        format_response → END
"""

import json
import time
import asyncio
import functools
from typing import Literal, Optional, Dict, Any, AsyncGenerator
from langgraph.graph import StateGraph, END

from app.rag.candidate.state import CandidateChatState
from app.rag.candidate.router import route_candidate_intent_node
from app.rag.candidate.nodes.session import load_session_history_node
from app.rag.candidate.nodes.expansion import query_expansion_node
from app.rag.candidate.nodes.retrieval import retrieve_context_node
from app.rag.candidate.nodes.scoring import scoring_node
from app.rag.candidate.nodes.prompts import build_prompts_node
from app.rag.candidate.nodes.reasoning import llm_reasoning_node
from app.rag.candidate.nodes.persistence import save_turn_node
from app.rag.candidate.nodes.formatting import format_response_node


def _timed(name: str, fn):
    """Wrap a sync or async node function with wall-clock timing output."""
    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async(state):
            t0 = time.perf_counter()
            result = await fn(state)
            print(f"[TIMER][Candidate] {name}: {time.perf_counter() - t0:.2f}s")
            return result
        return _async
    else:
        @functools.wraps(fn)
        def _sync(state):
            t0 = time.perf_counter()
            result = fn(state)
            print(f"[TIMER][Candidate] {name}: {time.perf_counter() - t0:.2f}s")
            return result
        return _sync


def _extract_text_token(chunk) -> str:
    """Return plain text from an AIMessageChunk, skipping tool-call blocks."""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


# Status messages emitted as SSE events when each heavy node starts.
# Node-based: messages appear only for stages the current intent actually visits.
_NODE_STATUS: dict[str, str] = {
    "load_session_history": "Đang tải lịch sử phiên...",
    "query_expansion":      "Đang phân tích từ khoá tìm kiếm...",
    "retrieve_context":     "Đang tìm kiếm thông tin liên quan...",
    "scoring":              "Đang đánh giá độ phù hợp với vị trí...",
    "llm_reasoning":        "Đang phân tích và tổng hợp kết quả...",
}


def _route_after_router(state: CandidateChatState) -> Literal["query_expansion", "retrieve_context"]:
    """Branch to expansion only for JD_SEARCH; all other strategies skip directly to retrieval."""
    if state.get("pipeline_strategy") == "JD_SEARCH":
        return "query_expansion"
    return "retrieve_context"


def create_candidate_graph():
    workflow = StateGraph(CandidateChatState)

    workflow.add_node("load_session_history",    _timed("load_session_history",   load_session_history_node))
    workflow.add_node("route_candidate_intent",  _timed("route_candidate_intent", route_candidate_intent_node))
    workflow.add_node("query_expansion",         _timed("query_expansion",        query_expansion_node))
    workflow.add_node("retrieve_context",        _timed("retrieve_context",       retrieve_context_node))
    workflow.add_node("scoring",                 _timed("scoring",                scoring_node))
    workflow.add_node("build_prompts",           _timed("build_prompts",          build_prompts_node))
    workflow.add_node("llm_reasoning",           _timed("llm_reasoning",          llm_reasoning_node))
    workflow.add_node("save_turn",               _timed("save_turn",              save_turn_node))
    workflow.add_node("format_response",         _timed("format_response",        format_response_node))

    workflow.set_entry_point("load_session_history")
    workflow.add_edge("load_session_history", "route_candidate_intent")

    workflow.add_conditional_edges(
        "route_candidate_intent",
        _route_after_router,
        {
            "query_expansion":  "query_expansion",
            "retrieve_context": "retrieve_context",
        },
    )

    workflow.add_edge("query_expansion", "retrieve_context")
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
            "active_position_ids":  None,
            "position_ref_map":     {},
            "cv_context":           [],
            "jd_context":           [],
            "retrieval_stats":      {},
            "scored_jobs":          None,
            # Router fields
            "pipeline_strategy":    "",
            "query_intent":         "",
            "query_entities":       {},
            "expanded_query":       None,
            "skill_variants":       [],
            # Legacy fields (for scoring/prompts backward compat)
            "intent":               "general",
            "intent_confidence":    0.0,
            "domain":               "candidate",
            "is_apply_intent":      False,
            # LLM pipeline
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

    async def stream_chat(
        self,
        query: str,
        session_id: str,
        candidate_id: str,
        cv_id: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        initial_state: CandidateChatState = {
            "query":                query,
            "session_id":           session_id,
            "candidate_id":         candidate_id,
            "cv_id":                cv_id,
            "jd_id":                None,
            "conversation_history": [],
            "active_position_ids":  None,
            "position_ref_map":     {},
            "cv_context":           [],
            "jd_context":           [],
            "retrieval_stats":      {},
            "scored_jobs":          None,
            "pipeline_strategy":    "",
            "query_intent":         "",
            "query_entities":       {},
            "expanded_query":       None,
            "skill_variants":       [],
            "intent":               "general",
            "intent_confidence":    0.0,
            "domain":               "candidate",
            "is_apply_intent":      False,
            "system_prompt":        "",
            "user_prompt":          "",
            "llm_response":         "",
            "function_calls":       None,
            "final_answer":         "",
            "metadata":             {},
        }

        final_answer    = ""
        final_metadata: Dict[str, Any] = {}
        _sent_statuses: set[str] = set()

        async for event in self.graph.astream_events(
            initial_state, version="v2", config={"recursion_limit": 50}
        ):
            event_type = event["event"]
            node       = event.get("metadata", {}).get("langgraph_node", "")

            if event_type == "on_chain_start" and node not in _sent_statuses:
                status = _NODE_STATUS.get(node)
                if status:
                    _sent_statuses.add(node)
                    yield f"data: {json.dumps({'status': status}, ensure_ascii=False)}\n\n"

            elif event_type == "on_chat_model_stream" and node == "llm_reasoning":
                token = _extract_text_token(event["data"]["chunk"])
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"

            elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
                output         = event["data"].get("output") or {}
                final_answer   = output.get("final_answer", "")
                final_metadata = output.get("metadata", {})

        yield f"data: {json.dumps({'done': True, 'metadata': final_metadata, 'fallback_answer': final_answer})}\n\n"


candidate_chatbot = CandidateChatbot()