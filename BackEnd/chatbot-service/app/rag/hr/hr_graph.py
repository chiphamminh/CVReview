"""
LangGraph workflow for the HR chatbot.

Graph topology (Sprint 2 — Hybrid Retrieval + Expansion):

  load_hr_session_history
          │
  load_candidate_scope
          │
  route_hr_intent          ← Hard-rule router, pure function, no API calls
          │
          ├─ ACTION / AGGREGATE ──────────────────────────────────────┐
          │    (bypass Qdrant; retrieval node sets cv_context=[])      │
          │                                                             │
          ├─ COMPARE / DETAIL ────────────────────────────────────── ──┤
          │    (pinned scroll fetch from active_cv_ids, no expansion)   │
          │                                                             │
          └─ RANK / FILTER / FIND_MORE ──────────────────────────────┐ │
               │                                                      │ │
          query_expansion   ← LLM Flash (synonym expand, ~50-100ms)  │ │
               │                                                      │ │
               └──────────────────────────────────────────────────────┘ │
                                                                         │
                                                          retrieve_hr_context ←┘
                                                                    │
                                                          build_hr_prompts
                                                                    │
                                                          llm_hr_reasoning
                                                           │            │
                                                     hr_scoring    save_hr_turn
                                                           └────────────┘
                                                                    │
                                                        format_hr_response → END

ISSUE #5 FIX: query_expansion_node is now registered and wired between
route_hr_intent and retrieve_hr_context.

  - RANK / FILTER / FIND_MORE → query_expansion → retrieve_hr_context
  - ACTION / AGGREGATE / COMPARE / DETAIL → retrieve_hr_context (skip expansion)

The conditional edge _route_after_router handles this branch.
"""

import json
import time
import asyncio
import functools
from typing import Literal, Dict, Any, AsyncGenerator
from langgraph.graph import StateGraph, END

from app.rag.hr.state import HRChatState
from app.rag.hr.router import route_hr_intent_node
from app.rag.hr.nodes.session import load_hr_session_history_node
from app.rag.hr.nodes.scope import load_candidate_scope_node
from app.rag.hr.nodes.expansion import query_expansion_node          # FIX ISSUE #5
from app.rag.hr.nodes.retrieval import retrieve_hr_context_node
from app.rag.hr.nodes.prompts import build_hr_prompts_node
from app.rag.hr.nodes.reasoning import llm_hr_reasoning_node
from app.rag.hr.nodes.scoring import hr_scoring_node
from app.rag.hr.nodes.persistence import save_hr_turn_node
from app.rag.hr.nodes.formatting import format_hr_response_node


def _timed(name: str, fn):
    """Wrap a sync or async node function with wall-clock timing output."""
    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async(state):
            t0 = time.perf_counter()
            result = await fn(state)
            print(f"[TIMER][HR] {name}: {time.perf_counter() - t0:.2f}s")
            return result
        return _async
    else:
        @functools.wraps(fn)
        def _sync(state):
            t0 = time.perf_counter()
            result = fn(state)
            print(f"[TIMER][HR] {name}: {time.perf_counter() - t0:.2f}s")
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

# Strategies that require LLM query expansion before Qdrant retrieval.
# Must stay in sync with _EXPANSION_STRATEGIES in hr/nodes/expansion.py.
_EXPANSION_STRATEGIES = {"RANK", "FILTER", "FIND_MORE"}

# Status messages emitted as SSE events when each heavy node starts.
# Node-based (not intent-based): each intent only visits the nodes it needs,
# so messages appear automatically for relevant stages and never for skipped ones.
_NODE_STATUS: dict[str, str] = {
    "load_hr_session_history": "Đang tải lịch sử phiên...",
    "load_candidate_scope":    "Đang tải danh sách ứng viên...",
    "query_expansion":         "Đang phân tích từ khoá tìm kiếm...",
    "retrieve_hr_context":     "Đang tìm kiếm ứng viên phù hợp...",
    "llm_hr_reasoning":        "Đang phân tích và tổng hợp kết quả...",
    "hr_scoring":              "Đang đánh giá và chấm điểm ứng viên...",
}


def _route_after_router(state: HRChatState) -> str:
    """
    Conditional edge after route_hr_intent.

    RANK / FILTER / FIND_MORE → query_expansion (then retrieval)
    All others                → retrieve_hr_context directly (expansion skipped)

    This keeps ACTION/AGGREGATE/COMPARE/DETAIL at zero LLM expansion cost.
    """
    strategy = state.get("pipeline_strategy", "RANK")
    if strategy in _EXPANSION_STRATEGIES:
        return "query_expansion"
    return "retrieve_hr_context"


def _route_after_reasoning(state: HRChatState) -> str:
    """Branch to hr_scoring if evaluate_candidates was intercepted, else persist."""
    if state.get("pending_scoring_candidates"):
        return "hr_scoring"
    return "save_hr_turn"


def create_hr_graph():
    workflow = StateGraph(HRChatState)

    # Register nodes — each wrapped with _timed() to log wall-clock duration per node.
    # Check the server console for [TIMER][HR] lines to locate the bottleneck.
    workflow.add_node("load_hr_session_history", _timed("load_hr_session_history", load_hr_session_history_node))
    workflow.add_node("load_candidate_scope",    _timed("load_candidate_scope",    load_candidate_scope_node))
    workflow.add_node("route_hr_intent",         _timed("route_hr_intent",         route_hr_intent_node))
    workflow.add_node("query_expansion",         _timed("query_expansion",         query_expansion_node))
    workflow.add_node("retrieve_hr_context",     _timed("retrieve_hr_context",     retrieve_hr_context_node))
    workflow.add_node("build_hr_prompts",        _timed("build_hr_prompts",        build_hr_prompts_node))
    workflow.add_node("llm_hr_reasoning",        _timed("llm_hr_reasoning",        llm_hr_reasoning_node))
    workflow.add_node("hr_scoring",              _timed("hr_scoring",              hr_scoring_node))
    workflow.add_node("save_hr_turn",            _timed("save_hr_turn",            save_hr_turn_node))
    workflow.add_node("format_hr_response",      _timed("format_hr_response",      format_hr_response_node))

    # Linear head: session → scope → router
    workflow.set_entry_point("load_hr_session_history")
    workflow.add_edge("load_hr_session_history", "load_candidate_scope")
    workflow.add_edge("load_candidate_scope",    "route_hr_intent")

    # RANK / FILTER / FIND_MORE → query_expansion → retrieve_hr_context
    # ACTION / AGGREGATE / COMPARE / DETAIL → retrieve_hr_context (skip expansion)
    workflow.add_conditional_edges(
        "route_hr_intent",
        _route_after_router,
        {
            "query_expansion":   "query_expansion",
            "retrieve_hr_context": "retrieve_hr_context",
        },
    )
    workflow.add_edge("query_expansion", "retrieve_hr_context")

    # Linear body: retrieval → prompts → LLM
    workflow.add_edge("retrieve_hr_context", "build_hr_prompts")
    workflow.add_edge("build_hr_prompts",    "llm_hr_reasoning")

    # After reasoning: optionally score, then persist
    workflow.add_conditional_edges(
        "llm_hr_reasoning",
        _route_after_reasoning,
        {"hr_scoring": "hr_scoring", "save_hr_turn": "save_hr_turn"},
    )
    workflow.add_edge("hr_scoring",         "save_hr_turn")
    workflow.add_edge("save_hr_turn",       "format_hr_response")
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
        mode: Literal["INTERNAL", "EXTERNAL"],
    ) -> Dict[str, Any]:
        initial_state: HRChatState = {
            "query":                      query,
            "hr_id":                      hr_id,
            "session_id":                 session_id,
            "position_id":                position_id,
            "mode":                       mode,
            "is_cv_count_query":          False,
            "sql_metadata":               [],
            "cv_id_to_meta":              {},
            "pending_emails":             None,
            "pending_scoring_candidates": None,
            "conversation_history":       [],
            "active_cv_ids":              [],
            "hr_query_intent":            "RANK",
            "cv_context":                 [],
            "jd_context":                 [],
            "full_jd_text":               "",
            "retrieval_stats":            {},
            "system_prompt":              "",
            "user_prompt":                "",
            "llm_response":               "",
            "function_calls":             None,
            "final_answer":               "",
            "metadata":                   {},
            "pipeline_strategy":          "",
            "query_intent":               "",
            "query_entities":             {},
            "expanded_query":             None,
            "skill_variants":             [],
            "scored_cv_ids":              [],
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
        hr_id: str,
        position_id: int,
        mode: Literal["INTERNAL", "EXTERNAL"],
    ) -> AsyncGenerator[str, None]:
        initial_state: HRChatState = {
            "query":                      query,
            "hr_id":                      hr_id,
            "session_id":                 session_id,
            "position_id":                position_id,
            "mode":                       mode,
            "is_cv_count_query":          False,
            "sql_metadata":               [],
            "cv_id_to_meta":              {},
            "pending_emails":             None,
            "pending_scoring_candidates": None,
            "conversation_history":       [],
            "active_cv_ids":              [],
            "hr_query_intent":            "RANK",
            "cv_context":                 [],
            "jd_context":                 [],
            "full_jd_text":               "",
            "retrieval_stats":            {},
            "system_prompt":              "",
            "user_prompt":                "",
            "llm_response":               "",
            "function_calls":             None,
            "final_answer":               "",
            "metadata":                   {},
            "pipeline_strategy":          "",
            "query_intent":               "",
            "query_entities":             {},
            "expanded_query":             None,
            "skill_variants":             [],
            "scored_cv_ids":              [],
        }

        final_answer      = ""
        final_metadata: Dict[str, Any] = {}
        _sent_statuses: set[str] = set()
        # Only stream tokens for intents that go through hr_scoring_node.
        # For all other intents the done event's fallback_answer is the canonical
        # response — emitting tokens causes a race where the frontend receives
        # `done` before rendering all tokens, producing truncated output.
        _STREAMING_STRATEGIES = {"RANK", "FIND_MORE"}
        _pipeline_strategy    = ""

        async for event in self.graph.astream_events(
            initial_state, version="v2", config={"recursion_limit": 50}
        ):
            event_type = event["event"]
            node       = event.get("metadata", {}).get("langgraph_node", "")

            if event_type == "on_chain_start":
                # Capture strategy from llm_hr_reasoning input — by this point
                # route_hr_intent has already written pipeline_strategy to state.
                if node == "llm_hr_reasoning":
                    _inp = event["data"].get("input")
                    if isinstance(_inp, dict):
                        _pipeline_strategy = _inp.get("pipeline_strategy", "")
                # Status message
                if node not in _sent_statuses:
                    status = _NODE_STATUS.get(node)
                    if status:
                        _sent_statuses.add(node)
                        yield f"data: {json.dumps({'status': status}, ensure_ascii=False)}\n\n"

            elif event_type == "on_chat_model_stream" and node == "llm_hr_reasoning":
                if _pipeline_strategy in _STREAMING_STRATEGIES:
                    token = _extract_text_token(event["data"]["chunk"])
                    if token:
                        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            elif event_type == "on_chain_end" and node == "format_hr_response":
                # Primary capture: format_hr_response is the last node and always
                # sets final_answer = llm_response. More reliable than the graph-end
                # event whose name can vary across LangGraph versions.
                _out = event["data"].get("output")
                if isinstance(_out, dict):
                    final_answer   = _out.get("final_answer", "")
                    final_metadata = _out.get("metadata", {})

            elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
                # Fallback: also try graph-end event in case format_hr_response
                # output was not a dict (LangGraph version-specific behaviour).
                _out = event["data"].get("output") or {}
                if isinstance(_out, dict) and not final_answer:
                    final_answer   = _out.get("final_answer", "")
                    final_metadata = _out.get("metadata", {})

        yield f"data: {json.dumps({'done': True, 'metadata': final_metadata, 'fallback_answer': final_answer}, ensure_ascii=False)}\n\n"


hr_chatbot = HRChatbot()