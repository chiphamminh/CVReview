"""
Node 2 — Context Retrieval for Candidate Chatbot.

Routes to the correct retrieval sub-strategy based on pipeline_strategy
set by the Candidate Router. Each strategy fetches a different mix of
CV / JD context to minimise Qdrant round-trips and LLM token spend.

Strategy routing:
  STATUS_CHECK → bypass Qdrant entirely (SQL handled by LLM tool)
  APPLY        → bypass Qdrant entirely (uses scored_jobs cache from state)
  CV_ANALYSIS  → CV chunks only (candidate's own CV, scoped by candidateId/cvId)
  JD_CONVERSE  → JD chunks only (specific JD conversation — benefits, process)
  JD_ANALYSIS  → JD chunks only (what does this JD require?)
  JD_SEARCH    → CV + JD hybrid retrieval with Two-Stage Mode A/B:
                   Mode A (Turn 1, no scoring cache): full JD text fetched for scoring
                   Mode B (Turn 2+, cache exists):   reranked section chunks reused
"""

import asyncio
import traceback
from typing import List, Optional

from app.rag.candidate.state import CandidateChatState
from app.services.retriever import retriever
from app.services.recruitment_api import recruitment_api


async def retrieve_context_node(state: CandidateChatState) -> CandidateChatState:
    """Dispatch to the correct retrieval sub-strategy based on pipeline_strategy."""
    try:
        strategy = state.get("pipeline_strategy", "JD_SEARCH")

        # No retrieval needed
        if strategy in ("STATUS_CHECK", "APPLY"):
            return _bypass_retrieval(state, strategy)

        # CV-only retrieval
        if strategy == "CV_ANALYSIS":
            return await _retrieve_cv_only(state)

        # JD-only retrieval
        if strategy in ("JD_CONVERSE", "JD_ANALYSIS"):
            return await _retrieve_jd_only(state)

        # Default: JD_SEARCH — CV + JD hybrid with Two-Stage Mode A/B
        return await _retrieve_jd_search(state)

    except asyncio.TimeoutError:
        print("[Candidate Retrieve] Qdrant timeout — returning empty context")

        state["cv_context"] = []
        state["jd_context"] = []
        state["retrieval_stats"] = {"error": "retrieval_timeout"}

        return state

    except Exception as e:
        print(f"[Candidate Retrieve] Error: {e}")
        traceback.print_exc()

        state["cv_context"] = []
        state["jd_context"] = []
        state["retrieval_stats"] = {"error": str(e)}

        return state


# ---------------------------------------------------------------------------
# Sub-strategy implementations
# ---------------------------------------------------------------------------

def _bypass_retrieval(state: CandidateChatState, strategy: str) -> CandidateChatState:
    """Bypass Qdrant entirely. Downstream LLM/tool handles the response directly."""
    print(f"[Candidate Retrieve] strategy={strategy} → Bypass Qdrant")
    state["cv_context"]      = []
    state["jd_context"]      = []
    state["retrieval_stats"] = {"strategy": strategy, "note": "qdrant_bypassed"}
    return state


async def _retrieve_cv_only(state: CandidateChatState) -> CandidateChatState:
    """Retrieve CV chunks scoped to this candidate only (no JD needed)."""
    result = await retriever.retrieve_for_intent(
        query=state["query"],
        intent="cv_analysis",
        candidate_id=state.get("candidate_id"),
        cv_id=state.get("cv_id"),
    )
    state["cv_context"]      = result.get("cv_context", [])
    state["jd_context"]      = []
    state["retrieval_stats"] = result.get("retrieval_stats", {})
    print(f"[Candidate Retrieve] CV_ANALYSIS → {len(state['cv_context'])} CV chunks")
    return state


async def _retrieve_jd_only(state: CandidateChatState) -> CandidateChatState:
    """Retrieve JD chunks only (JD_CONVERSE / JD_ANALYSIS). No CV context needed."""
    result = await retriever.retrieve_for_intent(
        query=state["query"],
        intent="jd_analysis",
        jd_id=state.get("jd_id"),
        active_jd_ids=state.get("active_position_ids"),
    )
    state["cv_context"]      = []
    state["jd_context"]      = result.get("jd_context", [])
    state["retrieval_stats"] = result.get("retrieval_stats", {})
    print(f"[Candidate Retrieve] {state.get('pipeline_strategy')} → {len(state['jd_context'])} JD chunks")
    return state


async def _retrieve_jd_search(state: CandidateChatState) -> CandidateChatState:
    """
    JD_SEARCH — Two-Stage retrieval with Mode A/B cache logic.

    skill_variants from expansion node are now forwarded to
    retriever.retrieve_for_intent() so the keyword search leg (Qdrant MatchAny
    on the `skills` field) is actually triggered. Previously, expansion produced
    skill_variants but this function discarded them — the keyword leg was a no-op.

    Uses expanded_query from expansion node (if available) for the dense
    vector search to improve JD recall via synonym matching.

    Mode A (no scoring cache): fetches full JD text for deep scoring.
    Mode B (cache exists):     reuses reranked section chunks, skips full fetch.
    """
    has_scoring_cache = bool(state.get("scored_jobs"))
    # Use expanded query if available, else fall back to original
    search_query   = state.get("expanded_query") or state["query"]
    # Pass skill_variants so keyword search is triggered
    skill_variants = state.get("skill_variants") or []

    result = await retriever.retrieve_for_intent(
        query=search_query,
        intent="jd_search",
        candidate_id=state.get("candidate_id"),
        cv_id=state.get("cv_id"),
        active_jd_ids=state.get("active_position_ids"),
        skill_variants=skill_variants,   # <-- FIX: was missing before
    )

    cv_context  = result.get("cv_context", [])
    chunk_hits  = result.get("jd_context", [])
    jd_context  = chunk_hits

    if chunk_hits:
        # Deduplicate positionIds, preserving Qdrant rank order
        seen: set         = set()
        position_ids: List[int] = []
        for hit in chunk_hits:
            pid = hit.get("payload", {}).get("positionId")
            if pid is not None and pid not in seen:
                seen.add(pid)
                position_ids.append(pid)

        if not has_scoring_cache and position_ids:
            # Mode A: fetch full JD text so scoring_node has complete descriptions
            print(f"[Candidate Retrieve] Mode A (no cache): fetching full JD for {len(position_ids)} positions")
            try:
                full_jd_list = await recruitment_api.get_position_details(position_ids)
                jd_context = [
                    {
                        "score": 1.0,
                        "payload": {
                            "positionId":    jd["id"],
                            "positionTitle": jd.get("title", ""),
                            "seniority":     jd.get("seniority", ""),
                            "jdText":        jd.get("jdText", ""),
                        },
                    }
                    for jd in full_jd_list
                    if jd.get("jdText")
                ]
                print(f"[Candidate Retrieve] Mode A: {len(jd_context)} full-JD objects")
            except Exception as e:
                print(f"[Candidate Retrieve] Mode A: full JD fetch failed, using chunks: {e}")
                # jd_context already = chunk_hits, keep as-is
        else:
            # Mode B: reuse reranked chunks (scoring cache exists or no position IDs found)
            jd_context = chunk_hits
            reason = "scoring cache hit" if has_scoring_cache else "no position IDs"
            print(f"[Candidate Retrieve] Mode B ({reason}): {len(jd_context)} chunks")

        # active_position_ids == None means the API call failed in session_node;
        # skip the guard entirely to avoid wiping valid JD context.
        # active_position_ids == [] means the API succeeded but there are genuinely
        # no open positions — filtering is then valid (and will produce an empty list).
        active_ids_raw: Optional[List[int]] = state.get("active_position_ids")
        if active_ids_raw is not None and len(active_ids_raw) > 0:
            active_ids = set(active_ids_raw)
            before     = len(jd_context)
            jd_context = [
                jd for jd in jd_context
                if jd.get("payload", {}).get("positionId") in active_ids
            ]
            if len(jd_context) < before:
                print(
                    f"[Candidate Retrieve] Active-position guard: "
                    f"{before} → {len(jd_context)} JDs"
                )
        elif active_ids_raw is None:
            # API failed — log a warning but keep jd_context untouched
            print(
                "[Candidate Retrieve] WARNING: active_position_ids is None "
                "(API failure in session_node) — skipping active-position guard"
            )

    state["cv_context"]      = cv_context
    state["jd_context"]      = jd_context
    state["retrieval_stats"] = result.get("retrieval_stats", {})
    return state