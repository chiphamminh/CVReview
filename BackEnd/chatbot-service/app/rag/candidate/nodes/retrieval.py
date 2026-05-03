"""
Node 2 — Dual-mode JD retrieval with reranking.

Mode A (Turn 1, no scoring cache): Fetches full JD text after reranking so
  the scoring node has complete job descriptions for multi-dimensional scoring.
Mode B (Turn 2+, cache exists): Uses reranked section chunks directly to
  avoid redundant full-JD fetches.
"""

from typing import List

from app.rag.candidate.state import CandidateChatState
from app.services.retriever import retriever
from app.services.recruitment_api import recruitment_api


async def retrieve_context_node(state: CandidateChatState) -> CandidateChatState:
    """Retrieve CV and JD context from Qdrant based on the classified intent."""
    intent = state["intent"]
    has_scoring_cache = bool(state.get("scored_jobs"))

    result = await retriever.retrieve_for_intent(
        query=state["query"],
        intent=intent,
        candidate_id=state.get("candidate_id"),
        cv_id=state.get("cv_id"),
        jd_id=state.get("jd_id"),
        active_jd_ids=state.get("active_position_ids") if intent == "jd_search" else None,
    )

    cv_context = result.get("cv_context", [])
    chunk_hits  = result.get("jd_context", [])
    jd_context  = chunk_hits

    if intent in ("jd_search", "jd_analysis") and chunk_hits:
        seen: set = set()
        position_ids: List[int] = []
        for hit in chunk_hits:
            pid = hit.get("payload", {}).get("positionId")
            if pid is not None and pid not in seen:
                seen.add(pid)
                position_ids.append(pid)

        if not has_scoring_cache and position_ids:
            print(f"[Retriever] Mode A (no cache): fetching full JD for {len(position_ids)} positions")
            try:
                full_jd_list = await recruitment_api.get_position_details(position_ids)
                jd_context = [
                    {
                        "score": 1.0,
                        "payload": {
                            "positionId":   jd["id"],
                            "positionName": jd.get("name", ""),
                            "language":     jd.get("language", ""),
                            "level":        jd.get("level", ""),
                            "jdText":       jd.get("jdText", ""),
                        },
                    }
                    for jd in full_jd_list
                    if jd.get("jdText")
                ]
                print(f"[Retriever] Mode A: Expanded to {len(jd_context)} full-JD objects")
            except Exception as e:
                print(f"[Retriever] Mode A: Full JD fetch failed, falling back to chunks: {e}")
        else:
            jd_context = chunk_hits
            reason = "scoring cache hit" if has_scoring_cache else "no position IDs"
            print(f"[Retriever] Mode B ({reason}): using {len(jd_context)} section chunks")

        # Belt-and-suspenders: enforce active-position guard on Python side.
        if intent == "jd_search" and state.get("active_position_ids"):
            active_ids = set(state["active_position_ids"])
            jd_context = [
                jd for jd in jd_context
                if jd.get("payload", {}).get("positionId") in active_ids
            ]

    state["cv_context"]      = cv_context
    state["jd_context"]      = jd_context
    state["retrieval_stats"] = result.get("retrieval_stats", {})
    return state
