"""
Node: Context Retrieval for HR Chatbot.

Routes to the correct retrieval strategy based on pipeline_strategy set by
route_hr_intent_node. Each strategy fetches a different mix of CV / JD context
to minimise Qdrant round-trips and LLM token spend.

Strategy routing:
  ACTION    → bypass Qdrant entirely (email/confirm flow)
  AGGREGATE → bypass Qdrant entirely (SQL statistics)
  COMPARE   → pinned fetch from active_cv_ids via Qdrant scroll API (no rerank)
  DETAIL    → pinned fetch from active_cv_ids via Qdrant scroll API (no rerank)
  RANK      → hybrid retrieval (dense + keyword → RRF → cross-encoder rerank)
  FILTER    → hybrid retrieval (dense + keyword → RRF → cross-encoder rerank)
  FIND_MORE → hybrid retrieval, exclude active_cv_ids via Qdrant must_not
"""

from typing import List, Dict, Any, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from app.rag.hr.state import HRChatState
from app.rag.hr.router import _extract_top_n
from app.rag.shared.hybrid_retrieval import hybrid_retrieve_cv
from app.services.retriever import retriever
from app.services.embedding import embedding_service
from app.config import get_settings
from app.rag.hr.helpers.cv_assembler import assemble_virtual_full_cv

settings = get_settings()

# ---------------------------------------------------------------------------
# Pinned fetch — COMPARE / DETAIL
# ---------------------------------------------------------------------------

async def _fetch_pinned_cv_context(
    cv_ids: List[int],
    qdrant_svc,
    cv_collection: str,
) -> List[Dict[str, Any]]:
    """
    Pinned fetch: retrieve all chunks for given cvIds using Qdrant scroll API.

    BUG #2 FIX: The previous implementation called search_similar() with a
    zero-vector of hardcoded length 1024, while the collection was indexed at
    EMBEDDING_DIMENSION (e.g. 384 for BGE-small). This caused a Qdrant dimension
    mismatch exception and crashed every COMPARE/DETAIL turn.

    scroll() is the correct API for "fetch by filter without semantic ranking" —
    it does not require a query vector at all, avoids dimension coupling, and
    does not compute cosine similarity (saving Qdrant CPU).

    Sections are assembled in canonical order to form a "virtual full CV"
    per plan §6 ("Virtual Full CV für COMPARE/DETAIL").
    """
    MAX_CHUNKS_PER_CV = 12  # Higher cap than RANK (6) — COMPARE needs full sections

    results, _ = qdrant_svc.get_client().scroll(
        collection_name=cv_collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="cvId", match=MatchAny(any=cv_ids)),
            FieldCondition(key="is_latest", match=MatchValue(value=True)),
        ]),
        limit=len(cv_ids) * MAX_CHUNKS_PER_CV,
        with_payload=True,
        with_vectors=False,  # no vector needed — saves bandwidth
    )

    raw_chunks = [{"id": r.id, "score": 1.0, "payload": r.payload} for r in results]

    return assemble_virtual_full_cv(
        raw_chunks=raw_chunks,
        cv_ids=cv_ids,
        max_chunks_per_cv=MAX_CHUNKS_PER_CV
    )


# ---------------------------------------------------------------------------
# Base filter builders
# ---------------------------------------------------------------------------

def _build_hr_base_filters(position_id: int) -> List:
    """Must-filters for HR-uploaded CVs (HR_MODE)."""
    return [
        FieldCondition(key="positionId",  match=MatchValue(value=position_id)),
        FieldCondition(key="sourceType",  match=MatchValue(value="INTERNAL")),
        FieldCondition(key="is_latest",   match=MatchValue(value=True)),
    ]


def _build_candidate_base_filters(position_id: int) -> List:
    """Must-filters for candidate-applied CVs (CANDIDATE_MODE)."""
    return [
        FieldCondition(key="applied_position_ids", match=MatchAny(any=[position_id])),
        FieldCondition(key="sourceType",           match=MatchValue(value="EXTERNAL")),
        FieldCondition(key="is_latest",            match=MatchValue(value=True)),
    ]


# ---------------------------------------------------------------------------
# JD context fetch (shared between RANK / FILTER / FIND_MORE)
# ---------------------------------------------------------------------------

async def _fetch_jd_context(
    position_id: int,
    query_vector: List[float],
) -> List[Dict[str, Any]]:
    """Fetch top JD chunks for the given position to inject into prompt context."""
    return retriever.qdrant_service.search_similar(
        collection_name=retriever.jd_collection,
        query_vector=query_vector,
        limit=3,
        score_threshold=0.0,
        filters=Filter(must=[
            FieldCondition(key="positionId", match=MatchValue(value=position_id))
        ]),
    )


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

async def retrieve_hr_context_node(state: HRChatState) -> HRChatState:
    """
    Dispatch to the correct retrieval sub-strategy based on pipeline_strategy.

    RANK, FILTER, FIND_MORE now route through hybrid_retrieve_cv() (dense + keyword RRF)
    instead of the legacy dense-only retriever.
    expanded_query and skill_variants written by query_expansion_node are consumed here.
    """
    strategy      = state.get("pipeline_strategy", "RANK")
    active_cv_ids = state.get("active_cv_ids") or []

    # --- Strategies that bypass Qdrant entirely ---
    if strategy in ("ACTION", "AGGREGATE"):
        print(f"[HR Retrieve] strategy={strategy} → Bypass Qdrant")
        state["cv_context"]      = []
        state["jd_context"]      = []
        state["retrieval_stats"] = {"strategy": strategy, "note": "qdrant_bypassed"}
        return state

    # --- Pinned fetch for COMPARE / DETAIL ---
    if strategy in ("COMPARE", "DETAIL"):
        if not active_cv_ids:
            state["cv_context"]      = []
            state["jd_context"]      = []
            state["retrieval_stats"] = {"strategy": strategy, "note": "no_active_cv_ids"}
            state["llm_response"]    = (
                "Không có ứng viên nào đang được theo dõi trong phiên này. "
                "Vui lòng tìm kiếm ứng viên trước khi thực hiện so sánh hoặc xem chi tiết."
            )
            print(f"[HR Retrieve] {strategy} requested but active_cv_ids is empty — early return")
            return state

        print(f"[HR Retrieve] strategy={strategy} → Pinned scroll fetch for {len(active_cv_ids)} CV(s)")
        cv_chunks = await _fetch_pinned_cv_context(
            cv_ids=active_cv_ids,
            qdrant_svc=retriever.qdrant_service,
            cv_collection=retriever.cv_collection,
        )
        state["cv_context"]      = cv_chunks
        state["jd_context"]      = []  # JD not needed for compare/detail — saves tokens
        state["retrieval_stats"] = {
            "strategy":      "pinned_scroll_fetch",
            "cv_ids":        active_cv_ids,
            "chunks_fetched": len(cv_chunks),
        }
        return state

    # --- Hybrid retrieval for RANK / FILTER / FIND_MORE ---
    query          = state["query"]
    entities       = state.get("query_entities", {})
    top_n          = entities.get("top_n") or _extract_top_n(query)
    position_id    = state["position_id"]

    # Consume expanded_query + skill_variants written by query_expansion_node.
    # If expansion was skipped (passthrough), these fall back to the original values.
    expanded_query = state.get("expanded_query") or query
    skill_variants = state.get("skill_variants") or []

    # FIND_MORE passes currently active CV IDs as exclusion list
    exclude_ids: List[int] = active_cv_ids if strategy == "FIND_MORE" else []

    print(
        f"[HR Retrieve] strategy={strategy}, mode={state['mode']}, top_n={top_n}"
        f" | expanded='{expanded_query[:60]}...' | variants={len(skill_variants)}"
        f" | exclude={len(exclude_ids)} id(s)"
    )

    # Build base filters per mode
    if state["mode"] == "HR_MODE":
        base_filters = _build_hr_base_filters(position_id)
    else:
        base_filters = _build_candidate_base_filters(position_id)

    # ---- HYBRID RETRIEVAL (dense + keyword → RRF → cross-encoder rerank) ----
    cv_results = await hybrid_retrieve_cv(
        query=expanded_query,
        skill_variants=skill_variants,
        base_filters=base_filters,
        top_n=top_n,
        exclude_cv_ids=exclude_ids if exclude_ids else None,
    )

    # Fetch JD context using the ORIGINAL query vector (not expanded) for prompt relevance.
    # Expansion is tuned for CV recall; original query stays closer to JD section topics.
    query_vector = embedding_service.embed_text(query, is_query=True)
    jd_results   = await _fetch_jd_context(position_id, query_vector)

    state["cv_context"] = cv_results
    state["jd_context"] = jd_results
    state["retrieval_stats"] = {
        "strategy":               strategy,
        "cv_unique_ids_returned": len({c.get("payload", {}).get("cvId") for c in cv_results}),
        "jd_chunks_retrieved":    len(jd_results),
        "top_n_requested":        top_n,
        "skill_variants_used":    skill_variants,
        "excluded_cv_ids":        exclude_ids,
    }

    # Persist active_cv_ids for follow-up COMPARE / DETAIL turns.
    if strategy != "FIND_MORE":
        requested_n = entities.get("top_n") or _extract_top_n(query)
        seen: set = set()
        deduped_ids: List[int] = []
        for chunk in cv_results:
            cv_id = chunk.get("payload", {}).get("cvId")
            if cv_id is not None and cv_id not in seen:
                seen.add(cv_id)
                deduped_ids.append(cv_id)
                if len(deduped_ids) == requested_n:
                    break
        
        state["active_cv_ids"] = deduped_ids
        print(f"[HR Retrieve] active_cv_ids set to top-{requested_n}: {deduped_ids}")
    else:
        new_ids = [
            chunk.get("payload", {}).get("cvId")
            for chunk in cv_results
            if chunk.get("payload", {}).get("cvId") is not None
        ]
        # Merge: old (shown before) + new (just found), deduped, order preserved
        merged = list(dict.fromkeys(active_cv_ids + new_ids))
        state["active_cv_ids"] = merged
        print(f"[HR Retrieve] FIND_MORE merged active_cv_ids={merged}")

    return state