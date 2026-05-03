from app.rag.hr.router import _extract_top_n
from typing import List, Dict, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from app.rag.hr.state import HRChatState
from app.services.retriever import retriever
from app.services.embedding import embedding_service
from app.config import get_settings
from app.rag.shared.hybrid_retrieval import hybrid_retrieve_cv

_SECTION_ORDER = ["SUMMARY", "EXPERIENCE", "SKILLS", "EDUCATION", "PROJECTS"]

settings = get_settings()

def normalize_section(section: str) -> str:
    s = section.upper()

    if s == "PROJECTS" or s.startswith("PROJECT_"):
        return "PROJECTS"

    return s


async def _fetch_pinned_cv_context(
    cv_ids: List[int],
    qdrant_svc,
    cv_collection: str,
) -> List[Dict[str, Any]]:
    """
    Pinned fetch: retrieve all chunks for given cvIds directly by ID filter.
    No reranking. Sections are assembled in canonical order (virtual full CV).
    Used for COMPARE / DETAIL pipeline strategies.
    """
    MAX_CHUNKS_PER_CV = 12  # Higher cap for COMPARE so LLM has enough content

    results = qdrant_svc.search_similar(
        collection_name=cv_collection,
        query_vector=[0.0] * settings.EMBEDDING_DIMENSION,
        limit=len(cv_ids) * MAX_CHUNKS_PER_CV,
        score_threshold=0.0,
        filters=Filter(must=[
            FieldCondition(key="cvId", match=MatchAny(any=cv_ids)),
            FieldCondition(key="is_latest", match=MatchValue(value=True)),
        ]),
    )

    # Group by cvId
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for chunk in results:
        cv_id = chunk.get("payload", {}).get("cvId")
        if cv_id is not None:
            grouped.setdefault(cv_id, []).append(chunk)

    # Assemble in canonical section order (virtual full CV)
    pinned: List[Dict[str, Any]] = []
    for cv_id in cv_ids:
        raw_chunks = grouped.get(cv_id, [])
        ordered = sorted(
            raw_chunks,
            key=lambda c: _SECTION_ORDER.index(
                normalize_section(
                    c.get("payload", {}).get("section", "")
                )
            )
            if normalize_section(
                c.get("payload", {}).get("section", "")
            ) in _SECTION_ORDER
            else 99,
        )
        pinned.extend(ordered[:MAX_CHUNKS_PER_CV])

    return pinned


def _build_hr_base_filters(position_id: int, source_type: str) -> List:
    """Build the common Qdrant must-filter list for HR CV retrieval."""
    return [
        FieldCondition(key="positionId", match=MatchValue(value=position_id)),
        FieldCondition(key="sourceType", match=MatchValue(value=source_type)),
        FieldCondition(key="is_latest", match=MatchValue(value=True)),
    ]


def _build_candidate_base_filters(position_id: int) -> List:
    """Build the common Qdrant must-filter list for Candidate CV retrieval."""
    return [
        FieldCondition(key="applied_position_ids", match=MatchAny(any=[position_id])),
        FieldCondition(key="sourceType", match=MatchValue(value="CANDIDATE")),
        FieldCondition(key="is_latest", match=MatchValue(value=True)),
    ]


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


async def retrieve_hr_context_node(state: HRChatState) -> HRChatState:
    """
    Routes to the correct retrieval strategy based on pipeline_strategy set by the router.

    ACTION    → bypass Qdrant entirely (cv_context=[], jd_context=[])
    AGGREGATE → bypass Qdrant entirely (data comes from SQL statistics API)
    COMPARE   → pinned fetch from active_cv_ids (no rerank, full sections)
    DETAIL    → pinned fetch from active_cv_ids (no rerank, full sections)
    RANK      → hybrid retrieval (dense + keyword → RRF → rerank)
    FILTER    → hybrid retrieval (dense + keyword → RRF → rerank)
    FIND_MORE → hybrid retrieval, exclude active_cv_ids via Qdrant must_not
    """
    strategy      = state.get("pipeline_strategy", "RANK")
    query         = state["query"]
    active_cv_ids = state.get("active_cv_ids") or []

    # --- Strategies that bypass Qdrant entirely ---
    if strategy in ("ACTION", "AGGREGATE"):
        print(f"[HR Retrieve] strategy={strategy} → Bypass Qdrant")
        state["cv_context"]      = []
        state["jd_context"]      = []
        state["retrieval_stats"] = {"strategy": strategy, "note": "qdrant_bypassed"}
        return state

    # --- Pinned fetch for COMPARE / DETAIL ---
    if strategy in ("COMPARE", "DETAIL") and not active_cv_ids:
        state["cv_context"]      = []
        state["jd_context"]      = []
        state["retrieval_stats"] = {"strategy": strategy, "note": "no_active_cv_ids"}
        state["llm_response"]    = (
            "Không có ứng viên nào đang được theo dõi trong phiên này. "
            "Vui lòng tìm kiếm ứng viên trước khi thực hiện so sánh hoặc xem chi tiết."
        )
        print(f"[HR Retrieve] {strategy} requested but active_cv_ids is empty — early return")
        return state

    if strategy in ("COMPARE", "DETAIL") and active_cv_ids:
        print(f"[HR Retrieve] strategy={strategy} → Pinned fetch for {len(active_cv_ids)} CV(s)")
        cv_chunks = await _fetch_pinned_cv_context(
            cv_ids=active_cv_ids,
            qdrant_svc=retriever.qdrant_service,
            cv_collection=retriever.cv_collection,
        )
        state["cv_context"]      = cv_chunks
        state["jd_context"]      = []   # JD not needed for compare/detail — saves tokens
        state["retrieval_stats"] = {
            "strategy": "pinned_cv_fetch",
            "cv_ids": active_cv_ids,
            "chunks_fetched": len(cv_chunks),
        }
        return state

    # --- Hybrid retrieval for RANK / FILTER / FIND_MORE ---
    entities       = state.get("query_entities", {})
    top_n          = entities.get("top_n") or _extract_top_n(query)
    expanded_query = state.get("expanded_query") or query
    skill_variants = state.get("skill_variants") or []
    position_id    = state["position_id"]

    # FIND_MORE passes the currently active CV IDs as exclusion list
    exclude_ids: List[int] = active_cv_ids if strategy == "FIND_MORE" else []

    print(
        f"[HR Retrieve] strategy={strategy}, mode={state['mode']}, top_n={top_n}"
        f" | expanded='{expanded_query[:60]}...' | variants={len(skill_variants)}"
        f" | exclude={len(exclude_ids)} id(s)"
    )

    if state["mode"] == "HR_MODE":
        base_filters = _build_hr_base_filters(position_id, source_type="HR")
    else:
        base_filters = _build_candidate_base_filters(position_id)

    cv_results = await hybrid_retrieve_cv(
        query=expanded_query,
        skill_variants=skill_variants,
        base_filters=base_filters,
        top_n=top_n,
        exclude_cv_ids=exclude_ids if exclude_ids else None,
    )

    # Fetch JD context using the original (unexpanded) query vector for prompt relevance
    query_vector = embedding_service.embed_text(query, is_query=True)
    jd_results   = await _fetch_jd_context(position_id, query_vector)

    state["cv_context"]      = cv_results
    state["jd_context"]      = jd_results
    state["retrieval_stats"] = {
        "strategy":               strategy,
        "cv_unique_ids_returned": len({c.get("payload", {}).get("cvId") for c in cv_results}),
        "jd_chunks_retrieved":    len(jd_results),
        "top_n_requested":        top_n,
        "skill_variants_used":    skill_variants,
        "excluded_cv_ids":        exclude_ids,
    }

    # Persist active_cv_ids for follow-up COMPARE / DETAIL turns.
    # FIND_MORE intentionally keeps the OLD active_cv_ids so the next COMPARE
    # still covers all candidates seen so far (both original + newly found).
    if strategy != "FIND_MORE":
        new_active_ids = list({
            chunk.get("payload", {}).get("cvId")
            for chunk in cv_results
            if chunk.get("payload", {}).get("cvId") is not None
        })
        state["active_cv_ids"] = new_active_ids
        print(f"[HR Retrieve] Updated active_cv_ids={new_active_ids}")

    return state