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

import asyncio
from typing import Dict, List, Any, Optional, Tuple

from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from app.rag.hr.state import HRChatState
from app.rag.hr.router import _extract_top_n
from app.rag.shared.hybrid_retrieval import hybrid_retrieve_cv
from app.services.retriever import retriever, get_chunk_text
from app.config import get_settings
from app.rag.hr.helpers.cv_assembler import assemble_virtual_full_cv

settings = get_settings()

_JD_RANKING_CACHE: Dict[int, Tuple[List[float], str, str]] = {}

_REQUIREMENT_SECTION_MARKERS = (
    "REQUIRE", "QUALIF", "SKILL", "RESPONSIBIL",
    "MUST_HAVE", "LOOKING_FOR", "WHAT_YOU", "COMPETEN",
)

_RERANK_QUERY_MAX_WORDS = 128

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
    query: str,
) -> List[Dict[str, Any]]:
    """Fetch top JD chunks for the given position.

    Embeds the query internally so this coroutine is self-contained and can
    run concurrently with hybrid_retrieve_cv via asyncio.gather.
    """
    query_vector = await retriever._embed_async(query)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: retriever.qdrant_service.search_similar(
            collection_name=retriever.jd_collection,
            query_vector=query_vector,
            limit=3,
            score_threshold=0.0,
            filters=Filter(must=[
                FieldCondition(key="positionId", match=MatchValue(value=position_id))
            ]),
        ),
    )


# ---------------------------------------------------------------------------
# JD ranking vector builder (INTERNAL mode only)
# ---------------------------------------------------------------------------

async def _get_jd_ranking_vector(position_id: int) -> Tuple[Optional[List[float]], str, str]:
    """
    Build and cache the JD ranking signals for INTERNAL CV ranking.

    Fetches every JD chunk via Qdrant scroll (no query vector), orders them by
    chunkIndex, then derives three artifacts (cached per position_id for the
    process lifetime — JDs are immutable after upload):

      - ranking_vector: dense vector embedded from the REQUIREMENTS section(s) only
        (fuzzy-matched by section name). Falls back to the full JD when no
        requirements-like section exists. Ranking against the focused requirements
        text — rather than the whole JD (intro/benefits dilute the average, and the
        embedder truncates at 512 tokens) — gives a far more discriminative signal.
      - rerank_query: first _RERANK_QUERY_MAX_WORDS words of that same requirements
        text, used as the cross-encoder query instead of HR's conversational string.
      - full_jd_text: the entire JD (all sections, ordered) — consumed by the scoring
        node for holistic CV evaluation.

    Returns (None, "", "") if no JD chunks exist for the position.
    """
    if position_id in _JD_RANKING_CACHE:
        print(f"[JD Vector] Cache hit for position_id={position_id}")
        return _JD_RANKING_CACHE[position_id]

    loop = asyncio.get_running_loop()
    results, _ = await loop.run_in_executor(
        None,
        lambda: retriever.qdrant_service.get_client().scroll(
            collection_name=retriever.jd_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="positionId", match=MatchValue(value=position_id))
            ]),
            limit=50,
            with_payload=True,
            with_vectors=False,
        ),
    )

    if not results:
        print(f"[JD Vector] No JD chunks for position_id={position_id} — dense search will fall back to query embedding")
        return None, "", ""

    # Order by document position — scroll order is by point id (non-deterministic
    # relative to JD layout), so without this the "full text" and the truncated
    # embedding would represent an arbitrary slice of the JD.
    ordered = sorted(results, key=lambda r: r.payload.get("chunkIndex", 0))

    full_jd_text = " ".join(
        t for t in (get_chunk_text(r.payload).strip() for r in ordered) if t
    )

    # Select requirements-like sections for the ranking signal.
    req_chunks = [
        r for r in ordered
        if any(marker in (r.payload.get("sectionName") or "").upper()
               for marker in _REQUIREMENT_SECTION_MARKERS)
    ]
    req_text = " ".join(
        t for t in (get_chunk_text(r.payload).strip() for r in req_chunks) if t
    ).strip()

    # Fallback: no requirements-like section found → rank against the whole JD.
    ranking_text = req_text or full_jd_text
    used_sections = (
        sorted({(r.payload.get("sectionName") or "?") for r in req_chunks})
        if req_text else ["<full JD fallback>"]
    )

    rerank_query = " ".join(ranking_text.split()[:_RERANK_QUERY_MAX_WORDS])
    ranking_vector = await retriever._embed_async(ranking_text)

    _JD_RANKING_CACHE[position_id] = (ranking_vector, rerank_query, full_jd_text)
    print(
        f"[JD Vector] Built and cached for position_id={position_id} "
        f"({len(results)} chunks, ranking_text={len(ranking_text.split())} words, "
        f"sections={used_sections})"
    )
    return ranking_vector, rerank_query, full_jd_text


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
    skill_keywords = entities.get("skill_keywords") or []
    skill_logic    = entities.get("skill_logic", "OR")

    # FIND_MORE passes currently active CV IDs as exclusion list
    exclude_ids: List[int] = active_cv_ids if strategy == "FIND_MORE" else []

    print(
        f"[HR Retrieve] strategy={strategy}, mode={state['mode']}, top_n={top_n}"
        f" | expanded='{expanded_query[:60]}...' | variants={len(skill_variants)}"
        f" | exclude={len(exclude_ids)} id(s)"
    )

    # Build base filters per mode
    if state["mode"] == "INTERNAL":
        base_filters = _build_hr_base_filters(position_id)
    else:
        base_filters = _build_candidate_base_filters(position_id)

    # EXTERNAL mode skips the cross-encoder: rrf_score is sufficient as query-relevance
    # tiebreaker since avg_score (pre-computed by candidate chatbot) is the primary signal.
    is_external = state["mode"] == "EXTERNAL"

    # ---- For INTERNAL mode: resolve JD ranking vector ----
    # EXTERNAL mode uses pre-computed avg_score as primary signal — JD vector not needed.
    # For INTERNAL mode, ranking CVs against the JD vector
    # is the correct semantic approach. Cache hit is instant after the first query.
    jd_vector: Optional[List[float]] = None
    jd_rerank_query: Optional[str] = None
    if not is_external:
        jd_vector, jd_rerank_query, full_jd_text = await _get_jd_ranking_vector(position_id)
        # Persist the full JD so the scoring node evaluates each CV against the
        # complete JD instead of the 3 query-retrieved chunks in jd_context.
        state["full_jd_text"] = full_jd_text

    # ---- HYBRID RETRIEVAL + JD fetch — run concurrently ----
    # hybrid_retrieve_cv uses expanded_query as fallback when jd_vector is None.
    # _fetch_jd_context uses the original query for prompt-relevance (not expansion).
    cv_results, jd_results = await asyncio.gather(
        hybrid_retrieve_cv(
            query=expanded_query,
            skill_variants=skill_variants,
            base_filters=base_filters,
            top_n=top_n,
            exclude_cv_ids=exclude_ids if exclude_ids else None,
            skill_keywords=skill_keywords,
            skill_logic=skill_logic,
            skip_rerank=is_external,
            query_vector=jd_vector,
            rerank_query=jd_rerank_query if jd_rerank_query else None,
        ),
        _fetch_jd_context(position_id, query),
    )

    # EXTERNAL dual-sort: primary = avg_score (JD fit, static), tiebreaker = rrf_score (query relevance)
    if is_external:
        cv_id_to_meta = state.get("cv_id_to_meta", {})
        def _external_sort_key(chunk: Dict[str, Any]) -> tuple:
            cv_id = chunk.get("payload", {}).get("cvId")
            meta  = cv_id_to_meta.get(cv_id, {})
            avg   = (meta.get("score") or 0)
            rrf   = chunk.get("rrf_score", 0.0)
            return (avg, rrf)
        cv_results = sorted(cv_results, key=_external_sort_key, reverse=True)

    state["cv_context"] = cv_results
    state["jd_context"] = jd_results
    state["retrieval_stats"] = {
        "strategy":               strategy,
        "cv_unique_ids_returned": len({c.get("payload", {}).get("cvId") for c in cv_results}),
        "jd_chunks_retrieved":    len(jd_results),
        "top_n_requested":        top_n,
        "skill_variants_used":    skill_variants,
        "excluded_cv_ids":        exclude_ids,
        "jd_vector_used":         jd_vector is not None,
    }

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
        cv_id_to_meta = state.get("cv_id_to_meta", {})
        state["ranked_cv_list"] = [
            {"rank": i + 1, "cvId": cid, "name": cv_id_to_meta.get(cid, {}).get("candidateName", f"CV-{cid}")}
            for i, cid in enumerate(deduped_ids)
        ]
        print(f"[HR Retrieve] active_cv_ids set to top-{requested_n}: {deduped_ids}")
    else:
        new_ids = [
            chunk.get("payload", {}).get("cvId")
            for chunk in cv_results
            if chunk.get("payload", {}).get("cvId") is not None
        ]
        merged = list(dict.fromkeys(active_cv_ids + new_ids))
        state["active_cv_ids"] = merged
        print(f"[HR Retrieve] FIND_MORE merged active_cv_ids={merged}")

    return state