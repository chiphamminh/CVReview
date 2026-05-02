from app.rag.hr.router import _extract_top_n
from typing import List, Dict, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from app.rag.hr.state import HRChatState
from app.services.retriever import retriever
from app.config import get_settings

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


async def retrieve_hr_context_node(state: HRChatState) -> HRChatState:
    """
    Routes to the correct retrieval strategy based on pipeline_strategy set by the router.

    ACTION    → bypass Qdrant entirely (cv_context=[], jd_context=[])
    AGGREGATE → bypass Qdrant entirely (data comes from SQL statistics API)
    COMPARE   → pinned fetch from active_cv_ids (no rerank, full sections)
    DETAIL    → pinned fetch from active_cv_ids (no rerank, full sections)
    RANK      → full dense retrieval via retriever
    FILTER    → full dense retrieval via retriever
    FIND_MORE → full dense retrieval, exclude active_cv_ids
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

    # --- Full retrieval for RANK / FILTER / FIND_MORE ---
    entities = state.get("query_entities", {})
    top_n = entities.get("top_n") or _extract_top_n(query)
    print(f"[HR Retrieve] strategy={strategy}, mode={state['mode']}, top_n={top_n}")

    if state["mode"] == "HR_MODE":
        result = await retriever.retrieve_for_hr_mode_hr(
            query=query,
            position_id=state["position_id"],
            top_n=top_n,
        )
    else:
        result = await retriever.retrieve_for_hr_mode_candidate(
            query=query,
            position_id=state["position_id"],
            top_n=top_n,
        )

    state["cv_context"]      = result.get("cv_context", [])
    state["jd_context"]      = result.get("jd_context", [])
    state["retrieval_stats"] = result.get("retrieval_stats", {})

    # Persist active_cv_ids for follow-up COMPARE / DETAIL / FIND_MORE turns
    if strategy != "FIND_MORE":
        new_active_ids = list({
            chunk.get("payload", {}).get("cvId")
            for chunk in state["cv_context"]
            if chunk.get("payload", {}).get("cvId") is not None
        })
        state["active_cv_ids"] = new_active_ids
        print(f"[HR Retrieve] Updated active_cv_ids={new_active_ids}")

    return state
