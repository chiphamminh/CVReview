"""
Hybrid Retrieval Module — Dense + Keyword search merged via RRF.

Architecture (per implement_plan.md §5):
  expanded_query
        │
        ├───────────────────────────────────────────┐
        ▼                                           ▼
  Dense Search (async)                   Keyword Search (async)
  Qdrant cosine similarity               Qdrant MatchAny on `skills` field
  embed(expanded_query)                  filter: skills MatchAny skill_variants
  limit = top_n * 4                      limit = top_n * 3
        │                                           │
        └─────────────────┬─────────────────────────┘
                          ▼
                   RRF Merge
                   weight: 0.6 dense + 0.4 keyword
                          │
                          ▼
              Cross-encoder Rerank (group by cvId, Max Score)
              max_chunks_per_id = 6
                   → top_n results

RRF formula:  RRF_score(doc) = Σ weight_i / (60 + rank_i)
No score normalization between systems — this is why RRF is chosen over weighted sum.
"""

import asyncio
from typing import List, Dict, Any, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from app.services.embedding import embedding_service
from app.services.qdrant import qdrant_service
from app.services.reranker import reranker
from app.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RRF_K = 60                     # Standard RRF constant — smooths rank differences
_DENSE_WEIGHT = 0.6             # Dense search contribution to RRF score
_KEYWORD_WEIGHT = 0.4           # Keyword search contribution to RRF score
_DENSE_FETCH_MULTIPLIER = 4     # dense limit = top_n * 4
_KEYWORD_FETCH_MULTIPLIER = 3   # keyword limit = top_n * 3
_MAX_CHUNKS_PER_CV = 6          # Cap chunks per cvId before passing to reranker


# ---------------------------------------------------------------------------
# Individual search functions
# ---------------------------------------------------------------------------

async def _dense_search(
    query_text: str,
    collection: str,
    base_filters: List,
    limit: int,
    score_threshold: float = 0.25,
    must_not_conditions: Optional[List] = None,
) -> List[Dict[str, Any]]:
    """
    Dense vector search using the expanded query.
    embedding_service.embed_text is CPU-bound — run in executor to avoid blocking.
    """
    loop = asyncio.get_event_loop()
    query_vector = await loop.run_in_executor(
        None, lambda: embedding_service.embed_text(query_text, is_query=True)
    )

    qdrant_filter: Optional[Filter] = None
    if base_filters or must_not_conditions:
        qdrant_filter = Filter(
            must=base_filters or [],
            must_not=must_not_conditions or [],
        )

    results = qdrant_service.search_similar(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        filters=qdrant_filter,
    )
    return results


async def _keyword_search(
    skill_variants: List[str],
    collection: str,
    base_filters: List,
    limit: int,
    must_not_conditions: Optional[List] = None,
) -> List[Dict[str, Any]]:
    """
    Keyword search on the `skills` metadata field via Qdrant MatchAny.
    Uses a zero vector (dummy) so only the filter matters — no semantic ranking here.
    score_threshold=0.0 accepts all keyword matches regardless of cosine score.
    """
    if not skill_variants:
        return []

    dim = settings.EMBEDDING_DIMENSION
    dummy_vector = [0.0] * dim

    keyword_filters = list(base_filters) + [
        FieldCondition(key="skills", match=MatchAny(any=skill_variants))
    ]

    results = qdrant_service.search_similar(
        collection_name=collection,
        query_vector=dummy_vector,
        limit=limit,
        score_threshold=0.0,
        filters=Filter(
            must=keyword_filters,
            must_not=must_not_conditions or [],
        ),
    )
    return results


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

def _rrf_merge(
    dense_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.

    RRF_score(doc) = Σ weight_i / (K + rank_i)
    where rank_i is 1-indexed position in each list.

    Returns documents sorted by descending RRF score.
    Document identity is determined by Qdrant point `id`.
    """
    scores: Dict[Any, float] = {}
    doc_map: Dict[Any, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _DENSE_WEIGHT / (_RRF_K + rank)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(keyword_results, start=1):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _KEYWORD_WEIGHT / (_RRF_K + rank)
        doc_map.setdefault(doc_id, doc)

    # Attach computed RRF score and sort descending
    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    result = []
    for doc_id in ranked:
        doc = dict(doc_map[doc_id])
        doc["rrf_score"] = round(scores[doc_id], 6)
        result.append(doc)

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def hybrid_retrieve_cv(
    query: str,
    skill_variants: List[str],
    base_filters: List,
    top_n: int,
    score_threshold: float = 0.25,
    exclude_cv_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval for CV collection: Dense + Keyword → RRF → Cross-encoder rerank.

    Args:
        query:           The expanded query string (output of expansion.py).
        skill_variants:  Skill synonyms for keyword filter (output of expansion.py).
        base_filters:    Qdrant FieldCondition list already built by the caller
                         (e.g. positionId, sourceType, is_latest filters).
        top_n:           Number of unique CVs to return after reranking.
        score_threshold: Minimum cosine similarity for dense leg.
        exclude_cv_ids:  Optional list of cvIds to exclude (FIND_MORE strategy).

    Returns:
        List of reranked CV chunks (one per unique cvId, highest-scoring chunk).
    """
    collection    = settings.CV_COLLECTION_NAME
    dense_limit   = top_n * _DENSE_FETCH_MULTIPLIER
    keyword_limit = top_n * _KEYWORD_FETCH_MULTIPLIER

    # Build the must_not exclusion list once — passed to both search legs.
    # Using Filter(must=..., must_not=...) so Qdrant handles exclusion at index level
    # rather than filtering in Python, which would waste fetched quota.
    must_not_conditions: List = (
        [FieldCondition(key="cvId", match=MatchAny(any=exclude_cv_ids))]
        if exclude_cv_ids
        else []
    )

    # Run dense and keyword legs in parallel
    dense_task   = _dense_search(query, collection, base_filters, dense_limit, score_threshold, must_not_conditions)
    keyword_task = _keyword_search(skill_variants, collection, base_filters, keyword_limit, must_not_conditions)

    dense_results, keyword_results = await asyncio.gather(dense_task, keyword_task)

    print(
        f"[HybridRetrieve] dense={len(dense_results)}, keyword={len(keyword_results)}"
        f" | top_n={top_n}, skills={skill_variants[:3]}..."
    )

    if not dense_results and not keyword_results:
        return []

    # RRF merge
    merged = _rrf_merge(dense_results, keyword_results)

    # Cap chunks per cvId before reranking to avoid prompt bloat
    cv_chunk_count: Dict[Any, int] = {}
    capped: List[Dict[str, Any]] = []
    for doc in merged:
        cv_id = doc.get("payload", {}).get("cvId")
        if cv_id is None:
            continue
        if cv_chunk_count.get(cv_id, 0) < _MAX_CHUNKS_PER_CV:
            capped.append(doc)
            cv_chunk_count[cv_id] = cv_chunk_count.get(cv_id, 0) + 1

    # Cross-encoder rerank — group by cvId, keep top_n unique CVs
    reranked = reranker.rerank_and_group(
        query=query,
        chunks=capped,
        id_field="cvId",
        top_n=top_n,
    )

    print(f"[HybridRetrieve] After rerank: {len(reranked)} chunks from unique CVs")
    return reranked