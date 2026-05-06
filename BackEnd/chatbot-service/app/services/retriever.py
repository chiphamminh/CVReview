from typing import List, Dict, Any, Optional, Literal
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from app.services.embedding import embedding_service
from app.services.qdrant import qdrant_service
from app.services.reranker import reranker
from app.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Retrieval fetch multiplier: fetch (N * RERANK_FETCH_MULTIPLIER) chunks from
# Qdrant so the Cross-Encoder has a representative pool to rerank.
# ---------------------------------------------------------------------------
_RERANK_FETCH_MULTIPLIER = 3
_RERANK_FETCH_MIN = 30


def get_chunk_text(payload: dict) -> str:
    """Safely read chunk text from a Qdrant payload.
    Handles both CV chunks ('chunkText') and JD chunks ('jdText' fallback to 'chunkText').
    """
    return (payload.get("jdText") or payload.get("chunkText") or "").strip()


def _rerank_fetch_limit(top_n: int) -> int:
    """Calculate how many chunks to pull from Qdrant before reranking."""
    return max(top_n * _RERANK_FETCH_MULTIPLIER, _RERANK_FETCH_MIN)


def get_adaptive_threshold(intent: str, has_specific_filter: bool) -> float:
    """
    Calculate adaptive similarity threshold.

    Lower when a specific ID filter is already applied (already narrowed scope),
    higher for broad searches where semantic quality matters more.
    """
    if has_specific_filter:
        return 0.30

    thresholds = {
        "cv_analysis": 0.40,
        "jd_search": 0.50,
        "jd_analysis": 0.35,
        "general": 0.35,
        "hr_candidate": 0.30,  # position_id filter is highly specific
    }
    return thresholds.get(intent, 0.45)


def get_adaptive_top_k(intent: str, query_length: int) -> tuple[int, int]:
    """
    Calculate adaptive top_k based on intent and query word count.

    Returns:
        (cv_top_k, jd_top_k) — number of *unique IDs* to return after reranking.
    """
    base_config = {
        "cv_analysis": (8, 0),
        "jd_search": (5, 5),
        "jd_analysis": (5, 2),
        "general": (3, 3),
    }
    cv_k, jd_k = base_config.get(intent, (5, 3))

    # Boost for complex queries (> 15 words)
    if query_length > 15:
        cv_k = min(cv_k + 2, 10)
        jd_k = min(jd_k + 1, 7)

    return cv_k, jd_k


class CareerCounselorRetriever:
    """
    Two-Stage retrieval layer: Qdrant Vector Search → Local Cross-Encoder Reranking.

    Phase 4 changes:
    - All public HR methods support dynamic top_n parsed from user query.
    - Reranking is applied at chunk level; results are grouped by cvId/positionId
      using Max Score before being trimmed to Top-N.
    - Candidate Chatbot uses the same rerank pattern but groups by positionId.
    """

    def __init__(self) -> None:
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.cv_collection = settings.CV_COLLECTION_NAME
        self.jd_collection = settings.JD_COLLECTION_NAME

    async def retrieve_for_intent(
        self,
        query: str,
        intent: Literal["jd_search", "jd_analysis", "cv_analysis", "general"],
        candidate_id: Optional[str] = None,
        cv_id: Optional[int] = None,
        jd_id: Optional[int] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        active_jd_ids: Optional[List[int]] = None,
        skill_variants: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Candidate Chatbot: intent-based retrieval with adaptive parameters."""
        query_vector = self.embedding_service.embed_text(query, is_query=True)
        query_length = len(query.split())

        if top_k is None:
            cv_top_k, jd_top_k = get_adaptive_top_k(intent, query_length)
        else:
            cv_top_k = jd_top_k = top_k

        print(f"[Retriever] Intent: {intent}, Query length: {query_length} words")
        print(f"[Retriever] Adaptive top_k: CV={cv_top_k}, JD={jd_top_k}")

        if intent == "jd_search":
            return await self._retrieve_jd_search_with_cv(
                query=query,
                query_vector=query_vector,
                candidate_id=candidate_id,
                cv_id=cv_id,
                cv_top_k=cv_top_k,
                jd_top_k=jd_top_k,
                score_threshold=score_threshold,
                active_jd_ids=active_jd_ids,
                skill_variants=skill_variants,
                intent=intent,
            )

        elif intent == "jd_analysis":
            if jd_id:
                return await self._retrieve_cv_jd_match(
                    query_vector=query_vector,
                    cv_id=cv_id,
                    jd_id=jd_id,
                    cv_top_k=cv_top_k,
                    jd_top_k=jd_top_k,
                    score_threshold=score_threshold,
                )
            else:
                return await self._retrieve_jd_search_with_cv(
                    query=query,
                    query_vector=query_vector,
                    candidate_id=candidate_id,
                    cv_id=cv_id,
                    cv_top_k=cv_top_k,
                    jd_top_k=min(jd_top_k, 3),
                    score_threshold=score_threshold,
                    active_jd_ids=active_jd_ids,
                    skill_variants=skill_variants,
                    intent=intent,
                )

        elif intent == "cv_analysis":
            return await self._retrieve_cv_analysis(
                query_vector=query_vector,
                candidate_id=candidate_id,
                cv_id=cv_id,
                cv_top_k=cv_top_k,
                score_threshold=score_threshold,
            )

        else:  # general
            return {"cv_context": [], "jd_context": [], "retrieval_stats": {}}

    async def _retrieve_jd_search_with_cv(
        self,
        query: str,
        query_vector: List[float],
        candidate_id: Optional[str] = None,
        cv_id: Optional[int] = None,
        cv_top_k: int = 5,
        jd_top_k: int = 5,
        score_threshold: Optional[float] = None,
        active_jd_ids: Optional[List[int]] = None,
        skill_variants: Optional[List[str]] = None,
        intent: str = "jd_search",
    ) -> Dict[str, Any]:
        """
        Candidate Chatbot — retrieve CV context and reranked JD context.

        JD reranking: Qdrant returns Top (jd_top_k * 3) JD chunks → Cross-Encoder
        scores each chunk → Group by positionId (Max Score) → Top jd_top_k positions.
        """
        has_cv_filter = cv_id is not None or candidate_id is not None
        if score_threshold is None:
            cv_threshold = get_adaptive_threshold(intent, has_cv_filter)
            jd_threshold = get_adaptive_threshold(intent, False)
        else:
            cv_threshold = jd_threshold = score_threshold

        print(f"[Retrieval] CV threshold: {cv_threshold:.2f}, JD threshold: {jd_threshold:.2f}")

        # --- CV filter (no reranking needed — already scoped to 1 candidate) ---
        cv_must = [FieldCondition(key="is_latest", match=MatchValue(value=True))]
        if cv_id:
            cv_must.append(FieldCondition(key="cvId", match=MatchValue(value=cv_id)))
        elif candidate_id:
            cv_must.append(FieldCondition(key="candidateId", match=MatchValue(value=candidate_id)))
        cv_results = self.qdrant_service.search_similar(
            collection_name=self.cv_collection,
            query_vector=query_vector,
            limit=cv_top_k,
            score_threshold=cv_threshold,
            filters=Filter(must=cv_must),
        )

        # --- JD filter: over-fetch then rerank → group by positionId ---
        jd_fetch_limit = _rerank_fetch_limit(jd_top_k)
        jd_must = []
        if active_jd_ids is not None:
            jd_must.append(FieldCondition(key="positionId", match=MatchAny(any=active_jd_ids)))
            
        jd_filter = Filter(must=jd_must) if jd_must else None

        jd_chunks = self.qdrant_service.search_similar(
            collection_name=self.jd_collection,
            query_vector=query_vector,
            limit=jd_fetch_limit,
            score_threshold=jd_threshold,
            filters=jd_filter,
        )

        jd_results = reranker.rerank_and_group(
            query=query,
            chunks=jd_chunks,
            id_field="positionId",
            top_n=jd_top_k,
        )

        cv_quality = self._assess_retrieval_quality(cv_results, "jd_search")
        jd_quality = self._assess_retrieval_quality(jd_results, "jd_search")
        print(f"[Quality] CV: {cv_quality['quality']} (max={cv_quality['max_score']:.2f})")
        print(f"[Quality] JD: {jd_quality['quality']} (max={jd_quality['max_score']:.2f})")

        return {
            "cv_context": cv_results,
            "jd_context": jd_results,
            "retrieval_stats": {
                "cv_chunks_retrieved": len(cv_results),
                "jd_positions_retrieved": len(jd_results),
                "jd_chunks_fetched": len(jd_chunks),
                "cv_score_range": self._get_score_range(cv_results),
                "jd_score_range": self._get_score_range(jd_results),
                "cv_quality": cv_quality,
                "jd_quality": jd_quality,
                "thresholds_used": {"cv_threshold": cv_threshold, "jd_threshold": jd_threshold},
            },
        }

    async def _retrieve_cv_jd_match(
        self,
        query_vector: List[float],
        cv_id: Optional[int],
        jd_id: Optional[int],
        cv_top_k: int = 5,
        jd_top_k: int = 2,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve specific CV chunks and JD chunks for a CV-JD analysis turn (no reranking needed — IDs are explicit)."""
        if not cv_id or not jd_id:
            print(f"[Warning] cv_jd_match called without cv_id or jd_id")
            return {"cv_context": [], "jd_context": [], "retrieval_stats": {"error": "Missing cv_id or jd_id"}}

        threshold = score_threshold if score_threshold is not None else get_adaptive_threshold("jd_analysis", True)
        print(f"[Retrieval] Using threshold: {threshold:.2f} (specific CV+JD match)")

        cv_results = self.qdrant_service.search_similar(
            collection_name=self.cv_collection,
            query_vector=query_vector,
            limit=cv_top_k,
            score_threshold=threshold,
            filters=Filter(must=[
                FieldCondition(key="cvId", match=MatchValue(value=cv_id)),
                FieldCondition(key="is_latest", match=MatchValue(value=True)),
            ]),
        )

        # JD chunks replaced atomically by positionId — no is_latest needed
        jd_results = self.qdrant_service.search_similar(
            collection_name=self.jd_collection,
            query_vector=query_vector,
            limit=jd_top_k,
            score_threshold=0.0,
            filters=Filter(must=[FieldCondition(key="positionId", match=MatchValue(value=jd_id))]),
        )

        cv_quality = self._assess_retrieval_quality(cv_results, "jd_analysis")
        print(f"[Quality] CV: {cv_quality['quality']} (max={cv_quality['max_score']:.2f})")
        print(f"[Quality] JD: Retrieved {len(jd_results)} specific JD chunks")

        return {
            "cv_context": cv_results,
            "jd_context": jd_results,
            "retrieval_stats": {
                "cv_chunks_retrieved": len(cv_results),
                "jd_retrieved": len(jd_results) > 0,
                "cv_score_range": self._get_score_range(cv_results),
                "jd_score_range": self._get_score_range(jd_results),
                "cv_quality": cv_quality,
                "threshold_used": threshold,
            },
        }

    async def _retrieve_cv_analysis(
        self,
        query_vector: List[float],
        candidate_id: Optional[str] = None,
        cv_id: Optional[int] = None,
        cv_top_k: int = 8,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve CV chunks for a pure CV analysis turn (no JD needed, no reranking — 1 candidate scope)."""
        has_filter = cv_id is not None or candidate_id is not None
        threshold = score_threshold if score_threshold is not None else get_adaptive_threshold("cv_analysis", has_filter)
        print(f"[Retrieval] CV analysis threshold: {threshold:.2f}")

        must_conditions = [FieldCondition(key="is_latest", match=MatchValue(value=True))]
        if cv_id:
            must_conditions.append(FieldCondition(key="cvId", match=MatchValue(value=cv_id)))
        elif candidate_id:
            must_conditions.append(FieldCondition(key="candidateId", match=MatchValue(value=candidate_id)))

        cv_results = self.qdrant_service.search_similar(
            collection_name=self.cv_collection,
            query_vector=query_vector,
            limit=cv_top_k,
            score_threshold=threshold,
            filters=Filter(must=must_conditions),
        )

        cv_quality = self._assess_retrieval_quality(cv_results, "cv_analysis")
        print(f"[Quality] CV: {cv_quality['quality']} (max={cv_quality['max_score']:.2f})")

        return {
            "cv_context": cv_results,
            "jd_context": [],
            "retrieval_stats": {
                "cv_chunks_retrieved": len(cv_results),
                "jd_docs_used": 0,
                "cv_score_range": self._get_score_range(cv_results),
                "cv_quality": cv_quality,
                "threshold_used": threshold,
            },
        }

    # ---------------------------------------------------------------------------
    # HR Chatbot — HR Mode
    # ---------------------------------------------------------------------------

    async def retrieve_for_hr_mode_hr(
        self,
        query: str,
        position_id: int,
        top_n: int = 10,
        score_threshold: float = 0.30,
    ) -> Dict[str, Any]:
        """
        HR Chatbot — HR Mode.

        Qdrant filter: positionId + sourceType=HR + is_latest=True.
        Over-fetches (top_n * multiplier) chunks then reranks at chunk level,
        grouping by cvId (Max Score) to select the most relevant Top-N CVs.

        Key design: CVs are ranked against the **Job Description** vector — NOT the
        HR's raw query string. This guarantees that skills aligned with the JD
        receive the highest scores rather than ranking by accidental textual
        similarity to conversational phrases like "tìm top 2 ứng viên".

        Args:
            query: HR's original question — used only for Reranker cross-encoding.
            top_n: Number of unique CVs to return. Parsed dynamically from HR's query
                   by the calling node (e.g. 'top 10' → top_n=10).
        """
        fetch_limit = _rerank_fetch_limit(top_n)
        print(f"[HR Mode] position_id={position_id}, top_n={top_n}, fetch_limit={fetch_limit}")

        # Step 1: Fetch JD chunks for this position to use as the ranking signal.
        # We embed the JD text so that CV chunks are ranked by skill-match, not by
        # how similar they are to the HR's conversational query.
        jd_chunks_for_vector = self.qdrant_service.search_similar(
            collection_name=self.jd_collection,
            query_vector=self.embedding_service.embed_text(query, is_query=True),
            limit=5,
            score_threshold=0.0,
            filters=Filter(must=[FieldCondition(key="positionId", match=MatchValue(value=position_id))]),
        )

        if jd_chunks_for_vector:
            # Concatenate JD chunk texts to build a rich ranking signal
            jd_text_for_search = " ".join(
                get_chunk_text(c.get("payload", {})) for c in jd_chunks_for_vector
            ).strip()
            ranking_vector = self.embedding_service.embed_text(jd_text_for_search, is_query=True)
            print(f"[HR Mode] Using JD-driven vector for CV ranking ({len(jd_chunks_for_vector)} JD chunks)")
        else:
            # Fallback: use HR query if no JD is indexed yet
            ranking_vector = self.embedding_service.embed_text(query, is_query=True)
            print("[HR Mode] No JD chunks found — falling back to query vector for CV ranking")

        # Step 2: Fetch CV chunks using the JD vector
        cv_chunks = self.qdrant_service.search_similar(
            collection_name=self.cv_collection,
            query_vector=ranking_vector,
            limit=fetch_limit,
            score_threshold=score_threshold,
            filters=Filter(must=[
                FieldCondition(key="positionId", match=MatchValue(value=position_id)),
                FieldCondition(key="sourceType", match=MatchValue(value="HR")),
                FieldCondition(key="is_latest", match=MatchValue(value=True)),
            ]),
        )

        # Step 3: Rerank with HR's actual query for conversational relevance
        cv_results = reranker.rerank_and_group(
            query=query,
            chunks=cv_chunks,
            id_field="cvId",
            top_n=top_n,
        )

        # JD context for the LLM prompt: always use query-based vector here
        jd_results = jd_chunks_for_vector[:3] if jd_chunks_for_vector else []

        return {
            "cv_context": cv_results,
            "jd_context": jd_results,
            "retrieval_stats": {
                "cv_chunks_fetched": len(cv_chunks),
                "cv_unique_ids_returned": len(cv_results),
                "jd_chunks_retrieved": len(jd_results),
                "cv_score_range": self._get_score_range(cv_results),
                "threshold_used": score_threshold,
                "position_id": position_id,
                "source_type": "HR",
                "ranking_strategy": "jd_driven",
                "top_n_requested": top_n,
            },
        }

    # ---------------------------------------------------------------------------
    # HR Chatbot — Candidate Mode
    # ---------------------------------------------------------------------------

    async def retrieve_for_hr_mode_candidate(
        self,
        query: str,
        position_id: int,
        top_n: int = 10,
        score_threshold: Optional[float] = None,
        active_jd_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        HR Chatbot — Candidate Mode.

        Qdrant filter: applied_position_ids contains position_id + sourceType=CANDIDATE + is_latest=True.
        Over-fetches chunks then reranks at chunk level, grouping by cvId (Max Score)
        to select the most relevant Top-N applied candidates.

        Args:
            top_n: Number of unique candidate CVs to return. Parsed dynamically from HR's query.
        """
        query_vector = self.embedding_service.embed_text(query, is_query=True)
        threshold = score_threshold if score_threshold is not None else get_adaptive_threshold("hr_candidate", True)
        fetch_limit = _rerank_fetch_limit(top_n)

        print(f"[Candidate Mode] position_id={position_id}, top_n={top_n}, fetch_limit={fetch_limit}, threshold={threshold:.2f}")

        cv_chunks = self.qdrant_service.search_similar(
            collection_name=self.cv_collection,
            query_vector=query_vector,
            limit=fetch_limit,
            score_threshold=threshold,
            filters=Filter(must=[
                FieldCondition(key="applied_position_ids", match=MatchAny(any=[position_id])),
                FieldCondition(key="sourceType", match=MatchValue(value="CANDIDATE")),
                FieldCondition(key="is_latest", match=MatchValue(value=True)),
            ]),
        )

        cv_results = reranker.rerank_and_group(
            query=query,
            chunks=cv_chunks,
            id_field="cvId",
            top_n=top_n,
        )

        print(f"[Candidate Mode] CV unique IDs after reranking: {len(cv_results)}")

        jd_ids_to_search = active_jd_ids if active_jd_ids else [position_id]
        jd_results = self.qdrant_service.search_similar(
            collection_name=self.jd_collection,
            query_vector=query_vector,
            limit=3,
            score_threshold=0.0,
            filters=Filter(must=[
                FieldCondition(key="positionId", match=MatchAny(any=jd_ids_to_search))
            ]),
        )

        print(f"[Candidate Mode] JD chunks found: {len(jd_results)}")

        return {
            "cv_context": cv_results,
            "jd_context": jd_results,
            "retrieval_stats": {
                "cv_chunks_fetched": len(cv_chunks),
                "cv_unique_ids_returned": len(cv_results),
                "jd_chunks_retrieved": len(jd_results),
                "cv_score_range": self._get_score_range(cv_results),
                "jd_score_range": self._get_score_range(jd_results),
                "threshold_used": threshold,
                "position_id": position_id,
                "source_type": "CANDIDATE",
                "filter_strategy": "applied_position_ids",
                "top_n_requested": top_n,
            },
        }

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    async def retrieve_cv_by_skills(
        self,
        required_skills: List[str],
        top_k: int = 5,
        seniority_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve CVs by skill keyword matching via Qdrant metadata filter."""
        must_conditions = [
            FieldCondition(key="is_latest", match=MatchValue(value=True)),
            FieldCondition(key="skills", match=MatchAny(any=required_skills)),
        ]
        if seniority_level:
            must_conditions.append(
                FieldCondition(key="seniorityLevel", match=MatchValue(value=seniority_level))
            )

        generic_query = " ".join(required_skills)
        query_vector = self.embedding_service.embed_text(generic_query, is_query=True)
        threshold = get_adaptive_threshold("cv_analysis", True)

        results = self.qdrant_service.search_similar(
            collection_name=self.cv_collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=threshold,
            filters=Filter(must=must_conditions),
        )

        return {
            "cv_context": results,
            "jd_context": [],
            "retrieval_stats": {
                "cv_chunks_retrieved": len(results),
                "skills_queried": required_skills,
                "score_range": self._get_score_range(results),
            },
        }

    def _get_score_range(self, results: List[Dict]) -> Dict[str, float]:
        """Extract min/max similarity scores from a result list."""
        if not results:
            return {"min": 0.0, "max": 0.0}
        scores = [r.get("reranker_score", r.get("score", 0.0)) for r in results]
        return {"min": min(scores), "max": max(scores)}

    def _assess_retrieval_quality(self, results: List[Dict], intent: str) -> Dict[str, Any]:
        """Classify retrieval quality as GOOD / ACCEPTABLE / POOR."""
        if not results:
            return {
                "quality": "POOR",
                "max_score": 0.0,
                "avg_score": 0.0,
                "count": 0,
                "recommendation": "No results — consider relaxing filters or rewriting query",
            }

        scores = [r.get("reranker_score", r.get("score", 0.0)) for r in results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        thresholds = {
            "cv_analysis": {"good": 0.60, "acceptable": 0.40},
            "jd_search":   {"good": 0.70, "acceptable": 0.50},
            "jd_analysis": {"good": 0.60, "acceptable": 0.40},
        }
        t = thresholds.get(intent, {"good": 0.60, "acceptable": 0.40})

        if max_score >= t["good"]:
            quality = "GOOD"
            recommendation = "High quality retrieval"
        elif max_score >= t["acceptable"]:
            quality = "ACCEPTABLE"
            recommendation = "Acceptable quality — results may be less precise"
        else:
            quality = "POOR"
            recommendation = "Low quality — consider query rewriting or expanding search"

        return {
            "quality": quality,
            "max_score": round(max_score, 3),
            "avg_score": round(avg_score, 3),
            "count": len(results),
            "recommendation": recommendation,
        }


# Global singleton instance
retriever = CareerCounselorRetriever()