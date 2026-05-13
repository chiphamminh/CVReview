"""
Local Cross-Encoder reranker for the Two-Stage RAG Pipeline.

Strategy: Chunk-level reranking → Group by ID (Max Score) → Top-N selection.

Rationale: CV/JD chunks are stored individually in Qdrant (typically < 512 tokens
each). The Cross-Encoder scores each (query, chunk) pair. Grouping by cvId/positionId
with Max Score ensures that if even one chunk in a CV is highly relevant, the entire
CV is promoted — which is the correct semantic behaviour for HR talent search.
"""

import asyncio
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from app.config import get_settings

settings = get_settings()

_RERANKER_ID_FIELDS = {
    "cv": "cvId",
    "jd": "positionId",
}


class LocalReranker:
    """
    Wraps a local Cross-Encoder model for two-stage retrieval.

    Loaded lazily on first use to avoid blocking startup if the model
    is not yet cached locally.
    """

    def __init__(self) -> None:
        self._model: Optional[CrossEncoder] = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            print(f"[Reranker] Loading Cross-Encoder model: {settings.RERANKER_MODEL_NAME}")
            self._model = CrossEncoder(settings.RERANKER_MODEL_NAME, max_length=512)
            print("[Reranker] Model loaded successfully.")
        return self._model

    def rerank_and_group(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        id_field: str,
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Rerank Qdrant chunk results at chunk level, then group by ID using Max Score.

        Args:
            query:    The user query string used as the Cross-Encoder 'sentence A'.
            chunks:   Raw Qdrant search results (each item has 'payload' and 'score').
            id_field: Payload key used to group chunks ('cvId' or 'positionId').
            top_n:    Number of unique IDs to return after grouping.

        Returns:
            List of up to top_n representative chunks — one per unique ID,
            carrying the highest reranker score for that ID.
        """
        if not chunks:
            return []

        model = self._get_model()

        # Build (query, chunk_text) pairs for the Cross-Encoder.
        # JD chunks store text in 'jdText'; CV chunks use 'chunkText'.
        def _chunk_text(p: dict) -> str:
            text = (p.get("jdText") or p.get("chunkText") or "").strip()
            words = text.split()
            return " ".join(words[:100]) if len(words) > 100 else text

        pairs = [
            (query, _chunk_text(chunk.get("payload", {})))
            for chunk in chunks
        ]
        rerank_scores: List[float] = model.predict(pairs).tolist()

        # Annotate each chunk with its reranker score
        for chunk, score in zip(chunks, rerank_scores):
            chunk["reranker_score"] = score

        # Group chunks by ID and keep track of the max score per ID
        chunks_by_id: Dict[Any, List[Dict[str, Any]]] = {}
        best_score_by_id: Dict[Any, float] = {}

        for chunk in chunks:
            group_id = chunk.get("payload", {}).get(id_field)
            if group_id is None:
                continue

            if group_id not in chunks_by_id:
                chunks_by_id[group_id] = []
                best_score_by_id[group_id] = chunk["reranker_score"]
            else:
                if chunk["reranker_score"] > best_score_by_id[group_id]:
                    best_score_by_id[group_id] = chunk["reranker_score"]
            
            chunks_by_id[group_id].append(chunk)

        # Sort grouped results by best reranker_score descending, take top_n IDs
        ranked_ids = sorted(best_score_by_id.keys(), key=lambda k: best_score_by_id[k], reverse=True)
        top_ids = ranked_ids[:top_n]

        # Flatten all chunks for the top IDs
        selected = []
        for gid in top_ids:
            cv_chunks = chunks_by_id[gid]
            # Sort chunks within the CV by score descending (most relevant sections first)
            cv_chunks.sort(key=lambda c: c["reranker_score"], reverse=True)
            selected.extend(cv_chunks)

        print(
            f"[Reranker] {len(chunks)} chunks → {len(best_score_by_id)} unique IDs "
            f"→ top {len(top_ids)} IDs selected ({len(selected)} total chunks, field='{id_field}')"
        )
        return selected

    async def rerank_and_group_async(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        id_field: str,
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """Non-blocking wrapper — runs rerank_and_group in a thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.rerank_and_group(query, chunks, id_field, top_n)
        )

    def rerank_chunks_for_ids(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        id_field: str,
        top_n: int,
    ) -> List[int]:
        """
        Convenience wrapper: returns the ordered list of unique IDs after reranking.
        """
        selected = self.rerank_and_group(query, chunks, id_field, top_n)
        return [chunk["payload"][id_field] for chunk in selected]


# Global singleton — model is lazy-loaded on first call
reranker = LocalReranker()
