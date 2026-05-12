from collections import OrderedDict
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from app.config import get_settings

settings = get_settings()

# Module-level LRU cache for query/text embeddings.
# Key: (text[:300], is_query) — 512 entries covers typical session diversity.
_EMBED_CACHE: "OrderedDict[Tuple[str, bool], List[float]]" = OrderedDict()
_EMBED_CACHE_MAX = 512


class EmbeddingService:
    """Singleton service for text embedding with lazy loading"""
    
    _instance: Optional['EmbeddingService'] = None
    _model: Optional[SentenceTransformer] = None
    
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_model(self):
        """Lazy load the embedding model"""
        if self._model is None:
            print(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            print(f"Model loaded successfully! Dimension: {settings.EMBEDDING_DIMENSION}")
    
    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Embed a single text with LRU cache to avoid redundant model calls."""
        cache_key: Tuple[str, bool] = (text[:300], is_query)
        if cache_key in _EMBED_CACHE:
            _EMBED_CACHE.move_to_end(cache_key)
            return _EMBED_CACHE[cache_key]

        self._load_model()
        full_text = (self.QUERY_INSTRUCTION + text) if is_query else text
        embedding = self._model.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)
        result: List[float] = embedding.tolist()

        _EMBED_CACHE[cache_key] = result
        if len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
            _EMBED_CACHE.popitem(last=False)
        return result
    
    def embed_batch(
        self, 
        texts: List[str], 
        is_query: bool = False,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Embed a batch of texts
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        self._load_model()

        if is_query:
            texts = [self.QUERY_INSTRUCTION + text for text in texts]

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  
            show_progress_bar=show_progress,
            batch_size=32
        )
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return settings.EMBEDDING_DIMENSION
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None


# Global instance
embedding_service = EmbeddingService()