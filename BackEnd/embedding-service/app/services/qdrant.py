from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.config import get_settings

settings = get_settings()


class QdrantService:
    """Singleton service for Qdrant operations"""
    
    _instance: Optional['QdrantService'] = None
    _client: Optional[QdrantClient] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client"""
        if self._client is None:
            if settings.QDRANT_USE_CLOUD:
                # Qdrant Cloud
                print(f"Connecting to Qdrant Cloud: {settings.QDRANT_URL}")
                self._client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=60
                )
            else:
                # Local Qdrant
                print(f"Connecting to Qdrant Local: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
                self._client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=60
                )
            print("Qdrant client connected successfully!")
        return self._client
    
    def test_connection(self) -> bool:
        """Test Qdrant connection"""
        try:
            client = self._get_client()
            collections = client.get_collections()
            print(f"Qdrant connection OK. Collections: {len(collections.collections)}")
            return True
        except Exception as e:
            print(f"Qdrant connection failed: {e}")
            return False
    
    def create_collection(self, collection_name: str, vector_size: int):
        """Create a collection if not exists"""
        client = self._get_client()
        
        # Check if collection exists
        collections = client.get_collections().collections
        if any(col.name == collection_name for col in collections):
            print(f"Collection '{collection_name}' already exists")
            return
        
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE  # Cosine similarity for semantic search
            )
        )
        print(f"Collection '{collection_name}' created successfully!")
    
    def upsert_points(self, collection_name: str, points: List[PointStruct]) -> bool:
        """
        Upsert points to collection
        
        Args:
            collection_name: Name of collection
            points: List of PointStruct objects
            
        Returns:
            True if successful
        """
        try:
            client = self._get_client()
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"Error upserting points: {e}")
            return False
    
    def search_similar(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            collection_name: Name of collection
            query_vector: Query embedding vector
            limit: Number of results
            score_threshold: Minimum similarity score (0-1)
            filters: Optional filters
            
        Returns:
            List of search results
        """
        try:
            client = self._get_client()
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filters
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def delete_by_filter(self, collection_name: str, filters: Filter) -> bool:
        """Delete points by filter"""
        try:
            client = self._get_client()
            client.delete(
                collection_name=collection_name,
                points_selector=filters
            )
            return True
        except Exception as e:
            print(f"Error deleting by filter: {e}")
            return False

    def update_applied_positions(self, collection_name: str, cv_id: int, position_id: int) -> bool:
        """
        Append a position_id to the applied_position_ids array for all chunks of a CV.
        Used for live Phase 3 synchronization when a candidate finalizes an application.
        """
        try:
            client = self._get_client()
            
            # 1. Scroll to find all points for this cvId
            filter_condition = Filter(
                must=[FieldCondition(key="cvId", match=MatchValue(value=cv_id))]
            )
            
            offset = None
            total_updated = 0
            
            while True:
                results, offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                    limit=100
                )
                
                if not results:
                    break
                
                for point in results:
                    payload = point.payload or {}
                    applied_ids = payload.get("applied_position_ids", [])
                    
                    if not isinstance(applied_ids, list):
                        applied_ids = []
                    
                    if position_id not in applied_ids:
                        applied_ids.append(position_id)
                        
                        # Overwrite specific key in payload
                        client.set_payload(
                            collection_name=collection_name,
                            payload={"applied_position_ids": applied_ids},
                            points=[point.id]
                        )
                        total_updated += 1
                
                if offset is None:
                    break
            
            print(f"Updated applied_position_ids for CV {cv_id}: added {position_id} to {total_updated} chunks")
            return True
        except Exception as e:
            print(f"Error updating applied positions for CV {cv_id}: {e}")
            return False
    
    def update_jd_metadata(self, collection_name: str, position_id: int, position_title: str, seniority: str) -> bool:
        """
        Overwrite positionTitle and seniority for all JD chunks of a position.
        Called when position metadata changes — avoids full re-embedding since vectors are unchanged.
        """
        try:
            client = self._get_client()
            filter_condition = Filter(
                must=[FieldCondition(key="positionId", match=MatchValue(value=position_id))]
            )
            client.set_payload(
                collection_name=collection_name,
                payload={"positionTitle": position_title, "seniority": seniority},
                points=filter_condition,
            )
            print(f"[Qdrant] Updated positionTitle/seniority for position {position_id}")
            return True
        except Exception as e:
            print(f"[Qdrant] Error updating JD metadata for position {position_id}: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> dict | None:
        try:
            client = self._get_client()
            
            info = client.get_collection(collection_name)
            
            points_count = 0
            if hasattr(info, 'points_count'):
                points_count = info.points_count
            elif hasattr(info, 'vectors_count'):
                points_count = info.vectors_count
            elif hasattr(info, 'status') and hasattr(info.status, 'points_count'):
                points_count = info.status.points_count
            
            vectors_config = info.config.params.vectors
            
            if isinstance(vectors_config, dict):
                vector_params = vectors_config.get('') or list(vectors_config.values())[0]
            else:
                vector_params = vectors_config
            
            return {
                "name": collection_name,
                "vector_size": vector_params.size,
                "distance": vector_params.distance.value,
                "points_count": points_count,
            }
        except Exception as e:
            print(f"Error getting collection info: {e}") 
            import traceback
            traceback.print_exc()
            return None
    
    def get_client(self) -> QdrantClient:
        """Public method to get client"""
        return self._get_client()


# Global instance
qdrant_service = QdrantService()