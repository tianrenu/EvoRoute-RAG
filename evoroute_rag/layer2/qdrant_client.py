"""Qdrant vector search client with DashScope embedding support."""

from typing import List, Optional

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, VectorParams


COLLECTION_NAME = "campus_knowledge"
EMBEDDING_DIM = 1536


class QdrantRetriever:
    """Qdrant retrieval wrapper using DashScope text-embedding-v4."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        embedding_api_key: str = "",
        embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.embedding_client = openai.OpenAI(
            base_url=embedding_base_url,
            api_key=embedding_api_key,
        )

    def ensure_collection(self):
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding using DashScope text-embedding-v4."""
        response = self.embedding_client.embeddings.create(
            model="text-embedding-v4",
            input=text,
        )
        return response.data[0].embedding

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[dict] = None,
    ) -> List[dict]:
        """Search Qdrant for similar documents.

        Returns:
            List of dicts with keys: id, score, content, metadata
        """
        query_vector = self.embed_query(query)

        # Build filter conditions from metadata
        qdrant_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        docs = []
        for point in results:
            docs.append({
                "id": point.id,
                "score": point.score,
                "content": point.payload.get("content", ""),
                "metadata": {
                    k: v for k, v in point.payload.items() if k != "content"
                },
            })
        return docs
