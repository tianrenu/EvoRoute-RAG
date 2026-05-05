"""L2 Agentic RAG Pipeline - unified entry point."""

import os
from typing import Any, Dict, List, Optional

from evoroute_rag.layer2.langgraph_pipeline import AgenticRAGState, build_graph
from evoroute_rag.layer2.llm_client import LLMClient
from evoroute_rag.layer2.qdrant_client import QdrantRetriever


class AgenticRAGPipeline:
    """L2 Agentic RAG Pipeline - unified entry point.

    Orchestrates Router → Retriever → Quality Gate → Generator via LangGraph.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        llm_config = llm_config or {}

        # LLM client (MiniMax API via OpenAI-compatible format)
        self.llm_client = LLMClient(
            base_url=llm_config.get("base_url", os.getenv("LLM_API_BASE", "https://api.minimaxi.com/v1")),
            api_key=llm_config.get("api_key", os.getenv("OPENAI_API_KEY", "")),
            model=llm_config.get("model", "abab6.5-chat"),
        )

        # Qdrant retriever (DashScope embedding)
        self.retriever = QdrantRetriever(
            qdrant_url=qdrant_url,
            embedding_api_key=llm_config.get(
                "embedding_api_key",
                os.getenv("DASHSCOPE_API_KEY", ""),
            ),
            embedding_base_url=llm_config.get(
                "embedding_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )

        # Build LangGraph
        self.graph = build_graph(self.llm_client, self.retriever)

    def run(self, query: str, skill_config: Optional[dict] = None) -> Dict[str, Any]:
        """Run the Agentic RAG pipeline.

        Args:
            query: User's question
            skill_config: Optional L1 skill configuration dict containing
                          retrieval params (top_k, filter_metadata, boost_keywords)

        Returns:
            {
                "final_answer": str,
                "query_type": str,
                "router_confidence": float,
                "quality_score": float,
                "attribution_evidence": dict,
                "retrieved_docs": List[dict]
            }
        """
        initial_state: AgenticRAGState = {
            "query": query,
            "query_type": "",
            "router_confidence": 0.0,
            "skill_config": skill_config or {},
            "retrieved_docs": [],
            "mean_similarity": 0.0,
            "top1_margin": 0.0,
            "doc_agreement": 0.0,
            "coverage_score": 0.0,
            "quality_score": 0.0,
            "reretrieve_count": 0,
            "attribution_evidence": {},
            "final_answer": "",
        }

        result = self.graph.invoke(initial_state)

        return {
            "final_answer": result.get("final_answer", ""),
            "query_type": result.get("query_type", ""),
            "router_confidence": result.get("router_confidence", 0.0),
            "quality_score": result.get("quality_score", 0.0),
            "attribution_evidence": result.get("attribution_evidence", {}),
            "retrieved_docs": result.get("retrieved_docs", []),
        }
