"""LangGraph state graph orchestration for L2 Agentic RAG pipeline."""

from typing import List, TypedDict

from langgraph.graph import END, StateGraph


class AgenticRAGState(TypedDict):
    query: str
    query_type: str
    router_confidence: float
    skill_config: dict
    retrieved_docs: List[dict]
    mean_similarity: float
    top1_margin: float
    doc_agreement: float
    coverage_score: float
    quality_score: float
    reretrieve_count: int
    attribution_evidence: dict
    final_answer: str


def build_graph(llm_client, retriever) -> StateGraph:
    """Build and compile the LangGraph state graph for Agentic RAG.

    Args:
        llm_client: LLMClient instance for Router and Generator nodes
        retriever: QdrantRetriever instance for Retriever node

    Returns:
        Compiled LangGraph StateGraph
    """
    from evoroute_rag.layer2.router_node import router_node
    from evoroute_rag.layer2.retriever_node import retriever_node
    from evoroute_rag.layer2.quality_gate_node import quality_gate_node, quality_gate_router
    from evoroute_rag.layer2.generator_node import generator_node

    # Wrap nodes to inject dependencies
    def _router(state: dict) -> dict:
        return router_node(state, llm_client)

    def _retriever(state: dict) -> dict:
        return retriever_node(state, retriever)

    def _quality_gate(state: dict) -> dict:
        return quality_gate_node(state)

    def _generator(state: dict) -> dict:
        return generator_node(state, llm_client)

    def _reretrieve(state: dict) -> dict:
        """Increment reretrieve_count and loop back to retriever."""
        return {**state, "reretrieve_count": state.get("reretrieve_count", 0) + 1}

    # Build graph
    graph = StateGraph(AgenticRAGState)

    graph.add_node("router", _router)
    graph.add_node("retriever", _retriever)
    graph.add_node("quality_gate", _quality_gate)
    graph.add_node("generator", _generator)
    graph.add_node("reretrieve", _reretrieve)

    # Edges
    graph.set_entry_point("router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "quality_gate")

    # Conditional routing after quality gate
    graph.add_conditional_edges(
        "quality_gate",
        quality_gate_router,
        {
            "pass": "generator",
            "pass_with_log": "generator",
            "reretrieve": "reretrieve",
        },
    )

    graph.add_edge("reretrieve", "retriever")
    graph.add_edge("generator", END)

    return graph.compile()
