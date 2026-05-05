"""Retriever node - Qdrant search with quality metrics computation."""

import itertools
from typing import Any, List

import numpy as np

from evoroute_rag.layer2.qdrant_client import QdrantRetriever


def retriever_node(state: dict, retriever: QdrantRetriever) -> dict:
    """Retriever node: execute Qdrant search and compute S, M, A, C metrics.

    Reads skill_config for: boost_keywords, filter_metadata, top_k.
    On reretrieve (reretrieve_count > 0), optionally expands the query.
    """
    query = state["query"]
    skill_config = state.get("skill_config") or {}
    reretrieve_count = state.get("reretrieve_count", 0)

    # Read retrieval config from L1 skill
    retrieval_cfg = skill_config.get("retrieval", {})
    top_k = retrieval_cfg.get("top_k", 5)
    filter_metadata = retrieval_cfg.get("filter_metadata")
    boost_keywords = retrieval_cfg.get("boost_keywords", [])

    # On reretrieve, expand query with boost keywords
    search_query = query
    if reretrieve_count > 0 and boost_keywords:
        search_query = f"{query} {' '.join(boost_keywords)}"

    # Execute search
    docs = retriever.search(
        query=search_query,
        top_k=top_k,
        filter_metadata=filter_metadata,
    )

    # Compute quality metrics
    s_score = _compute_mean_similarity(docs)
    m_score = _compute_top1_margin(docs)
    a_score = _compute_doc_agreement(docs, retriever)
    c_score = _compute_coverage(query, docs, boost_keywords)

    return {
        **state,
        "retrieved_docs": docs,
        "mean_similarity": s_score,
        "top1_margin": m_score,
        "doc_agreement": a_score,
        "coverage_score": c_score,
        "reretrieve_count": reretrieve_count,
    }


def _compute_mean_similarity(docs: List[dict]) -> float:
    """S: mean similarity score of retrieved docs."""
    if not docs:
        return 0.0
    return float(np.mean([d["score"] for d in docs]))


def _compute_top1_margin(docs: List[dict]) -> float:
    """M: difference between top-1 and top-2 similarity scores."""
    if len(docs) < 2:
        return 0.0
    return float(docs[0]["score"] - docs[1]["score"])


def _compute_doc_agreement(docs: List[dict], retriever: QdrantRetriever) -> float:
    """A: mean pairwise cosine similarity among retrieved doc embeddings.

    Uses a lightweight heuristic: pairwise score comparison.
    Since we already have scores, approximate agreement by score variance inverse.
    Lower variance = higher agreement.
    """
    if len(docs) < 2:
        return 1.0

    scores = [d["score"] for d in docs]
    score_std = float(np.std(scores))
    # Normalize: low std → high agreement (1.0), high std → low agreement (0.0)
    # std of cosine scores typically ranges 0.0 ~ 0.3
    agreement = max(0.0, 1.0 - score_std * 3.0)
    return agreement


def _compute_coverage(query: str, docs: List[dict], boost_keywords: List[str]) -> float:
    """C: keyword/entity coverage score (heuristic, no LLM needed).

    Checks how many query tokens and boost_keywords appear in retrieved docs.
    """
    if not docs:
        return 0.0

    # Extract keywords from query (simple tokenization)
    import jieba
    query_tokens = set(jieba.cut(query))
    # Remove stop words (single chars)
    query_tokens = {t for t in query_tokens if len(t) > 1}

    # Add boost keywords
    all_keywords = query_tokens | set(boost_keywords)
    if not all_keywords:
        return 1.0

    # Check coverage in retrieved docs
    doc_text = " ".join(d.get("content", "") for d in docs)
    covered = sum(1 for kw in all_keywords if kw in doc_text)
    return covered / len(all_keywords)
