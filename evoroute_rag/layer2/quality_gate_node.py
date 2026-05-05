"""Quality Gate node - composite quality scoring and decision routing."""

from typing import Any


# Phase 1 initial weights
ALPHA = 0.40  # S: mean similarity
BETA = 0.25   # M: top1 margin
GAMMA = 0.20  # A: doc agreement
DELTA = 0.15  # C: coverage

# Thresholds
Q_HIGH = 0.75
Q_LOW = 0.55
MAX_RERETRIEVE = 3


def quality_gate_node(state: dict) -> dict:
    """Quality Gate: compute Q score and build attribution_evidence."""
    s = state.get("mean_similarity", 0.0)
    m = state.get("top1_margin", 0.0)
    a = state.get("doc_agreement", 0.0)
    c = state.get("coverage_score", 0.0)

    # Q = α·S + β·M + γ·A + δ·C
    q_score = ALPHA * s + BETA * m + GAMMA * a + DELTA * c

    # Determine gap_type
    gap_type = _determine_gap_type(s, m, c, state.get("retrieved_docs", []))

    # Build attribution_evidence
    docs = state.get("retrieved_docs", [])
    top1_sim = docs[0]["score"] if docs else 0.0
    top_k_max_sim = max((d["score"] for d in docs), default=0.0)
    has_correct_entity = False  # Heuristic: will be refined in L3

    attribution_evidence = {
        "S": round(s, 4),
        "M": round(m, 4),
        "A": round(a, 4),
        "C": round(c, 4),
        "Q": round(q_score, 4),
        "gap_type": gap_type,
        "top_k_max_sim": round(top_k_max_sim, 4),
        "top1_sim": round(top1_sim, 4),
        "has_correct_entity": has_correct_entity,
        "reretrieve_count": state.get("reretrieve_count", 0),
    }

    return {
        **state,
        "quality_score": q_score,
        "attribution_evidence": attribution_evidence,
    }


def quality_gate_router(state: dict) -> str:
    """Decision routing based on quality score Q.

    Returns:
        "pass" - high confidence (Q >= 0.75)
        "pass_with_log" - gray zone or fallback
        "reretrieve" - retry retrieval (Q < 0.55 and retries remaining)
    """
    q = state.get("quality_score", 0.0)
    reretrieve_count = state.get("reretrieve_count", 0)

    if q >= Q_HIGH:
        return "pass"
    elif q >= Q_LOW:
        return "pass_with_log"
    elif reretrieve_count < MAX_RERETRIEVE:
        return "reretrieve"
    else:
        return "pass_with_log"  # Fallback: force pass after max retries


def _determine_gap_type(s: float, m: float, c: float, docs: list) -> str:
    """Heuristic gap type determination.

    - coverage_score < 0.4 → keyword_missing
    - top1_margin < 0.05 and multiple docs → precision_dilution
    - mean_similarity < 0.4 → doc_missing
    """
    if c < 0.4:
        return "keyword_missing"
    if m < 0.05 and len(docs) > 2:
        return "precision_dilution"
    if s < 0.4:
        return "doc_missing"
    return "keyword_missing"  # Default fallback
