"""QualityGateNode — 检索质量判断。"""
import logging
import re

from common.config import get_config
from .state import AgenticRAGState

logger = logging.getLogger(__name__)


def _tokenize_query(query: str) -> set[str]:
    """
    对 query 做简易分词，返回关键词集合（全小写）。

    英文/数字：按空白分词，长度>=2才保留。
    中文：贪婪切分为 2-3 字词（优先3字，再2字），单字丢弃避免虚高命中。
    """
    if not query:
        return set()

    query = query.lower()
    tokens: set[str] = set()
    segments = re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+', query)

    for seg in segments:
        if re.fullmatch(r'[\u4e00-\u9fff]+', seg):
            # 中文段：贪婪切分 3字 → 2字
            i = 0
            while i < len(seg):
                if i + 3 <= len(seg):
                    tokens.add(seg[i:i + 3])
                    i += 3
                elif i + 2 <= len(seg):
                    tokens.add(seg[i:i + 2])
                    i += 2
                else:
                    i += 1  # 剩余单字丢弃
        else:
            if len(seg) >= 2:
                tokens.add(seg)
    return tokens


def _compute_Q(docs: list[dict], query: str = "") -> tuple[float, float, float, float, float]:
    """
    计算 Quality Gate 综合评分 Q = α·S + β·M + γ·A + δ·C。

    S（相关性）   = 所有 doc 的 score 均值
    M（区分度）   = top1_score - top2_score
    A（一致性）   = 1 / (1 + variance_of_scores)
    C（覆盖率）   = query 中关键词在 doc 内容里的覆盖率（中文2-3字词匹配）
    """
    cfg = get_config()

    if not docs:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    scores = [d.get("score", 0.0) for d in docs]

    # S：所有 doc 的 score 均值
    S = sum(scores) / len(scores)

    # M：区分度 = top1_score - top2_score
    sorted_scores = sorted(scores, reverse=True)
    M = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else sorted_scores[0]

    # A：一致性 = 1 / (1 + variance_of_scores)
    variance = sum((s - S) ** 2 for s in scores) / len(scores)
    A = 1.0 / (1.0 + variance)

    # C：覆盖率 = query 中关键词在 doc 内容里的覆盖率（中文2-3字词）
    keywords = _tokenize_query(query)
    if keywords:
        all_content = " ".join(d.get("content", "") for d in docs).lower()
        hit = sum(1 for kw in keywords if kw in all_content)
        C = hit / len(keywords)
    else:
        C = 0.0

    Q = cfg.q_weight_s * S + cfg.q_weight_m * M + cfg.q_weight_a * A + cfg.q_weight_c * C
    Q = min(Q, 1.0)

    return Q, S, M, A, C


# 递进重试策略（对应设计书 §4.2.2）
RERETRIEVE_STRATEGIES = [
    {"top_k_multiplier": 1.0, "filter_level": "strict", "query_expansion": False},
    {"top_k_multiplier": 1.6, "filter_level": "relaxed", "query_expansion": False},
    {"top_k_multiplier": 2.4, "filter_level": "minimal", "query_expansion": True},
]


def quality_gate_node(state: AgenticRAGState) -> dict:
    """
    QualityGateNode：判断检索质量是否足够支撑生成。

    Phase 1 使用固定阈值 0.60，最多重试 3 次。

    输入: state["retrieved_docs"], state["query"]
    输出: state["quality_gate_output"], state["reretrieve_count"]
    """
    cfg = get_config()
    docs = state.get("retrieved_docs", [])
    reretrieve_count = state.get("reretrieve_count", 0)

    query = state.get("query", "")
    Q, S, M, A, C = _compute_Q(docs, query)
    passed = Q >= cfg.quality_gate_threshold

    # 计算本轮重试参数（用于 retriever）
    idx = reretrieve_count  # 第0次=初始，第1次=第2次重试...
    strategy = RERETRIEVE_STRATEGIES[idx] if idx < len(RERETRIEVE_STRATEGIES) else RERETRIEVE_STRATEGIES[-1]

    output = {
        "passed": passed,
        "Q": Q,
        "attempts": reretrieve_count + 1,
        "reretrieve_params": strategy,
        "retrieval_signal": {
            "retrieval_quality": Q,
            "router_confidence": (state.get("router_output") or {}).get("confidence", 0.0),
            "top_k_scores": [d["score"] for d in docs],
            "avg_score": sum(d["score"] for d in docs) / max(len(docs), 1),
            "low_score_count": sum(1 for d in docs if d.get("score", 0) < 0.4),
            "attempts": reretrieve_count + 1,
            "passed": passed,
            "threshold": cfg.quality_gate_threshold,
            "needs_skill_evolution": False,  # Phase 1 不触发
            "skill_id": None,
            "query": state.get("query", ""),
            "docs": docs,
        },
    }

    logger.info(
        f"QualityGate: Q={Q:.3f} (S={S:.2f} M={M:.2f} A={A:.2f} C={C:.2f}), "
        f"passed={passed}, attempts={output['attempts']}"
    )

    return {
        "quality_gate_output": output,
        "reretrieve_count": reretrieve_count + 1 if not passed else reretrieve_count,
    }
