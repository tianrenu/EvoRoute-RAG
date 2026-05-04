"""AgenticRAGState — LangGraph 状态定义。"""
from typing import TypedDict


class AgenticRAGState(TypedDict, total=False):
    """L2 Agentic RAG 流水线状态。"""

    # ========== 输入 ==========
    query: str  # 用户原始问题

    # ========== L1 匹配结果（Phase 1 不实现，保留字段） ==========
    matched_skill: dict | None  # L1 命中的技能对象，Phase 1 恒为 None

    # ========== RouterNode 输出 ==========
    router_output: dict | None  # {"type": str, "confidence": float, "candidates": list}

    # ========== RetrieverNode 输出 ==========
    retrieved_docs: list[dict]  # [{"content": str, "score": float, "metadata": dict}]

    # ========== QualityGateNode 输出 ==========
    quality_gate_output: dict | None  # {"passed": bool, "Q": float, "attempts": int, "retrieval_signal": dict}
    reretrieve_count: int  # 当前重试次数

    # ========== GeneratorNode 输出 ==========
    answer: str  # 最终答案

    # ========== 错误处理 ==========
    error: str | None  # 异常信息
