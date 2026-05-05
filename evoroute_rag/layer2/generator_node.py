"""Generator node - LLM answer generation based on retrieved documents."""

from typing import Any

from evoroute_rag.layer2.llm_client import LLMClient


GENERATOR_SYSTEM_PROMPT = """你是校园知识问答助手。请基于以下检索到的参考文档回答用户问题。

要求：
1. 答案必须基于提供的参考文档内容
2. 如果参考文档不足以回答问题，请明确说明"我无法根据现有文档回答此问题"
3. 回答简洁准确，不要编造信息
4. 如有多个文档相关，综合各文档内容作答"""


def generator_node(state: dict, llm_client: LLMClient) -> dict:
    """Generator node: produce final answer using LLM based on retrieved docs."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])

    # Build context from retrieved documents
    context = _build_context(docs)

    user_msg = f"参考文档：\n{context}\n\n用户问题：{query}"

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        answer = llm_client.call(messages)
    except Exception:
        answer = "抱歉，系统暂时无法生成回答，请稍后再试。"

    return {
        **state,
        "final_answer": answer,
    }


def _build_context(docs: list) -> str:
    """Build context string from retrieved documents."""
    if not docs:
        return "（无相关文档）"

    parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("content", "")
        score = doc.get("score", 0.0)
        parts.append(f"[文档{i}] (相关度: {score:.2f})\n{content}")

    return "\n\n".join(parts)
