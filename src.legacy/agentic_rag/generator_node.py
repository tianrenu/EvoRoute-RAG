"""GeneratorNode — LLM 生成答案。"""
import logging

from common.config import get_config
from common.llm_client import LLMClient
from .state import AgenticRAGState

logger = logging.getLogger(__name__)


GENERATOR_SYSTEM_PROMPT = """你是一个校园知识问答助手。请根据提供的上下文信息，准确、简洁地回答用户的问题。

要求：
1. 只基于提供的上下文回答，不要编造信息
2. 如果上下文中没有相关信息，诚实地说明"我无法从已知信息中找到答案"
3. 答案要清晰、有条理
4. 适当引用上下文中的具体规定和数据
"""

GENERATOR_USER_TEMPLATE = """上下文信息：
{context}

用户问题：{question}

请根据上述上下文信息回答用户问题。"""


def _build_context(docs: list[dict]) -> str:
    """将检索到的文档拼装成上下文字符串。"""
    if not docs:
        return "（未检索到相关文档）"

    lines = []
    for i, doc in enumerate(docs, 1):
        score = doc.get("score", 0.0)
        content = doc.get("content", "")
        source = doc.get("metadata", {}).get("source", "未知来源")
        lines.append(f"[文档{i}]（相关度:{score:.2f}，来源:{source}）\n{content}")
    return "\n\n".join(lines)


def generator_node(state: AgenticRAGState) -> dict:
    """
    GeneratorNode：基于检索结果生成答案。

    输入: state["query"], state["retrieved_docs"], state["router_output"]
    输出: state["answer"]
    """
    cfg = get_config()
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    router_output = state.get("router_output") or {}
    semantic_type = router_output.get("type", "other")

    context = _build_context(docs)
    user_prompt = GENERATOR_USER_TEMPLATE.format(context=context, question=query)

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 补充语义类型提示（让生成更专业）
    type_hints = {
        "procedure_query": "（这是流程手续类问题，请重点说明具体步骤和注意事项）",
        "knowledge_query": "（这是知识类问题，请准确解释相关概念和规定）",
        "qualification_query": "（这是资质资格类问题，请列出具体条件和标准）",
        "other": "",
    }
    messages[1]["content"] += f"\n\n{type_hints.get(semantic_type, '')}"

    try:
        client = LLMClient()
        answer = client.text(messages, temperature=cfg.generator_temperature)
        if not answer:
            answer = "抱歉，系统暂时无法生成答案，请稍后再试。"
    except Exception as e:
        logger.error(f"Generator LLM 调用失败: {e}")
        answer = "抱歉，系统生成答案时出现故障，请稍后再试。"

    logger.info(f"Generator: 生成答案长度={len(answer)}字")
    return {"answer": answer}
