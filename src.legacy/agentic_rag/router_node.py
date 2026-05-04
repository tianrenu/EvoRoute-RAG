"""RouterNode — 语义分类 + 技能候选。"""
import logging
from typing import cast

from common.llm_client import LLMClient
from .state import AgenticRAGState

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """你是一个校园知识服务查询分类专家。你必须且只能输出JSON格式的分类结果，禁止输出任何其他内容。

用户问题类型定义：
- procedure_query：流程手续类（选课、休学、请假、成绩复议、借书超期处理等）
- knowledge_query：知识类（政策、规定、概念解释、开放时间、计算公式等）
- qualification_query：资质资格类（保研条件、奖学金资格、竞赛加分等）
- other：上述类型都无法覆盖的问题

你必须只输出以下JSON格式，禁止输出任何解释、分析或文字说明：
{"type":"类型","confidence":0.85,"candidates":["次选1","次选2"]}
"""


def router_node(state: AgenticRAGState) -> dict:
    """
    RouterNode：调用 LLM 对用户问题做语义分类。

    输入: state["query"]
    输出: state["router_output"] = {"type", "confidence", "candidates"}

    Phase 1 不做 L1 技能匹配，全部走 L2。
    """
    query = state.get("query", "")
    if not query:
        return {"router_output": None, "error": "Router: query 为空"}

    client = LLMClient()
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"用户问题：{query}"},
    ]

    try:
        result = client.chat(messages, temperature=0.1)
    except Exception as e:
        logger.error(f"Router LLM 调用失败: {e}")
        return {"router_output": None, "error": f"Router LLM 调用失败: {e}"}

    # 防御：确保返回字段完整
    if not result or "type" not in result:
        logger.warning(f"Router LLM 返回格式异常: {result}，降级为 other")
        output = {"type": "other", "confidence": 0.0, "candidates": []}
    else:
        output = {
            "type": result.get("type", "other"),
            "confidence": float(result.get("confidence", 0.0)),
            "candidates": result.get("candidates", []),
        }

    logger.info(f"Router: type={output['type']}, confidence={output['confidence']:.2f}")
    return {"router_output": output}
