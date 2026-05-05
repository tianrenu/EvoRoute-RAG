"""Router node - LLM zero-shot classification for query type."""

from typing import Any

from evoroute_rag.layer2.llm_client import LLMClient


ROUTER_SYSTEM_PROMPT = """你是一个查询分类器。根据用户查询，判断其类型并输出置信度。

可选类型：
- factual_query: 事实性问题（如"图书馆几点开门"）
- procedure_query: 流程/步骤类问题（如"如何办理学生证"）
- policy_query: 规章制度类问题（如"考试作弊如何处分"）
- navigation_query: 地点/路线类问题（如"食堂在哪"）
- schedule_query: 时间/日程类问题（如"下学期什么时候开学"）

请严格按以下格式输出（无额外文字）：
type: <类型>
confidence: <0到1的小数>"""


def router_node(state: dict, llm_client: LLMClient) -> dict:
    """Router node: LLM zero-shot classification for query type and confidence.

    If L1 skill_config provides a semantic_type, use it as a reference hint.
    """
    query = state["query"]
    skill_config = state.get("skill_config") or {}

    # Build user message with optional L1 hint
    user_msg = f"用户查询: {query}"
    if skill_config.get("semantic_type"):
        user_msg += f"\n参考提示（来自L1路由）: {skill_config['semantic_type']}"

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        response = llm_client.call(messages)
        query_type, confidence = _parse_router_response(response)
    except Exception:
        # Fallback: use L1 hint or default
        query_type = skill_config.get("semantic_type", "factual_query")
        confidence = 0.3

    return {
        **state,
        "query_type": query_type,
        "router_confidence": confidence,
    }


def _parse_router_response(response: str) -> tuple:
    """Parse router LLM response into (query_type, confidence)."""
    query_type = "factual_query"
    confidence = 0.5

    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("type:"):
            query_type = line.split(":", 1)[1].strip()
        elif line.startswith("confidence:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

    return query_type, confidence
