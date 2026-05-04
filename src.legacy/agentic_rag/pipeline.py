"""主流水线 — Router → Retriever → QualityGate → Generator。"""
import logging
import sys
from pathlib import Path

# 添加 src 根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import get_config
from agentic_rag.state import AgenticRAGState
from agentic_rag.router_node import router_node
from agentic_rag.retriever_node import retriever_node
from agentic_rag.quality_gate_node import quality_gate_node
from agentic_rag.generator_node import generator_node

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run(question: str) -> dict:
    """
    主流水线入口。

    串联：Router → Retriever → QualityGate(loop) → Generator

    Returns:
        {
            "answer": str,        # 最终答案
            "Q": float,           # Quality Gate 评分
            "attempts": int,      # 尝试次数
            "router_type": str,   # 分类类型
            "error": str | None,
        }
    """
    cfg = get_config()

    # 初始化状态
    state: AgenticRAGState = {
        "query": question,
        "matched_skill": None,  # Phase 1 不实现 L1
        "router_output": None,
        "retrieved_docs": [],
        "quality_gate_output": None,
        "reretrieve_count": 0,
        "answer": "",
        "error": None,
    }

    # Step 1: Router（语义分类）
    state.update(router_node(state))
    if state.get("error") or state.get("router_output") is None:
        return {"answer": "抱歉，系统无法理解您的问题，请重试。", "Q": 0.0, "attempts": 0, "router_type": "error", "error": state.get("error")}

    # Step 2: Retriever（文档检索，Phase 1 mock）
    state.update(retriever_node(state))

    # Step 3: QualityGate + 重试循环
    attempts = 0
    max_retries = cfg.max_reretrieve_count

    while attempts < max_retries:
        state.update(quality_gate_node(state))
        qg_output = state.get("quality_gate_output") or {}
        if qg_output.get("passed"):
            break
        # 取递进参数传给 Retriever
        retry_params = qg_output.get("reretrieve_params", {})
        logger.info(f"QualityGate 未通过，重试 {attempts + 1}/{max_retries}，params={retry_params}")
        state.update(retriever_node({**state, "reretrieve_params": retry_params}))
        attempts += 1

    qg_output = state.get("quality_gate_output") or {}

    # Step 4: Generator（生成答案）
    if not qg_output.get("passed") and attempts >= max_retries:
        logger.warning(f"QualityGate 重试耗尽（{max_retries}次），返回兜底答案")
        return {
            "answer": "抱歉，检索到的信息不足以生成可靠回答，请尝试换一种表述或联系图书馆/教务处。",
            "Q": qg_output.get("Q", 0.0),
            "attempts": attempts,
            "router_type": state.get("router_output", {}).get("type", "unknown"),
            "error": "QualityGate 未通过",
        }

    state.update(generator_node(state))

    return {
        "answer": state.get("answer", ""),
        "Q": qg_output.get("Q", 0.0),
        "attempts": qg_output.get("attempts", 1),
        "router_type": state.get("router_output", {}).get("type", "unknown"),
        "error": None,
    }


# ========== 测试用例 ==========
TEST_QUESTIONS = [
    "借书超期了怎么办",
    "GPA怎么计算的",
    "保研需要哪些条件",
    "图书馆开放时间",
    "成绩复议流程",
]


def main():
    """运行测试。"""
    print("=" * 60)
    print("RAGarden L2 Agentic RAG 流水线 — Demo 测试")
    print("=" * 60)

    for question in TEST_QUESTIONS:
        print(f"\n【问题】{question}")
        print("-" * 50)
        try:
            result = run(question)
            print(f"[分类] {result['router_type']}")
            print(f"[QualityGate] Q={result['Q']:.3f}, 尝试={result['attempts']}")
            print(f"[答案]\n{result['answer']}")
        except Exception as e:
            logger.exception(f"处理问题时出错: {e}")
            print(f"[错误] {e}")


if __name__ == "__main__":
    main()
