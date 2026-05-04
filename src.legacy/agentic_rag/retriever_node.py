"""RetrieverNode — Qdrant 向量检索。"""
import logging
from typing import TypedDict

from agentic_rag.state import AgenticRAGState

logger = logging.getLogger(__name__)


class MockData:
    """Phase 1 Mock 数据。"""

    LIBRARY_CARDS = [
        {
            "user": "借书超期了怎么办",
            "semantic_type": "procedure_query",
            "docs": [
                {"content": "借书超期罚款标准：每册每日0.10元，上限50元/册。请到图书馆一楼服务台办理缴纳。", "score": 0.92, "metadata": {"source": "图书馆规则", "category": "procedure"}},
                {"content": "寒暑假期间（1-2月、7-8月）超期不计入罚款时间，请联系图书馆申请豁免。", "score": 0.78, "metadata": {"source": "图书馆FAQ", "category": "policy"}},
                {"content": "借书期限：本科生30天，研究生90天，可续借一次。", "score": 0.71, "metadata": {"source": "借阅规则", "category": "knowledge"}},
            ],
        },
        {
            "user": "GPA怎么计算的",
            "semantic_type": "knowledge_query",
            "docs": [
                {"content": "GPA计算公式：Σ(课程学分×课程绩点) ÷ Σ课程学分。本校采用4.0制，90分=4.0，80分=3.0，70分=2.0，60分=1.0。", "score": 0.94, "metadata": {"source": "教务处规定", "category": "policy"}},
                {"content": "必修课和选修课均计入GPA，但重修课程只计最高成绩。", "score": 0.82, "metadata": {"source": "学生手册", "category": "policy"}},
                {"content": "研究生采用等级制：优秀/良好/合格/不合格，不计入GPA。", "score": 0.65, "metadata": {"source": "研究生手册", "category": "policy"}},
            ],
        },
        {
            "user": "保研需要哪些条件",
            "semantic_type": "qualification_query",
            "docs": [
                {"content": "推荐免试研究生基本条件：①本校应届本科毕业生；②德智体全面发展，成绩优良；③全国大学生英语四级考试成绩425分以上；④无考试作弊或违纪行为。", "score": 0.91, "metadata": {"source": "教务处", "category": "qualification"}},
                {"content": "特殊学术专长保研：获得国家级学科竞赛一等奖以上，或在核心期刊发表学术论文者，经学院推荐可破格申请。", "score": 0.83, "metadata": {"source": "学生处", "category": "qualification"}},
                {"content": "各学院具体名额分配不同，成绩排名前15%有资格申请，具体以当年学院通知为准。", "score": 0.76, "metadata": {"source": "学院通知", "category": "qualification"}},
            ],
        },
        {
            "user": "图书馆开放时间",
            "semantic_type": "knowledge_query",
            "docs": [
                {"content": "图书馆开放时间：周一至周五 8:00-22:00，周末 9:00-17:00。法定节假日开放时间另行通知。", "score": 0.95, "metadata": {"source": "图书馆公告", "category": "knowledge"}},
                {"content": "自习室开放至深夜24:00，需刷校园卡进入。", "score": 0.72, "metadata": {"source": "图书馆官网", "category": "knowledge"}},
                {"content": "寒暑假期间图书馆实行轮开制度，具体安排见图书馆主页通知。", "score": 0.68, "metadata": {"source": "图书馆公告", "category": "knowledge"}},
            ],
        },
        {
            "user": "成绩复议流程",
            "semantic_type": "procedure_query",
            "docs": [
                {"content": "成绩复议申请流程：①在成绩公布后5个工作日内向开课学院教务办提交书面申请；②学院在10个工作日内组织阅卷教师复核；③结果通知学生。", "score": 0.90, "metadata": {"source": "教务处规定", "category": "procedure"}},
                {"content": "成绩复议仅限核查是否存在计分错误，不重新评分。每门课程只能申请一次复议。", "score": 0.79, "metadata": {"source": "学生手册", "category": "policy"}},
                {"content": "复议期间成绩暂不录入教务系统，待复核完成后正式录入。", "score": 0.67, "metadata": {"source": "教务处FAQ", "category": "procedure"}},
            ],
        },
    ]

    @classmethod
    def get_docs(cls, query: str, semantic_type: str | None = None) -> list[dict]:
        """根据 query 模糊匹配返回 mock 文档列表。"""
        # 简单字符串包含匹配
        for item in cls.LIBRARY_CARDS:
            if item["user"] in query or query in item["user"]:
                return item["docs"]
        # 尝试语义类型匹配
        if semantic_type:
            for item in cls.LIBRARY_CARDS:
                if item["semantic_type"] == semantic_type:
                    return item["docs"]
        # 默认返回第一条
        return cls.LIBRARY_CARDS[0]["docs"]


def retriever_node(state: AgenticRAGState) -> dict:
    """
    RetrieverNode：执行向量检索。

    Phase 1 使用 Mock 数据，不连接真实 Qdrant。
    真实版：query + reretrieve_params → Qdrant ANN 检索

    输入: state["query"], state["router_output"], state["reretrieve_params"]
    输出: state["retrieved_docs"]
    """
    query = state.get("query", "")
    router_output = state.get("router_output") or {}
    semantic_type = router_output.get("type")
    params = state.get("reretrieve_params", {"top_k_multiplier": 1.0, "filter_level": "strict", "query_expansion": False})

    docs = MockData.get_docs(query, semantic_type)
    logger.info(f"Retriever: query='{query}', retrieved {len(docs)} docs (mock, params={params})")

    return {"retrieved_docs": docs}
