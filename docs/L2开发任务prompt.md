# L2 Agentic RAG 代码生成 Prompt

## 项目根目录
`~/projects/dachuang_project/EvoRoute-RAG/`

## 设计书（必须先读）
`~/projects/dachuang_project/EvoRoute-RAG/docs/架构设计说明书.md`
**重点读：第三章 L2 部分（3.1 ~ 3.7）**

## MiniMax API 配置
```bash
export OPENAI_API_KEY="sk-cp-m17lGHfK7xmrT-mO2wL9UHsfVKvddVuva6Lrg9vLvwI5r-bEAcYC2roi6668s51Rv4c20qAyj6m65lWjlsPKe2183IN7koI_amBPJS40SGsIOXxUUEh46Tc"
export LLM_API_BASE="https://api.minimaxi.com/v1"
```

## 任务
根据设计书生成 L2 Agentic RAG 完整代码，写入 `evoroute_rag/layer2/` 目录。

## L2 规格要点（来自设计书）

### 技术栈
- **状态图编排**：LangGraph
- **向量检索**：Qdrant（OpenAI兼容embedding，512维）
- **LLM**：MiniMax API（OpenAI兼容格式）
- **Python**：>= 3.10

### LangGraph 状态定义
```python
class AgenticRAGState(TypedDict):
    query: str
    query_type: str
    router_confidence: float
    skill_config: dict          # L1传入的检索配置
    retrieved_docs: List[dict]
    mean_similarity: float       # S
    top1_margin: float          # M
    doc_agreement: float        # A
    coverage_score: float        # C
    quality_score: float         # Q 综合质量分数
    reretrieve_count: int
    attribution_evidence: dict
    final_answer: str
```

### 四个节点（必须全部实现）

#### 1. Router 节点
```python
def router_node(state: AgenticRAGState) -> AgenticRAGState:
    # LLM zero-shot分类：query_type ∈ {factual_query, procedure_query, policy_query, ...}
    # 输出router_confidence（0~1）
    # 如有L1 skill_config，读取其semantic_type做参考
```

#### 2. Retriever 节点
```python
def retriever_node(state: AgenticRAGState) -> AgenticRAGState:
    # 读取 state["skill_config"] 中的 retrieval 配置（boost_keywords, filter_metadata, top_k）
    # 执行 Qdrant 检索（使用 MiniMax text-embedding-v1）
    # 计算 S, M, A, C 四维指标
    # 如 reretrieve_count > 0，调整 query 重检（可选）
```

#### 3. Quality Gate 节点
```python
def quality_gate_node(state: AgenticRAGState) -> AgenticRAGState:
    # Q = α·S + β·M + γ·A + δ·C
    # 权重：α=0.40, β=0.25, γ=0.20, δ=0.15
    # 决策：
    #   Q >= 0.75 → "pass"
    #   Q ∈ [0.55, 0.75) → "pass_with_log"
    #   Q < 0.55 and reretrieve_count < 3 → "reretrieve"
    #   Q < 0.55 and reretrieve_count >= 3 → "pass_with_log"（兜底）
    # 构建 attribution_evidence 字典（传给L3用）
```

#### 4. Generator 节点
```python
def generator_node(state: AgenticRAGState) -> AgenticRAGState:
    # 读取检索文档，LLM生成最终答案
    # 系统prompt需包含：你是校园知识问答助手，基于检索到的文档回答问题
    # 如果检索文档不足以回答，明确说明"我无法根据现有文档回答此问题"
```

### attribution_evidence 必须包含的字段
```python
{
    "S": float,                 # mean similarity
    "M": float,                 # top1 - top2
    "A": float,                 # doc agreement
    "C": float,                 # coverage score
    "Q": float,                 # 综合质量分数
    "gap_type": str,            # "keyword_missing" / "precision_dilution" / "doc_missing"
    "top_k_max_sim": float,     # top_k中最高相似度
    "top1_sim": float,          # top-1相似度
    "has_correct_entity": bool, # top-1是否含正确答案关键实体
    "reretrieve_count": int
}
```

### Qdrant 配置
```python
# Collection: campus_knowledge, 512维, COSINE距离
# Metadata字段: category, school_year, source, last_updated, school_id
# 检索时支持 top_k 和 score_threshold 过滤
```

### LLM 调用封装要求
- Circuit Breaker：连续5次失败熔断60秒
- 指数退避：1s → 2s → 4s
- 单次超时：30秒
- 使用 MiniMax 的 `abab6.5-chat` 模型（OpenAI兼容格式）

## 必须创建的文件

```
evoroute_rag/layer2/
├── __init__.py
├── llm_client.py              # MiniMax API封装（含CircuitBreaker+重试）
├── qdrant_client.py           # Qdrant检索封装
├── router_node.py            # Router节点
├── retriever_node.py         # Retriever节点
├── quality_gate_node.py       # Quality Gate节点
├── generator_node.py          # Generator节点
├── langgraph_pipeline.py     # LangGraph状态图编排
└── l2_pipeline.py             # L2统一入口

tests/
└── test_l2_pipeline.py        # 基础测试（mock LLM和Qdrant）
```

## Pipeline 入口设计
```python
class AgenticRAGPipeline:
    def __init__(self, qdrant_url: str, llm_config: dict):
        self.graph = self._build_graph()
    
    def run(self, query: str, skill_config: dict = None) -> dict:
        """
        Returns:
            {
                "final_answer": str,
                "query_type": str,
                "router_confidence": float,
                "quality_score": float,
                "attribution_evidence": dict,
                "retrieved_docs": List[dict]
            }
        """
```

## 测试要求
- Mock LLM 调用（不真实请求API）
- Mock Qdrant 检索（返回固定测试文档）
- 至少3个测试用例：
  1. 高质量答案（Q >= 0.75）
  2. 低质量重检索（Q < 0.55，reretrieve_count < 3）
  3. Generator处理无法回答的情况
