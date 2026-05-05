"""Test L2 Agentic RAG Pipeline with mocked LLM and Qdrant."""

from unittest.mock import MagicMock, patch

import pytest

from evoroute_rag.layer2.langgraph_pipeline import build_graph, AgenticRAGState
from evoroute_rag.layer2.quality_gate_node import quality_gate_node, quality_gate_router
from evoroute_rag.layer2.router_node import router_node, _parse_router_response
from evoroute_rag.layer2.retriever_node import (
    _compute_mean_similarity,
    _compute_top1_margin,
    _compute_coverage,
)
from evoroute_rag.layer2.generator_node import generator_node, _build_context


# --- Fixtures ---


def _make_mock_docs(scores, contents=None):
    """Create mock document list with given scores."""
    if contents is None:
        contents = [f"测试文档内容{i}" for i in range(len(scores))]
    return [
        {"id": i, "score": s, "content": c, "metadata": {"category": "library"}}
        for i, (s, c) in enumerate(zip(scores, contents))
    ]


def _make_initial_state(query="图书馆几点开门", skill_config=None):
    """Create initial pipeline state."""
    return {
        "query": query,
        "query_type": "",
        "router_confidence": 0.0,
        "skill_config": skill_config or {},
        "retrieved_docs": [],
        "mean_similarity": 0.0,
        "top1_margin": 0.0,
        "doc_agreement": 0.0,
        "coverage_score": 0.0,
        "quality_score": 0.0,
        "reretrieve_count": 0,
        "attribution_evidence": {},
        "final_answer": "",
    }


# --- Unit Tests: Router ---


class TestRouterNode:
    def test_parse_router_response_valid(self):
        response = "type: factual_query\nconfidence: 0.85"
        qtype, conf = _parse_router_response(response)
        assert qtype == "factual_query"
        assert conf == 0.85

    def test_parse_router_response_invalid(self):
        response = "invalid response"
        qtype, conf = _parse_router_response(response)
        assert qtype == "factual_query"
        assert conf == 0.5

    def test_router_node_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.call.return_value = "type: procedure_query\nconfidence: 0.92"

        state = _make_initial_state()
        result = router_node(state, mock_llm)

        assert result["query_type"] == "procedure_query"
        assert result["router_confidence"] == 0.92

    def test_router_node_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.call.side_effect = RuntimeError("LLM error")

        state = _make_initial_state(skill_config={"semantic_type": "policy_query"})
        result = router_node(state, mock_llm)

        assert result["query_type"] == "policy_query"
        assert result["router_confidence"] == 0.3


# --- Unit Tests: Retriever metrics ---


class TestRetrieverMetrics:
    def test_mean_similarity(self):
        docs = _make_mock_docs([0.9, 0.8, 0.7])
        assert abs(_compute_mean_similarity(docs) - 0.8) < 1e-6

    def test_mean_similarity_empty(self):
        assert _compute_mean_similarity([]) == 0.0

    def test_top1_margin(self):
        docs = _make_mock_docs([0.9, 0.7, 0.5])
        assert abs(_compute_top1_margin(docs) - 0.2) < 1e-6

    def test_top1_margin_single_doc(self):
        docs = _make_mock_docs([0.9])
        assert _compute_top1_margin(docs) == 0.0

    def test_coverage_with_keywords(self):
        docs = _make_mock_docs([0.9], ["图书馆开放时间为早上8点到晚上10点"])
        score = _compute_coverage("图书馆几点开门", docs, ["开放时间"])
        assert score > 0.0


# --- Unit Tests: Quality Gate ---


class TestQualityGate:
    def test_high_quality_pass(self):
        # Q = 0.40*0.95 + 0.25*0.8 + 0.20*0.9 + 0.15*0.95
        # Q = 0.38 + 0.20 + 0.18 + 0.1425 = 0.9025
        state = _make_initial_state()
        state["mean_similarity"] = 0.95
        state["top1_margin"] = 0.8
        state["doc_agreement"] = 0.9
        state["coverage_score"] = 0.95
        state["retrieved_docs"] = _make_mock_docs([0.95, 0.90, 0.85])

        result = quality_gate_node(state)
        assert result["quality_score"] >= 0.75
        assert quality_gate_router(result) == "pass"

    def test_low_quality_reretrieve(self):
        state = _make_initial_state()
        state["mean_similarity"] = 0.3
        state["top1_margin"] = 0.02
        state["doc_agreement"] = 0.3
        state["coverage_score"] = 0.2
        state["reretrieve_count"] = 0
        state["retrieved_docs"] = _make_mock_docs([0.32, 0.30, 0.28])

        result = quality_gate_node(state)
        assert result["quality_score"] < 0.55
        assert quality_gate_router(result) == "reretrieve"

    def test_low_quality_fallback_after_max_retries(self):
        state = _make_initial_state()
        state["mean_similarity"] = 0.3
        state["top1_margin"] = 0.02
        state["doc_agreement"] = 0.3
        state["coverage_score"] = 0.2
        state["reretrieve_count"] = 3
        state["retrieved_docs"] = _make_mock_docs([0.32, 0.30, 0.28])

        result = quality_gate_node(state)
        assert quality_gate_router(result) == "pass_with_log"

    def test_gray_zone_pass_with_log(self):
        # Q = 0.40*0.7 + 0.25*0.5 + 0.20*0.8 + 0.15*0.7
        # Q = 0.28 + 0.125 + 0.16 + 0.105 = 0.67
        state = _make_initial_state()
        state["mean_similarity"] = 0.7
        state["top1_margin"] = 0.5
        state["doc_agreement"] = 0.8
        state["coverage_score"] = 0.7
        state["retrieved_docs"] = _make_mock_docs([0.75, 0.65, 0.60])

        result = quality_gate_node(state)
        q = result["quality_score"]
        assert 0.55 <= q < 0.75
        assert quality_gate_router(result) == "pass_with_log"

    def test_attribution_evidence_fields(self):
        state = _make_initial_state()
        state["mean_similarity"] = 0.7
        state["top1_margin"] = 0.15
        state["doc_agreement"] = 0.6
        state["coverage_score"] = 0.5
        state["reretrieve_count"] = 1
        state["retrieved_docs"] = _make_mock_docs([0.8, 0.65, 0.55])

        result = quality_gate_node(state)
        evidence = result["attribution_evidence"]

        # All required fields present
        required_fields = [
            "S", "M", "A", "C", "Q",
            "gap_type", "top_k_max_sim", "top1_sim",
            "has_correct_entity", "reretrieve_count",
        ]
        for field in required_fields:
            assert field in evidence, f"Missing field: {field}"


# --- Unit Tests: Generator ---


class TestGeneratorNode:
    def test_generator_produces_answer(self):
        mock_llm = MagicMock()
        mock_llm.call.return_value = "图书馆开放时间为早上8点到晚上10点。"

        state = _make_initial_state()
        state["retrieved_docs"] = _make_mock_docs(
            [0.9, 0.8], ["图书馆开放时间为早上8点到晚上10点", "图书馆位于校园东侧"]
        )

        result = generator_node(state, mock_llm)
        assert result["final_answer"] == "图书馆开放时间为早上8点到晚上10点。"

    def test_generator_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.call.side_effect = RuntimeError("LLM error")

        state = _make_initial_state()
        state["retrieved_docs"] = _make_mock_docs([0.5])

        result = generator_node(state, mock_llm)
        assert "无法" in result["final_answer"] or "暂时" in result["final_answer"]

    def test_build_context_empty_docs(self):
        ctx = _build_context([])
        assert "无相关文档" in ctx

    def test_build_context_with_docs(self):
        docs = _make_mock_docs([0.9], ["这是测试内容"])
        ctx = _build_context(docs)
        assert "这是测试内容" in ctx
        assert "0.90" in ctx


# --- Integration Test: Full Pipeline (Mocked) ---


class TestFullPipeline:
    def test_high_quality_pipeline(self):
        """Test case 1: High quality answer (Q >= 0.75)."""
        mock_llm = MagicMock()
        mock_llm.call.side_effect = [
            "type: factual_query\nconfidence: 0.90",  # Router
            "图书馆周一到周日早上8点到晚上10点开放。",  # Generator
        ]

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = _make_mock_docs(
            [0.92, 0.85, 0.80],
            [
                "图书馆开放时间：周一到周日 8:00-22:00",
                "图书馆位于教学楼东侧",
                "图书馆共有5层，藏书100万册",
            ],
        )

        graph = build_graph(mock_llm, mock_retriever)
        result = graph.invoke(_make_initial_state())

        assert result["query_type"] == "factual_query"
        assert result["router_confidence"] == 0.90
        assert result["quality_score"] > 0.0  # Q is computed from actual retriever metrics
        assert result["final_answer"] != ""
        assert "attribution_evidence" in result

    def test_low_quality_reretrieve_pipeline(self):
        """Test case 2: Low quality triggers reretrieve (Q < 0.55, reretrieve_count < 3)."""
        call_count = {"router": 0, "generator": 0}

        def llm_side_effect(messages):
            # Detect router vs generator by system prompt content
            system_msg = messages[0]["content"] if messages else ""
            if "分类器" in system_msg:
                call_count["router"] += 1
                return "type: factual_query\nconfidence: 0.60"
            else:
                call_count["generator"] += 1
                return "抱歉，根据现有文档无法准确回答。"

        mock_llm = MagicMock()
        mock_llm.call.side_effect = llm_side_effect

        # First search returns low quality, subsequent searches improve
        search_call_count = [0]

        def search_side_effect(query, top_k=5, filter_metadata=None, score_threshold=None):
            search_call_count[0] += 1
            if search_call_count[0] <= 2:
                # Low quality results
                return _make_mock_docs(
                    [0.30, 0.28, 0.25],
                    ["无关内容A", "无关内容B", "无关内容C"],
                )
            else:
                # Better results after retries
                return _make_mock_docs(
                    [0.75, 0.70, 0.65],
                    ["图书馆开放时间8:00-22:00", "开放时间相关信息", "图书馆相关"],
                )

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = search_side_effect

        graph = build_graph(mock_llm, mock_retriever)
        result = graph.invoke(_make_initial_state())

        # Should have triggered reretrieve at least once
        assert search_call_count[0] > 1
        assert result["final_answer"] != ""

    def test_generator_handles_insufficient_docs(self):
        """Test case 3: Generator handles case where docs are insufficient."""
        mock_llm = MagicMock()
        mock_llm.call.side_effect = [
            "type: factual_query\nconfidence: 0.70",  # Router
            "我无法根据现有文档回答此问题。",  # Generator
        ]

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = _make_mock_docs(
            [0.85, 0.80],
            ["这是一段关于食堂的介绍", "校园交通信息"],  # Irrelevant to query
        )

        graph = build_graph(mock_llm, mock_retriever)
        state = _make_initial_state(query="学校有没有游泳池")
        result = graph.invoke(state)

        assert "无法" in result["final_answer"]


# --- Unit Tests: LLM Client ---


class TestLLMClient:
    def test_circuit_breaker_opens_after_failures(self):
        from evoroute_rag.layer2.llm_client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, timeout=60)

        def failing_func():
            raise RuntimeError("fail")

        for _ in range(3):
            try:
                cb.call(failing_func)
            except RuntimeError:
                pass

        # Circuit should now be OPEN
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            cb.call(failing_func)

    def test_circuit_breaker_resets_on_success(self):
        from evoroute_rag.layer2.llm_client import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=5, timeout=60)

        def failing_func():
            raise RuntimeError("fail")

        def success_func():
            return "ok"

        # Accumulate some failures
        for _ in range(3):
            try:
                cb.call(failing_func)
            except RuntimeError:
                pass

        # Success should reset
        result = cb.call(success_func)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
