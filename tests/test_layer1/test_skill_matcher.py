"""Tests for skill_matcher module (integration)."""

import pytest

from evoroute_rag.layer1.skill_matcher import SkillMatcher, SkillMatchResult


@pytest.fixture(scope="module")
def matcher():
    return SkillMatcher("skills/")


class TestSkillLoading:
    def test_loads_all_active_skills(self, matcher):
        assert len(matcher.skills) == 10

    def test_all_skills_are_active(self, matcher):
        for skill in matcher.skills.values():
            assert skill.status == "active"


class TestDictMatch:
    def test_example_query_exact_match(self, matcher):
        result = matcher.match("借书超期了怎么办")
        assert result is not None
        assert result.skill_id == "library_overdue_001"
        assert result.matched_by == "dict"
        assert result.skill_score == 1.0

    def test_short_keyword_query(self, matcher):
        result = matcher.match("选课")
        assert result is not None
        assert result.skill_id == "course_add_drop_001"
        assert result.matched_by == "dict"


class TestAhoCorasickMatch:
    def test_partial_match_with_scoring(self, matcher):
        result = matcher.match("超期罚款怎么交")
        assert result is not None
        assert result.skill_id == "library_overdue_001"
        assert result.matched_by == "aho_corasick"
        assert 0 < result.skill_score < 1

    def test_score_breakdown_has_all_dimensions(self, matcher):
        result = matcher.match("超期罚款怎么交")
        bd = result.score_breakdown
        for key in ["D_keyword", "D_pattern", "D_semtype", "D_alias", "D_hist", "D_cold"]:
            assert key in bd
            assert isinstance(bd[key], float)


class TestNoMatch:
    def test_irrelevant_query(self, matcher):
        assert matcher.match("今天天气怎么样") is None

    def test_empty_query(self, matcher):
        assert matcher.match("") is None


class TestDirectiveSkill:
    def test_directive_returns_retrieval_config(self, matcher):
        result = matcher.match("保研需要什么条件")
        assert result is not None
        assert result.answer_type == "directive"
        assert result.retrieval_config is not None
        assert "boost_keywords" in result.retrieval_config

    def test_template_returns_answer(self, matcher):
        result = matcher.match("借书超期了怎么办")
        assert result.answer_type == "template"
        assert result.answer_template is not None


class TestPerformance:
    def test_response_under_10ms(self, matcher):
        import time
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            matcher.match("借书超期了怎么办")
            times.append((time.perf_counter() - t0) * 1000)
        assert sum(times) / len(times) < 10
