"""Tests for skill_scorer module."""

import pytest

from evoroute_rag.layer1.skill_loader import Skill, TriggerConfig, EvolutionConfig
from evoroute_rag.layer1.skill_scorer import (
    calc_skill_score,
    _calc_d_keyword,
    _calc_d_pattern,
    _calc_d_alias,
    _calc_d_hist,
    _calc_d_cold,
)


def _make_skill(
    keywords=None,
    aliases=None,
    confidence_boost=0.1,
    hit_count=50,
    success_rate=0.85,
    match_threshold=0.4,
):
    return Skill(
        id="test_001",
        name="Test Skill",
        version=1,
        status="active",
        created_at="2026-01-01",
        trigger=TriggerConfig(
            keywords=keywords or ["借书", "超期", "罚款", "归还", "借阅"],
            aliases=aliases or ["借阅图书", "图书归还"],
            confidence_boost=confidence_boost,
        ),
        match_threshold=match_threshold,
        answer_type="template",
        answer_template="test",
        retrieval_config=None,
        evolution=EvolutionConfig(hit_count=hit_count, success_rate=success_rate),
    )


class TestDKeyword:
    def test_partial_match(self):
        tokens = {"借书", "超期", "怎么办"}
        assert _calc_d_keyword(tokens, ["借书", "超期", "罚款", "归还", "借阅"]) == 0.4

    def test_full_match(self):
        tokens = {"借书", "超期", "罚款", "归还", "借阅"}
        assert _calc_d_keyword(tokens, ["借书", "超期", "罚款", "归还", "借阅"]) == 1.0

    def test_no_match(self):
        tokens = {"天气", "今天"}
        assert _calc_d_keyword(tokens, ["借书", "超期"]) == 0.0

    def test_empty_keywords(self):
        assert _calc_d_keyword({"test"}, []) == 0.0


class TestDPattern:
    def test_partial_match(self):
        skill = _make_skill()
        score = _calc_d_pattern(["借书", "超期"], skill)
        assert 0 < score < 1

    def test_no_match(self):
        skill = _make_skill()
        assert _calc_d_pattern(["天气"], skill) == 0.0

    def test_empty_patterns(self):
        skill = _make_skill(keywords=[], aliases=[])
        assert _calc_d_pattern(["test"], skill) == 0.0


class TestDAlias:
    def test_partial_match(self):
        assert _calc_d_alias({"借阅图书", "其他"}, ["借阅图书", "图书归还"]) == 0.5

    def test_no_match(self):
        assert _calc_d_alias({"天气"}, ["借阅图书", "图书归还"]) == 0.0

    def test_empty_aliases(self):
        assert _calc_d_alias({"test"}, []) == 0.0


class TestDHist:
    def test_normal(self):
        skill = _make_skill(success_rate=0.9)
        assert _calc_d_hist(skill) == 0.9

    def test_none_defaults_to_half(self):
        skill = _make_skill(success_rate=None)
        assert _calc_d_hist(skill) == 0.5

    def test_zero(self):
        skill = _make_skill(success_rate=0.0)
        assert _calc_d_hist(skill) == 0.0


class TestDCold:
    def test_cold_start(self):
        skill = _make_skill(hit_count=5)
        assert _calc_d_cold(skill) == 0.8

    def test_warm(self):
        skill = _make_skill(hit_count=50)
        assert _calc_d_cold(skill) == 1.0

    def test_boundary(self):
        skill = _make_skill(hit_count=10)
        assert _calc_d_cold(skill) == 1.0


class TestCalcSkillScore:
    def test_returns_score_and_breakdown(self):
        skill = _make_skill()
        score, breakdown = calc_skill_score("借书超期了怎么办", skill, ["借书", "超期"])
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert "D_keyword" in breakdown
        assert "D_pattern" in breakdown
        assert "D_semtype" in breakdown
        assert "D_alias" in breakdown
        assert "D_hist" in breakdown
        assert "D_cold" in breakdown

    def test_cold_start_penalty(self):
        warm = _make_skill(hit_count=50)
        cold = _make_skill(hit_count=5)
        s_warm, _ = calc_skill_score("借书超期", warm, ["借书", "超期"])
        s_cold, _ = calc_skill_score("借书超期", cold, ["借书", "超期"])
        assert s_warm > s_cold

    def test_no_match_low_score(self):
        skill = _make_skill()
        score, _ = calc_skill_score("今天天气怎么样", skill, [])
        assert score < 0.3
