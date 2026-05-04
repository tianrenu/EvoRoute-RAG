"""Tests for skill_loader module."""

import os
import tempfile

import pytest
import yaml

from evoroute_rag.layer1.skill_loader import load_skills, Skill


@pytest.fixture
def skill_dir(tmp_path):
    """Create a temporary skill directory with test YAML files."""
    active_skill = {
        "id": "test_active_001",
        "name": "Test Active Skill",
        "version": 1,
        "status": "active",
        "created_at": "2026-01-01",
        "trigger": {
            "keywords": ["测试", "关键词"],
            "aliases": ["测试别名"],
            "semantic_type": "fact_query",
            "question_type": "事实查询类",
            "confidence_boost": 0.1,
        },
        "semantic_profile": {
            "example_queries": ["这是测试查询"],
        },
        "match_threshold": 0.6,
        "answer_type": "template",
        "answer_template": "这是测试答案",
        "action": {
            "retrieval": {
                "boost_keywords": ["测试"],
                "filter_metadata": {"category": "test"},
                "top_k": 3,
            }
        },
        "evolution": {
            "hit_count": 20,
            "success_rate": 0.9,
            "false_positive_count": 1,
            "depends_on": [],
            "last_evolution_time": None,
        },
    }

    dormant_skill = dict(active_skill)
    dormant_skill["id"] = "test_dormant_001"
    dormant_skill["status"] = "dormant"

    with open(tmp_path / "active.yaml", "w", encoding="utf-8") as f:
        yaml.dump(active_skill, f, allow_unicode=True)

    with open(tmp_path / "dormant.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dormant_skill, f, allow_unicode=True)

    # Malformed YAML
    with open(tmp_path / "bad.yaml", "w", encoding="utf-8") as f:
        f.write("{{invalid yaml content::")

    # Missing id field
    with open(tmp_path / "no_id.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"name": "no id"}, f)

    return str(tmp_path)


def test_load_active_skills_only(skill_dir):
    skills = load_skills(skill_dir)
    assert "test_active_001" in skills
    assert "test_dormant_001" not in skills


def test_load_skill_fields(skill_dir):
    skills = load_skills(skill_dir)
    skill = skills["test_active_001"]
    assert skill.name == "Test Active Skill"
    assert skill.trigger.keywords == ["测试", "关键词"]
    assert skill.trigger.aliases == ["测试别名"]
    assert skill.trigger.confidence_boost == 0.1
    assert skill.match_threshold == 0.6
    assert skill.answer_type == "template"
    assert skill.evolution.hit_count == 20
    assert skill.evolution.success_rate == 0.9
    assert skill.example_queries == ["这是测试查询"]


def test_load_handles_malformed_yaml(skill_dir):
    skills = load_skills(skill_dir)
    assert len(skills) == 1  # Only active skill loaded


def test_load_nonexistent_directory():
    skills = load_skills("/nonexistent/path")
    assert skills == {}


def test_load_retrieval_config(skill_dir):
    skills = load_skills(skill_dir)
    skill = skills["test_active_001"]
    assert skill.retrieval_config is not None
    assert skill.retrieval_config.boost_keywords == ["测试"]
    assert skill.retrieval_config.top_k == 3


def test_load_real_skills():
    skills = load_skills("skills/")
    assert len(skills) == 10
    assert "library_overdue_001" in skills
    assert "baoyan_qualification_001" in skills
