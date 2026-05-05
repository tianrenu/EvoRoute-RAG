"""YAML skill file loader with error handling."""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TriggerConfig:
    """Trigger configuration for a skill."""

    keywords: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    semantic_type: str = ""
    question_type: str = ""
    confidence_boost: float = 0.1


@dataclass
class EvolutionConfig:
    """Evolution tracking data for a skill."""

    hit_count: int = 0
    success_rate: Optional[float] = None
    false_positive_count: int = 0
    depends_on: list[str] = field(default_factory=list)
    last_evolution_time: Optional[str] = None


@dataclass
class RetrievalConfig:
    """Retrieval action configuration."""

    boost_keywords: list[str] = field(default_factory=list)
    filter_metadata: dict = field(default_factory=dict)
    top_k: int = 5


@dataclass
class Skill:
    """A single skill loaded from YAML."""

    id: str
    name: str
    version: int
    status: str
    created_at: str
    trigger: TriggerConfig
    match_threshold: float
    answer_type: str
    answer_template: Optional[str]
    retrieval_config: Optional[RetrievalConfig]
    evolution: EvolutionConfig
    example_queries: list[str] = field(default_factory=list)
    # 同义词展开后的 keywords（用于 D_keyword 匹配计算）
    expanded_keywords: list[str] = field(default_factory=list)


def _parse_trigger(data: dict) -> TriggerConfig:
    """Parse trigger section from YAML data."""
    trigger_data = data.get("trigger", {})
    return TriggerConfig(
        keywords=trigger_data.get("keywords", []),
        aliases=trigger_data.get("aliases", []),
        semantic_type=trigger_data.get("semantic_type", ""),
        question_type=trigger_data.get("question_type", ""),
        confidence_boost=trigger_data.get("confidence_boost", 0.1),
    )


def _parse_evolution(data: dict) -> EvolutionConfig:
    """Parse evolution section from YAML data."""
    evo_data = data.get("evolution", {})
    return EvolutionConfig(
        hit_count=evo_data.get("hit_count", 0),
        success_rate=evo_data.get("success_rate"),
        false_positive_count=evo_data.get("false_positive_count", 0),
        depends_on=evo_data.get("depends_on", []),
        last_evolution_time=evo_data.get("last_evolution_time"),
    )


def _parse_retrieval(data: dict) -> Optional[RetrievalConfig]:
    """Parse retrieval config from action section."""
    action_data = data.get("action", {})
    retrieval_data = action_data.get("retrieval")
    if not retrieval_data:
        return None
    return RetrievalConfig(
        boost_keywords=retrieval_data.get("boost_keywords", []),
        filter_metadata=retrieval_data.get("filter_metadata", {}),
        top_k=retrieval_data.get("top_k", 5),
    )


def _parse_skill(data: dict) -> Skill:
    """Parse a single skill from YAML data."""
    semantic = data.get("semantic_profile", {})
    return Skill(
        id=data["id"],
        name=data["name"],
        version=data.get("version", 1),
        status=data.get("status", "active"),
        created_at=str(data.get("created_at", "")),
        trigger=_parse_trigger(data),
        match_threshold=data.get("match_threshold", 0.6),
        answer_type=data.get("answer_type", "template"),
        answer_template=data.get("answer_template"),
        retrieval_config=_parse_retrieval(data),
        evolution=_parse_evolution(data),
        example_queries=semantic.get("example_queries", []),
    )


# 全局同义词词典（lazy 加载）
_SYNONYM_DICT: dict[str, list[str]] = {}
_SYNONYM_DICT_LOADED: bool = False


def _load_synonym_dict(synonym_dict_path: str) -> None:
    """Load the global synonym dictionary from YAML.

    Supports two dict formats:
    - v1.0 flat:  { "学校": ["校园", "高校"] }
    - v1.1 structured: { "学校": { "synonyms": [...], "domain": "general", ... } }

    Internally normalized to: key -> list of synonyms (flat format).
    """
    global _SYNONYM_DICT, _SYNONYM_DICT_LOADED
    if _SYNONYM_DICT_LOADED:
        return
    try:
        with open(synonym_dict_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        raw_dict = data.get("dict", {}) if isinstance(data, dict) else {}
        # Normalize to flat format: key -> list of synonyms
        normalized: dict[str, list[str]] = {}
        for key, value in raw_dict.items():
            if isinstance(value, list):
                # v1.0 flat format
                normalized[key] = value
            elif isinstance(value, dict):
                # v1.1 structured format: extract synonyms list
                synonyms = value.get("synonyms", [])
                normalized[key] = synonyms
            else:
                logger.warning("Skipping invalid synonym entry for key '%s'", key)
        _SYNONYM_DICT = normalized
        logger.info("Loaded synonym dict with %d entries (v1.1 structured)", len(_SYNONYM_DICT))
    except Exception as e:
        logger.warning("Failed to load synonym dict from %s: %s", synonym_dict_path, e)
        _SYNONYM_DICT = {}
    _SYNONYM_DICT_LOADED = True


def _expand_keywords_with_synonyms(keywords: list[str]) -> list[str]:
    """Expand keywords using the synonym dictionary.

    For each keyword, if it (or a synonym) exists as a key in the dict,
    include both the keyword itself and all its synonyms.
    If not found, only include the keyword itself.

    Returns:
        Deduplicated list of expanded keywords.
    """
    expanded = set()
    for kw in keywords:
        expanded.add(kw)
        # 同义词展开：kw 本身是 key，或者 kw 是某个 key 的同义词
        # 情况1：kw 本身是 synonym_dict 的 key
        if kw in _SYNONYM_DICT:
            expanded.update(_SYNONYM_DICT[kw])
        # 情况2：kw 是 synonym_dict 中某个 key 的同义词（反向查找）
        for _key, _synonyms in _SYNONYM_DICT.items():
            if kw in _synonyms and kw != _key:
                expanded.add(_key)
                expanded.update(_synonyms)
    return list(expanded)


def load_skills(skill_library_path: str, synonym_dict_path: str = "") -> dict[str, Skill]:
    """Load all active skills from YAML files in the given directory.

    Args:
        skill_library_path: Path to the directory containing YAML skill files.
        synonym_dict_path: Path to the synonym dictionary YAML file.
            If provided, keywords will be expanded with synonyms before matching.

    Returns:
        Dict mapping skill_id to Skill for all active skills.
    """
    global _SYNONYM_DICT_LOADED
    _SYNONYM_DICT = {}  # 清空旧值，防止残留污染
    _SYNONYM_DICT_LOADED = False

    if synonym_dict_path:
        _load_synonym_dict(synonym_dict_path)

    skills: dict[str, Skill] = {}

    if not os.path.isdir(skill_library_path):
        logger.warning("Skill library path does not exist: %s", skill_library_path)
        return skills

    for filename in sorted(os.listdir(skill_library_path)):
        if not filename.endswith((".yaml", ".yml")):
            continue
        # Skip non-skill YAML files like synonym_dict
        if filename.startswith("synonym_dict"):
            continue

        filepath = os.path.join(skill_library_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.warning("Skipping malformed YAML file %s: %s", filename, e)
            continue
        except OSError as e:
            logger.warning("Cannot read file %s: %s", filename, e)
            continue

        if not isinstance(data, dict) or "id" not in data:
            logger.warning("Skipping invalid skill file %s: missing 'id'", filename)
            continue

        try:
            skill = _parse_skill(data)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Skipping skill file %s due to parse error: %s", filename, e)
            continue

        if skill.status != "active":
            logger.info("Skipping non-active skill %s (status=%s)", skill.id, skill.status)
            continue

        # 同义词展开：如果加载了同义词词典，则展开 keywords
        if _SYNONYM_DICT:
            skill.expanded_keywords = _expand_keywords_with_synonyms(skill.trigger.keywords)
            logger.debug("Skill %s expanded keywords: %s", skill.id, skill.expanded_keywords)

        skills[skill.id] = skill
        logger.debug("Loaded skill: %s", skill.id)

    logger.info("Loaded %d active skills from %s", len(skills), skill_library_path)
    return skills
