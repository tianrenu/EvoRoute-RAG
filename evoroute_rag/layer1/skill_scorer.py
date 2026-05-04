"""SkillScore six-dimension scoring engine."""

import jieba

from evoroute_rag.layer1.skill_loader import Skill, _SYNONYM_DICT

WEIGHTS = {
    "keyword": 0.25,
    "pattern": 0.20,
    "semtype": 0.20,
    "alias": 0.15,
    "hist": 0.10,
}

COLD_START_THRESHOLD = 10
COLD_START_PENALTY = 0.8
DEFAULT_SUCCESS_RATE = 0.5
DEFAULT_SEMTYPE_SCORE = 0.5


def _tokenize(query: str) -> set[str]:
    """Tokenize a Chinese query using jieba."""
    return set(jieba.lcut(query))


def _calc_d_keyword(query_tokens: set[str], skill: Skill | list[str]) -> float:
    """D_keyword: fraction of skill keywords hit by query tokens.

    Synonym expansion: if a keyword has synonyms in the global synonym dict,
    query tokens matching any synonym also count as a hit.
    The denominator stays as the ORIGINAL keyword count (no dilution).

    Accepts either a Skill object or a raw keywords list (tests only).
    """
    if isinstance(skill, list):
        keywords = skill
        synonyms_map = {}
    else:
        keywords = skill.trigger.keywords
        synonyms_map = _SYNONYM_DICT

    if not keywords:
        return 0.0

    hits = 0
    for kw in keywords:
        # Build the full set: keyword itself + its synonyms
        hit_set = {kw}
        if synonyms_map and kw in synonyms_map:
            hit_set.update(synonyms_map[kw])
        # Also check reverse: if kw is a synonym of another key, include the key
        if synonyms_map:
            for _key, _syns in synonyms_map.items():
                if kw in _syns and kw != _key:
                    hit_set.add(_key)
                    hit_set.update(_syns)
        if query_tokens & hit_set:
            hits += 1

    return hits / len(keywords)


def _calc_d_pattern(matched_keywords: list[str], skill: Skill) -> float:
    """D_pattern: normalized Aho-Corasick hit score.

    Denominator uses ORIGINAL keywords + aliases (not expanded) to avoid
    dilution. matched_keywords from AC are filtered against original
    patterns only, since expanded keywords should not inflate pattern score.
    """
    original_keywords = skill.trigger.keywords
    original_aliases = skill.trigger.aliases
    all_patterns = original_keywords + original_aliases
    if not all_patterns:
        return 0.0
    boost = skill.trigger.confidence_boost
    max_possible = boost * len(all_patterns)
    if max_possible == 0:
        return 0.0
    matched_set = set(all_patterns) & set(matched_keywords)
    matched_boost = boost * len(matched_set)
    return matched_boost / max_possible


def _calc_d_alias(query_tokens: set[str], aliases: list[str]) -> float:
    """D_alias: fraction of skill aliases hit by query tokens."""
    if not aliases:
        return 0.0
    hits = query_tokens & set(aliases)
    return len(hits) / len(aliases)


def _calc_d_hist(skill: Skill) -> float:
    """D_hist: historical success rate, default 0.5 for cold start."""
    rate = skill.evolution.success_rate
    if rate is None:
        return DEFAULT_SUCCESS_RATE
    return rate


def _calc_d_cold(skill: Skill) -> float:
    """D_cold: cold start penalty multiplier."""
    if skill.evolution.hit_count < COLD_START_THRESHOLD:
        return COLD_START_PENALTY
    return 1.0


def calc_skill_score(
    query: str, skill: Skill, matched_keywords: list[str]
) -> tuple[float, dict]:
    """Calculate the six-dimension SkillScore.

    Args:
        query: User query string.
        skill: The candidate skill.
        matched_keywords: Keywords matched by Aho-Corasick.

    Returns:
        Tuple of (final_score, breakdown_dict).
    """
    query_tokens = _tokenize(query)

    d_keyword = _calc_d_keyword(query_tokens, skill)
    d_pattern = _calc_d_pattern(matched_keywords, skill)
    d_semtype = DEFAULT_SEMTYPE_SCORE
    d_alias = _calc_d_alias(query_tokens, skill.trigger.aliases)
    d_hist = _calc_d_hist(skill)
    d_cold = _calc_d_cold(skill)

    weighted_sum = (
        WEIGHTS["keyword"] * d_keyword
        + WEIGHTS["pattern"] * d_pattern
        + WEIGHTS["semtype"] * d_semtype
        + WEIGHTS["alias"] * d_alias
        + WEIGHTS["hist"] * d_hist
    )
    total_weight = sum(WEIGHTS.values())
    base_score = weighted_sum / total_weight
    final_score = base_score * d_cold

    breakdown = {
        "D_keyword": round(d_keyword, 4),
        "D_pattern": round(d_pattern, 4),
        "D_semtype": round(d_semtype, 4),
        "D_alias": round(d_alias, 4),
        "D_hist": round(d_hist, 4),
        "D_cold": round(d_cold, 4),
        "base_score": round(base_score, 4),
        "final_score": round(final_score, 4),
    }

    return round(final_score, 4), breakdown
