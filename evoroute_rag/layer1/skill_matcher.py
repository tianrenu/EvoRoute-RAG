"""Skill matching engine — main entry point for L1 routing."""

import logging
from dataclasses import dataclass
from typing import Optional

import jieba

from evoroute_rag.layer1.aho_corasick_matcher import AhoCorasickMatcher
from evoroute_rag.layer1.skill_loader import Skill, load_skills
from evoroute_rag.layer1.skill_scorer import calc_skill_score

logger = logging.getLogger(__name__)


@dataclass
class SkillMatchResult:
    """Result of a skill match."""

    skill_id: str
    skill_name: str
    answer_type: str
    answer_template: Optional[str]
    retrieval_config: Optional[dict]
    skill_score: float
    score_breakdown: dict
    matched_by: str


class SkillMatcher:
    """L1 skill matching engine using Dict + Aho-Corasick + SkillScore."""

    def __init__(self, skill_library_path: str, synonym_dict_path: str = "") -> None:
        """Load skills and build matching structures.

        Args:
            skill_library_path: Path to the directory containing YAML skill files.
            synonym_dict_path: Path to the synonym dictionary YAML file.
                If provided, keywords will be expanded with synonyms.
        """
        self.skills: dict[str, Skill] = load_skills(skill_library_path, synonym_dict_path)
        self._keyword_index: dict[str, str] = {}
        self._ac_matcher = AhoCorasickMatcher()
        self._build_keyword_dict()
        self._ac_matcher.build(self.skills)

    def _build_keyword_dict(self) -> None:
        """Build indexes for Dict exact matching.

        Two-tier Dict match:
          1. example_queries exact string match (highest confidence)
          2. token-based keyword match using expanded keywords (with synonyms)

        Note: expanded_keywords already includes original keywords + synonyms,
        so indexing expanded_keywords automatically covers synonym tokens.
        """
        self._example_query_index: dict[str, str] = {}
        for skill_id, skill in self.skills.items():
            for eq in skill.example_queries:
                if eq not in self._example_query_index:
                    self._example_query_index[eq] = skill_id
            # Use expanded_keywords (includes synonyms) for Dict tier-2 matching
            keywords_to_index = skill.expanded_keywords if skill.expanded_keywords else skill.trigger.keywords
            for kw in keywords_to_index:
                if kw not in self._keyword_index:
                    self._keyword_index[kw] = skill_id

    def _build_result(
        self, skill: Skill, score: float, breakdown: dict, matched_by: str
    ) -> SkillMatchResult:
        """Build a SkillMatchResult from a matched skill."""
        retrieval_dict = None
        if skill.retrieval_config:
            retrieval_dict = {
                "retrieval": {
                    "boost_keywords": skill.retrieval_config.boost_keywords,
                    "filter_metadata": skill.retrieval_config.filter_metadata,
                    "top_k": skill.retrieval_config.top_k,
                }
            }

        return SkillMatchResult(
            skill_id=skill.id,
            skill_name=skill.name,
            answer_type=skill.answer_type,
            answer_template=skill.answer_template if skill.answer_type == "template" else None,
            retrieval_config=retrieval_dict if skill.answer_type == "directive" else retrieval_dict,
            skill_score=score,
            score_breakdown=breakdown,
            matched_by=matched_by,
        )

    def _dict_match(self, query: str) -> Optional[SkillMatchResult]:
        """Attempt Dict exact match.

        Two-tier strategy:
          1. Check if query string exactly matches a skill's example_query
          2. Check if any jieba token exactly equals a keyword

        Tier 1 catches known high-confidence patterns. Tier 2 catches
        single-keyword queries like "借书" or "选课".

        Args:
            query: User query string.

        Returns:
            SkillMatchResult if matched, else None.
        """
        # Tier 1: example_query exact string match
        if query in self._example_query_index:
            skill_id = self._example_query_index[query]
            skill = self.skills[skill_id]
            breakdown = self._dict_breakdown(skill)
            return self._build_result(skill, 1.0, breakdown, "dict")

        # Tier 2: single-token keyword match (only for short queries
        # where a single keyword covers the query's intent)
        tokens = [t for t in jieba.lcut(query) if len(t) >= 2]
        if len(tokens) <= 2:
            for token in tokens:
                if token in self._keyword_index:
                    skill_id = self._keyword_index[token]
                    skill = self.skills[skill_id]
                    breakdown = self._dict_breakdown(skill)
                    return self._build_result(skill, 1.0, breakdown, "dict")
        return None

    def _dict_breakdown(self, skill: Skill) -> dict:
        """Build a score breakdown dict for a Dict match."""
        return {
            "D_keyword": 1.0,
            "D_pattern": 1.0,
            "D_semtype": 0.5,
            "D_alias": 0.0,
            "D_hist": skill.evolution.success_rate if skill.evolution.success_rate is not None else 0.5,
            "D_cold": 0.8 if skill.evolution.hit_count < 10 else 1.0,
            "base_score": 1.0,
            "final_score": 1.0,
        }

    def match(self, query: str) -> Optional[SkillMatchResult]:
        """Match a query against the skill library.

        Flow:
            1. Dict exact match (highest priority)
            2. Aho-Corasick multi-pattern match
            3. SkillScore scoring for each AC candidate
            4. Return highest-scoring skill above threshold

        Args:
            query: User query string.

        Returns:
            SkillMatchResult if matched, None if no skill qualifies (→ L2).
        """
        dict_result = self._dict_match(query)
        if dict_result:
            return dict_result

        ac_matches = self._ac_matcher.search(query)
        if not ac_matches:
            return None

        candidates: list[tuple[Skill, float, dict]] = []
        for skill_id, matched_keywords in ac_matches.items():
            skill = self.skills[skill_id]
            score, breakdown = calc_skill_score(query, skill, matched_keywords)
            if score >= skill.match_threshold:
                candidates.append((skill, score, breakdown))

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x[1])
        return self._build_result(best[0], best[1], best[2], "aho_corasick")
