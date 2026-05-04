"""Aho-Corasick multi-pattern matcher wrapping pyahocorasick."""

from typing import Optional

import ahocorasick

from evoroute_rag.layer1.skill_loader import Skill


class AhoCorasickMatcher:
    """Builds and queries an Aho-Corasick automaton over skill keywords and aliases."""

    def __init__(self) -> None:
        self._automaton: Optional[ahocorasick.Automaton] = None

    def build(self, skills: dict[str, Skill]) -> None:
        """Build the automaton from all active skills' keywords, aliases, and synonyms.

        Uses expanded_keywords (which includes synonym-expanded keywords) so that
        AC matching covers both original keywords and their synonyms.

        Args:
            skills: Dict mapping skill_id to Skill.
        """
        automaton = ahocorasick.Automaton()

        for skill_id, skill in skills.items():
            # Use expanded_keywords (includes original keywords + synonyms)
            keywords = skill.expanded_keywords if skill.expanded_keywords else skill.trigger.keywords
            for keyword in keywords:
                automaton.add_word(keyword, (skill_id, keyword))
            for alias in skill.trigger.aliases:
                automaton.add_word(alias, (skill_id, alias))

        if len(automaton) > 0:
            automaton.make_automaton()
        self._automaton = automaton

    def search(self, query: str) -> dict[str, list[str]]:
        """Search the query against the automaton.

        Args:
            query: User query string.

        Returns:
            Dict mapping skill_id to list of matched keywords/aliases.
        """
        if self._automaton is None or len(self._automaton) == 0:
            return {}

        matches: dict[str, list[str]] = {}
        for _, (skill_id, keyword) in self._automaton.iter(query):
            matches.setdefault(skill_id, [])
            if keyword not in matches[skill_id]:
                matches[skill_id].append(keyword)

        return matches
