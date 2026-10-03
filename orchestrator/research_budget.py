#!/usr/bin/env python3
"""Deterministic, zero-dollar research provider budgeting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchBudget:
    name: str
    target_independent_sources: int
    max_extended_providers: int
    max_results: int
    stage_size: int = 2


BUDGETS = {
    "fast": ResearchBudget("fast", 4, 1, 5),
    "balanced": ResearchBudget("balanced", 6, 2, 6),
    "deep": ResearchBudget("deep", 8, 4, 8),
}

DEFAULT_BUDGET = "balanced"

_GENERAL_ORDER = (
    "semantic_scholar",
    "openalex",
    "europe_pmc",
    "core",
)

_TOPIC_HINTS = {
    "europe_pmc": (
        "medical", "medicine", "clinical", "health", "disease", "patient",
        "drug", "pharma", "biomedical", "cancer", "genetic", "genetics",
        "neurology", "psychiatry", "biology",
    ),
    "semantic_scholar": (
        "research", "paper", "literature", "study", "science", "computer",
        "software", "ai", "artificial intelligence", "machine learning",
        "llm", "algorithm", "benchmark",
    ),
    "openalex": (
        "research", "compare", "literature", "study", "academic", "paper",
        "science", "history", "economics", "social", "education",
    ),
    "core": (
        "open access", "full text", "repository", "thesis", "dissertation",
    ),
}


def normalize_budget(name: str | None) -> ResearchBudget:
    key = str(name or DEFAULT_BUDGET).strip().lower()
    return BUDGETS.get(key, BUDGETS[DEFAULT_BUDGET])


def choose_extended_providers(
    query: str,
    *,
    budget: str | ResearchBudget = DEFAULT_BUDGET,
    available: tuple[str, ...] = _GENERAL_ORDER,
) -> list[str]:
    selected_budget = budget if isinstance(budget, ResearchBudget) else normalize_budget(budget)
    query_text = str(query or "").strip().lower()
    available_set = {str(item).strip().lower() for item in available if str(item).strip()}
    scored: list[tuple[int, int, str]] = []
    for index, provider in enumerate(_GENERAL_ORDER):
        if provider not in available_set:
            continue
        score = sum(2 if " " in hint else 1 for hint in _TOPIC_HINTS.get(provider, ()) if hint in query_text)
        scored.append((-score, index, provider))
    scored.sort()
    return [provider for _, _, provider in scored[: selected_budget.max_extended_providers]]


def budget_metadata(budget: str | ResearchBudget) -> dict[str, int | str]:
    selected = budget if isinstance(budget, ResearchBudget) else normalize_budget(budget)
    return {
        "name": selected.name,
        "target_independent_sources": selected.target_independent_sources,
        "max_extended_providers": selected.max_extended_providers,
        "max_results": selected.max_results,
        "stage_size": selected.stage_size,
    }


__all__ = [
    "BUDGETS",
    "ResearchBudget",
    "DEFAULT_BUDGET",
    "normalize_budget",
    "choose_extended_providers",
    "budget_metadata",
]
