import unittest

from research_budget import (
    choose_extended_providers,
    default_available_providers,
    budget_for_goal,
    normalize_budget,
)


class ResearchBudgetTests(unittest.TestCase):
    def test_budget_profiles_are_bounded(self):
        self.assertEqual(normalize_budget("fast").max_extended_providers, 1)
        self.assertEqual(normalize_budget("balanced").max_extended_providers, 2)
        self.assertEqual(normalize_budget("deep").max_extended_providers, 4)
        self.assertEqual(normalize_budget("unknown").name, "balanced")

    def test_goal_complexity_selects_deterministic_budget(self):
        self.assertEqual(budget_for_goal("research AI safety"), "balanced")
        self.assertEqual(budget_for_goal("systematic review of AI safety"), "deep")
        self.assertEqual(budget_for_goal("get current API status"), "fast")

    def test_free_only_defaults_to_public_providers(self):
        self.assertEqual(
            default_available_providers(),
            ("semantic_scholar", "europe_pmc"),
        )

    def test_provider_selection_is_deterministic_and_topic_aware(self):
        medical = choose_extended_providers(
            "clinical medicine cancer study",
            budget="balanced",
        )
        general = choose_extended_providers(
            "software architecture research",
            budget="balanced",
        )
        self.assertEqual(medical, choose_extended_providers(
            "clinical medicine cancer study",
            budget="balanced",
        ))
        self.assertEqual(len(medical), 2)
        self.assertEqual(len(general), 2)
        self.assertEqual(medical[0], "europe_pmc")


if __name__ == "__main__":
    unittest.main()
