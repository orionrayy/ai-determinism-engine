import unittest
from datetime import datetime, timezone
from scout import extract_money_mentions, render_markdown, score_issue

class OpportunityRadarTests(unittest.TestCase):
    def test_extracts_money_formats(self):
        items = extract_money_mentions("Reward: $100, US$1,200 and 75 USD.")
        self.assertIn("$100", items)
        self.assertIn("US$1,200", items)
        self.assertIn("75 USD", items)

    def test_ignores_issue_without_reward_signal(self):
        issue = {"title": "Improve Python tests", "body": "Add unit tests.",
                 "labels": [], "updated_at": "2026-10-19T00:00:00Z",
                 "html_url": "https://github.com/example/project/issues/1",
                 "repository_url": "https://api.github.com/repos/example/project"}
        self.assertIsNone(score_issue(issue, datetime(2026, 10, 19, tzinfo=timezone.utc)))

    def test_scores_paid_python_issue(self):
        issue = {"title": "Add Python regression tests for a $150 bounty",
                 "body": "Help wanted: improve test coverage and CI validation.",
                 "labels": [{"name": "bounty"}], "updated_at": "2026-10-18T00:00:00Z",
                 "html_url": "https://github.com/example/project/issues/2",
                 "repository_url": "https://api.github.com/repos/example/project"}
        result = score_issue(issue, datetime(2026, 10, 19, tzinfo=timezone.utc))
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["score"], 70)
        self.assertIn("$150", result["amount_mentions"])
        self.assertEqual(result["repo"], "example/project")

    def test_markdown_includes_uncertainty_and_goal_math(self):
        rendered = render_markdown({"generated_at": "2026-10-19T00:00:00+00:00",
                                    "github_issues": [], "manual_leads": [], "warnings": []})
        self.assertIn("not guaranteed earnings", rendered)
        self.assertIn("66.7 paid hours/month", rendered)
        self.assertIn("No qualifying GitHub issue", rendered)

if __name__ == "__main__":
    unittest.main()
