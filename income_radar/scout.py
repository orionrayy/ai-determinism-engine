#!/usr/bin/env python3
"""Free-first opportunity radar. Discover leads; never auto-apply or auto-submit."""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_ROOT = "https://api.github.com/search/issues"
QUERIES = (
    'is:issue is:open bounty language:Python',
    'is:issue is:open reward language:Python',
    'is:issue is:open "paid issue" language:Python',
)
SKILL_TERMS = (
    "python", "test", "testing", "documentation", "docs", "automation",
    "workflow", "github actions", "bug fix", "reproduc", "quality assurance",
    "qa", "evaluation", "prompt", "indonesian", "localization", "json",
    "api", "cli", "validation", "ci",
)
BOUNTY_TERMS = ("bounty", "reward", "paid issue", "cash prize", "payment")
MONEY_RE = re.compile(
    r"(?i)(?:US\$|\$)\s*(\d[\d,]*(?:\.\d{1,2})?)|"
    r"(\d[\d,]*(?:\.\d{1,2})?)\s*(?:USD|US dollars?)"
)
MANUAL_LEADS = [
    {
        "title": "Indonesian AI voice evaluator — Outlier",
        "kind": "Paid-task application",
        "url": "https://outlier.ai/languages/id-id",
        "payout_signal": "Official listing advertises up to US$7.50/hour; actual rate, task supply, screening and eligibility vary.",
        "fit": ["Indonesian", "AI evaluation", "language quality", "prompt writing"],
        "next_step": "Check the listing, eligibility, privacy terms and payment route; complete screening honestly.",
        "verification": "Official listing lead; application and acceptance have not been verified.",
    },
    {
        "title": "Indonesian ASR data collection — RWS TrainAI",
        "kind": "Paid-task application",
        "url": "https://app.tai.rws.com/opportunities",
        "payout_signal": "Official opportunity portal recently listed Indonesian ASR data collection at US$5/hour for Indonesia; verify it remains open.",
        "fit": ["Indonesian", "speech/audio", "data quality", "recording"],
        "next_step": "Open the portal, confirm the Indonesian role remains available, review recording and privacy requirements, then apply.",
        "verification": "Recent official listing lead; current availability and selection are not verified.",
    },
    {
        "title": "Indonesian Quality Assurance Rater — TELUS Digital",
        "kind": "Part-time AI-quality application",
        "url": "https://jobs.telusdigital.com/search/jobs/in/jakarta",
        "payout_signal": "Official careers search recently listed a remote, part-time Indonesian Quality Assurance Rater role; public search excerpt did not show pay.",
        "fit": ["Indonesian", "quality assurance", "critical thinking", "content evaluation"],
        "next_step": "Check the live vacancy, location eligibility, hours, compensation and assessment requirements.",
        "verification": "Official careers lead; rate, availability and acceptance are unverified.",
    },
    {
        "title": "OnlyDust open-source contributor opportunities",
        "kind": "Open-source contribution / grant lead",
        "url": "https://www.onlydust.com/",
        "payout_signal": "Some funded ecosystems reward contributors; not every project or contribution is paid.",
        "fit": ["Python", "testing", "documentation", "open-source"],
        "next_step": "Check current funded programs, eligibility and exact reward terms before starting work.",
        "verification": "Platform lead; task-specific reward must be verified.",
    },
    {
        "title": "Bugcrowd public bounty programs",
        "kind": "Security bounty lead",
        "url": "https://bugcrowd.com/hackers",
        "payout_signal": "Cash rewards apply only to eligible, unique, in-scope findings when a program offers payment.",
        "fit": ["reproducible testing", "careful documentation"],
        "next_step": "Read rules and written scope first; never test a system without explicit authorization.",
        "verification": "Platform lead; no reward is guaranteed.",
    },
    {
        "title": "HackerOne public bounty programs",
        "kind": "Security bounty lead",
        "url": "https://www.hackerone.com/hackers",
        "payout_signal": "Payment depends on an accepted valid finding and completed payout onboarding.",
        "fit": ["reproducible testing", "technical writing"],
        "next_step": "Review scope, eligibility and payment requirements before any research.",
        "verification": "Platform lead; no reward is guaranteed.",
    },
]
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None

def extract_money_mentions(text: str) -> list[str]:
    found: list[str] = []
    for match in MONEY_RE.finditer(text or ""):
        raw = " ".join(match.group(0).split())
        if raw not in found:
            found.append(raw)
    return found[:8]

def labels_for(issue: dict[str, Any]) -> list[str]:
    return [str(v.get("name", "")).lower() for v in issue.get("labels", []) if isinstance(v, dict)]

def score_issue(issue: dict[str, Any], now: datetime | None = None) -> dict[str, Any] | None:
    title, body = str(issue.get("title", "")), str(issue.get("body") or "")
    text = (title + "\n" + body).lower()
    labels = labels_for(issue)
    amounts = extract_money_mentions(title + "\n" + body)
    bounty_label = any(any(term in label for term in ("bounty", "reward", "paid")) for label in labels)
    bounty_words = any(term in text for term in BOUNTY_TERMS)
    if not amounts and not bounty_label and not bounty_words:
        return None
    score, reasons = 0, []
    if amounts:
        score += 40
        reasons.append("explicit money amount in issue text")
    if bounty_label:
        score += 25
        reasons.append("bounty/reward/paid label")
    elif bounty_words:
        score += 18
        reasons.append("bounty/reward wording; funding needs confirmation")
    matched = [term for term in SKILL_TERMS if term in text]
    score += min(24, 4 * len(matched))
    if matched:
        reasons.append("skill-fit signals: " + ", ".join(matched[:6]))
    updated = parse_date(issue.get("updated_at"))
    age = None if updated is None else max(0, ((now or utc_now()) - updated).days)
    if age is not None and age <= 30:
        score += 12
        reasons.append("updated within 30 days")
    elif age is not None and age <= 90:
        score += 6
        reasons.append("updated within 90 days")
    if "good first issue" in text or "beginner" in text or "help wanted" in labels:
        score += 4
        reasons.append("possible newcomer-friendly signal")
    repo = str(issue.get("repository_url", "")).replace("https://api.github.com/repos/", "")
    return {
        "title": title, "url": issue.get("html_url", ""), "repo": repo,
        "updated_at": issue.get("updated_at"), "labels": labels,
        "amount_mentions": amounts, "score": min(100, score),
        "matched_terms": matched, "reasons": reasons,
        "kind": "GitHub issue / possible bounty",
        "verification": "Unverified lead. Confirm funding, claim rules, eligibility, acceptance criteria and payout before investing effort.",
    }

def fetch_query(query: str, token: str | None, timeout: int) -> tuple[list[dict[str, Any]], str | None]:
    params = urlencode({"q": query, "sort": "updated", "order": "desc", "per_page": 30})
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "free-first-opportunity-radar/0.1"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = Request(API_ROOT + "?" + params, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        items = data.get("items", [])
        return (items, None) if isinstance(items, list) else ([], "GitHub Search API schema was unexpected.")
    except HTTPError as exc:
        return [], f"GitHub Search API HTTP {exc.code} for query {query!r}."
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"GitHub Search API unavailable for query {query!r}: {type(exc).__name__}."
    except Exception as exc:
        return [], f"Unexpected fetch failure for query {query!r}: {type(exc).__name__}."

def discover(token: str | None, timeout: int) -> tuple[list[dict[str, Any]], list[str]]:
    found: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for index, query in enumerate(QUERIES):
        items, warning = fetch_query(query, token, timeout)
        if warning:
            warnings.append(warning)
        for issue in items:
            candidate = score_issue(issue)
            if candidate and candidate["url"]:
                old = found.get(candidate["url"])
                if old is None or candidate["score"] > old["score"]:
                    found[candidate["url"]] = candidate
        if index < len(QUERIES) - 1:
            time.sleep(1.2)
    return sorted(found.values(), key=lambda x: (x["score"], x["updated_at"] or ""), reverse=True), warnings

def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Free-First Opportunity Radar", "",
        "Generated: " + report["generated_at"], "",
        "> Leads only, not guaranteed earnings. Verify active funding, eligibility, acceptance criteria and withdrawal route. The tool does not apply, claim rewards, test security targets or submit code automatically.",
        "", "## Highest-signal GitHub leads", "",
    ]
    issues = report["github_issues"]
    if not issues:
        lines.append("No qualifying GitHub issue with reward signals was found in this scan. This does not prove there is no paid work.")
    for index, item in enumerate(issues[:20], 1):
        amount = ", ".join(item["amount_mentions"]) if item["amount_mentions"] else "amount not detected"
        lines += [
            f"### {index}. [{item['title']}]({item['url']})",
            f"- Heuristic score: {item['score']}/100 (not a payout probability)",
            f"- Repository: {item['repo'] or 'unknown'}",
            f"- Reward text: {amount}",
            f"- Updated: {item['updated_at'] or 'unknown'}",
            "- Signals: " + ("; ".join(item["reasons"]) or "reward wording"),
            "- Gate: " + item["verification"], "",
        ]
    lines += ["## Direct application / platform leads", ""]
    for item in report["manual_leads"]:
        lines += [
            f"### [{item['title']}]({item['url']})",
            "- Type: " + item["kind"],
            "- Pay signal: " + item["payout_signal"],
            "- Fit: " + ", ".join(item["fit"]),
            "- Next step: " + item["next_step"],
            "- Status: " + item["verification"], "",
        ]
    lines += ["## Scan warnings", ""]
    lines += ["- " + warning for warning in report["warnings"]] if report["warnings"] else ["- No fetch warnings."]
    lines += [
        "", "## US$500/month target math", "",
        "- At US$7.50 per paid hour, US$500 gross requires 66.7 paid hours/month (~15.4 hours/week). This is arithmetic using the published maximum, not a guarantee of rate or task availability.",
        "- Count only money actually awarded and withdrawable. Exclude unclaimed bounties, speculative tokens, airdrops and dashboard estimates.",
        "- Never pay an application fee, buy a course, rent compute, share a seed phrase/private key, or run an unknown binary to access a supposed opportunity.", "",
    ]
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="reports/opportunity-radar")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--offline", action="store_true", help="Generate report without contacting GitHub.")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.offline:
        issues, warnings = [], ["Offline mode: live GitHub opportunities were not searched."]
    else:
        issues, warnings = discover(os.getenv("GITHUB_TOKEN"), max(3, args.timeout))
    report = {
        "schema_version": "0.1", "generated_at": utc_now().isoformat(timespec="seconds"),
        "policy": {"free_first": True, "auto_apply": False, "auto_claim_bounty": False,
                   "auto_security_testing": False, "auto_external_submission": False,
                   "guaranteed_income": False},
        "github_issues": issues, "manual_leads": MANUAL_LEADS, "warnings": warnings,
    }
    (out_dir / "opportunity-radar.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "opportunity-radar.md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote report to {out_dir.resolve()}")
    print(f"GitHub leads: {len(issues)}; platform leads: {len(MANUAL_LEADS)}; fetch warnings: {len(warnings)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
