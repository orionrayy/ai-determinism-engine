from __future__ import annotations

import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .http_safety import read_response_limited
except ImportError:
    from http_safety import read_response_limited

MAX_RESEARCH_RESPONSE_BYTES = 512 * 1024


def _request(url: str, user_agent: str) -> dict:
    from urllib.request import Request, urlopen
    req = Request(
        url,
        headers={'Accept': 'application/json', 'User-Agent': user_agent},
        method='GET',
    )
    with urlopen(req, timeout=30) as response:
        raw = read_response_limited(
            response,
            MAX_RESEARCH_RESPONSE_BYTES,
            error_message="research response exceeds 512 KiB safety limit",
        ).decode('utf-8', 'replace')
        return json.loads(raw) if raw else {}


def search_wikipedia(query: str) -> dict:
    params = urllib.parse.urlencode({
        'action': 'query', 'list': 'search',
        'srsearch': query[:250], 'srlimit': '8',
        'format': 'json', 'utf8': '1',
    })
    return _request(
        'https://en.wikipedia.org/w/api.php?' + params,
        'ai-orchestrator-research/1.0',
    )


def search_arxiv(query: str) -> dict:
    params = urllib.parse.urlencode({
        'search_query': 'all:' + query[:180],
        'start': '0', 'max_results': '8',
        'sortBy': 'relevance', 'sortOrder': 'descending',
    })
    import xml.etree.ElementTree as ET
    url = 'https://export.arxiv.org/api/query?' + params
    from urllib.request import Request, urlopen
    req = Request(url, headers={'User-Agent': 'ai-orchestrator-research/1.0'})
    with urlopen(req, timeout=30) as response:
        raw = read_response_limited(
            response,
            MAX_RESEARCH_RESPONSE_BYTES,
            error_message="research response exceeds 512 KiB safety limit",
        )
        root = ET.fromstring(raw)
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    items = []
    for entry in root.findall('a:entry', ns):
        items.append({
            'id': entry.findtext('a:id', default='', namespaces=ns),
            'title': entry.findtext('a:title', default='', namespaces=ns).strip(),
            'summary': entry.findtext('a:summary', default='', namespaces=ns).strip(),
            'published': entry.findtext('a:published', default='', namespaces=ns),
        })
    return {'query': query, 'results': items}


def search_crossref(query: str) -> dict:
    params = urllib.parse.urlencode({'query.bibliographic': query[:200], 'rows': '8'})
    data = _request(
        'https://api.crossref.org/works?' + params,
        'ai-orchestrator-research/1.0',
    )
    results = []
    for item in data.get('message', {}).get('items', []):
        results.append({
            'DOI': item.get('DOI'),
            'title': (item.get('title') or [''])[0],
            'published': item.get('published-print') or item.get('published-online') or {},
            'container-title': (item.get('container-title') or [''])[0],
        })
    return {'query': query, 'results': results}


def research_bundle(
    query: str,
    include_extended: bool = False,
    budget: str = "balanced",
) -> dict:
    errors = {}
    results = {}
    providers = (
        ('wikipedia', search_wikipedia),
        ('arxiv', search_arxiv),
        ('crossref', search_crossref),
    )
    with ThreadPoolExecutor(max_workers=len(providers), thread_name_prefix='research') as pool:
        futures = {pool.submit(fn, query): name for name, fn in providers}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                errors[name] = {'type': type(exc).__name__, 'message': str(exc)}
    ordered_results = {name: results[name] for name, _ in providers if name in results}
    ordered_errors = {name: errors[name] for name, _ in providers if name in errors}

    response = {
        'query': query,
        'sources': ordered_results,
        'errors': ordered_errors,
    }
    if not include_extended:
        if not ordered_results:
            raise RuntimeError('all research providers failed')
        return response

    try:
        try:
            from .evidence_records import (
                count_independent_sources,
                deduplicate_sources,
                normalize_source,
            )
            from .research_budget import (
                budget_metadata,
                normalize_budget,
            )
            from .research_providers import research_records
        except ImportError:
            from evidence_records import (
                count_independent_sources,
                deduplicate_sources,
                normalize_source,
            )
            from research_budget import (
                budget_metadata,
                normalize_budget,
            )
            from research_providers import research_records

        legacy_records = []

        wiki = ordered_results.get('wikipedia', {})
        wiki_search = (
            wiki.get('query', {}).get('search', [])
            if isinstance(wiki, dict)
            else []
        )
        if isinstance(wiki_search, list):
            for item in wiki_search:
                if isinstance(item, dict):
                    legacy_records.append(normalize_source('wikipedia', item))

        arxiv = ordered_results.get('arxiv', {})
        arxiv_results = arxiv.get('results', []) if isinstance(arxiv, dict) else []
        if isinstance(arxiv_results, list):
            for item in arxiv_results:
                if isinstance(item, dict):
                    legacy_records.append(normalize_source('arxiv', item))

        crossref = ordered_results.get('crossref', {})
        crossref_results = (
            crossref.get('results', [])
            if isinstance(crossref, dict)
            else []
        )
        if isinstance(crossref_results, list):
            for item in crossref_results:
                if isinstance(item, dict):
                    legacy_records.append(normalize_source('crossref', item))

        selected_budget = normalize_budget(budget)
        response['research_budget'] = budget_metadata(selected_budget)

        try:
            from .research_budget import choose_extended_providers
        except ImportError:
            from research_budget import choose_extended_providers

        def normalize_provider_order(
            query_text: str,
            budget_value,
            available_providers: list[str],
        ) -> list[str]:
            return choose_extended_providers(
                query_text,
                budget=budget_value,
                available=tuple(available_providers),
            )
        combined = deduplicate_sources(legacy_records)
        used_providers: list[str] = []
        provider_counts: dict[str, int] = {}
        extended_errors: dict[str, str] = {}
        extended_successes = 0

        while (
            count_independent_sources(combined)
            < selected_budget.target_independent_sources
            and len(used_providers) < selected_budget.max_extended_providers
        ):
            remaining = [
                provider for provider in (
                    'semantic_scholar',
                    'openalex',
                    'europe_pmc',
                    'core',
                )
                if provider not in used_providers
            ]
            ranked = [
                provider
                for provider in normalize_provider_order(query, selected_budget, remaining)
            ]
            if not ranked:
                break
            stage = tuple(ranked[: selected_budget.stage_size])
            extended_records = research_records(
                query,
                providers=stage,
                max_results=selected_budget.max_results,
            )
            successful = [
                provider for provider in stage
                if provider not in (extended_records.get('errors') or {})
            ]
            extended_successes += len(successful)
            used_providers.extend(stage)
            for provider, count in (extended_records.get('provider_counts') or {}).items():
                provider_counts[provider] = provider_counts.get(provider, 0) + int(count or 0)
            extended_errors.update(extended_records.get('errors') or {})
            combined = deduplicate_sources(
                combined + list(extended_records.get('evidence_records') or [])
            )
            if (
                not extended_records.get('evidence_records')
                and all(provider in extended_errors for provider in stage)
            ):
                break

        response['evidence_records'] = combined
        response['independent_source_count'] = count_independent_sources(combined)
        response['provider_counts'] = provider_counts
        response['extended_errors'] = extended_errors
        response['extended_providers_used'] = used_providers[:8]
        response['research_stopped_early'] = (
            response['independent_source_count'] >= selected_budget.target_independent_sources
        )
        response['research_extended_successes'] = extended_successes
        if not ordered_results and not combined:
            raise RuntimeError('all research providers failed')
    except Exception as exc:
        try:
            from .evidence_records import count_independent_sources, deduplicate_sources
        except ImportError:
            from evidence_records import count_independent_sources, deduplicate_sources
        combined = deduplicate_sources(legacy_records)
        response['evidence_records'] = combined
        response['independent_source_count'] = count_independent_sources(combined)
        response['provider_counts'] = {}
        response['extended_errors'] = {
            'fabric': {
                'type': type(exc).__name__,
                'message': str(exc),
            }
        }
    return response
