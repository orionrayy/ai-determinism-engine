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


def research_bundle(query: str, include_extended: bool = False) -> dict:
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
    if not ordered_results:
        raise RuntimeError('all research providers failed')
    return {'query': query, 'sources': ordered_results, 'errors': ordered_errors}