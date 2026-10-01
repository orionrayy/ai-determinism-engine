from __future__ import annotations

import os
import urllib.request
import json


def post_issue_status(workflow: dict, message: str, http_json) -> None:
    issue_number = workflow.get('trigger_issue')
    token = os.environ.get('GITHUB_TOKEN')
    repository = os.environ.get('GITHUB_REPOSITORY')
    if not issue_number or not token or not repository:
        return
    try:
        http_json(
            'https://api.github.com/repos/' + repository + '/issues/' + str(int(issue_number)) + '/comments',
            method='POST',
            body={'body': message},
            headers={
                'Authorization': 'Bearer ' + token,
                'X-GitHub-Api-Version': '2022-11-28',
                'Accept': 'application/vnd.github+json',
            },
            timeout=30,
        )
    except Exception:
        return