# Encrypted orchestrator state branch

Status: G0-A implementation candidate. The default branch is not changed by this work.
State branch: orchestrator-state
Implementation: orchestrator/state_branch.mjs

## Security model

Runtime files in .orchestrator are serialized as independent objects. Logical paths are represented by HMAC-SHA256 tokens. Object contents use AES-256-GCM. Per-repository encryption and path keys are derived from one 32-byte master secret with HKDF-SHA256. Authenticated additional data binds an object to the repository, state branch and opaque path token.

The dedicated branch must contain only encrypted objects at:
.orchestrator-state/v1/<opaque-token>.enc

Unexpected paths, invalid envelopes, authentication failures, symlinks, unsupported filesystem objects, size-limit violations and missing keys fail closed. The research cache is ephemeral and excluded from persistence. Git history still reveals commit timing, object count and approximate ciphertext sizes; it does not reveal logical paths or plaintext payloads.

This protects state confidentiality at rest in a public repository. It does not replace access control. The state branch remains publicly readable and can be deleted or corrupted by a repository writer. Restrict repository write access and keep the master key outside the repository.

## Configure the key before enabling runtime workflows

Generate the key locally. Never paste it into chat, an issue, a commit, workflow logs or a repository file.

Termux / Python:

    python - <<'PY'
    import base64, secrets
    print(base64.b64encode(secrets.token_bytes(32)).decode())
    PY

Add the generated value as a GitHub Actions repository secret named:

    ORCHESTRATOR_STATE_ENCRYPTION_KEY

Location: repository Settings -> Secrets and variables -> Actions -> New repository secret.

The value must be canonical Base64 for exactly 32 random bytes. If the secret is absent or malformed, restore and persist fail closed; there is no plaintext fallback.

## Runtime lifecycle

1. Checkout source code.
2. Restore state from orchestrator-state. On the first migration only, retain the checked-out legacy .orchestrator tree locally; the first successful persist encrypts it before creating the dedicated state branch.
3. Execute the orchestrator or scheduled recovery.
4. Snapshot the resulting state and perform a three-way merge against the restore-time snapshot and current remote branch.
5. Commit only encrypted object blobs to the dedicated branch; push without force and verify the remote head.
6. Retry remote-head races with a fresh read and merge. Same-path concurrent modifications fail closed; disjoint object writes may be combined.

The runner-local base checkpoint is stored under the system temporary directory and is not committed or uploaded as an artifact.

## Operations and recovery

- Keep the secret in a password manager or durable, access-controlled offline backup. Losing it makes encrypted history unrecoverable.
- Do not rotate the secret by simply replacing the repository secret. Rotation requires an explicit decrypt-and-re-encrypt migration while both old and new keys are securely available.
- To recover, restore the same secret, inspect state-branch object integrity, and rerun restore. Do not manually edit ciphertext.
- Never force-push the state branch. Resolve STATE_WRITE_CONFLICT by inspecting the conflicting logical object in a controlled local environment; do not blindly retry an external side effect.
- Do not delete the state branch while a workflow may be active.

## Acceptance boundary

Local Node tests validate encryption/authentication, path validation, symlink rejection, three-way merge semantics, ciphertext-only Git persistence and restore. They do not prove hosted GitHub Actions permissions, repository-secret configuration, branch governance, concurrent hosted writes or production recovery. Those remain separate G0 acceptance requirements.
