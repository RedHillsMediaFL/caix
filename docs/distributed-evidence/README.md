# Distributed Evidence

Do not add an evidence file until the test has passed. The readiness gate reads these files:

- `same-machine-qwen3-0.6b-token-match.txt`
- `loopback-qwen3-0.6b-token-match.txt`

Each file is line-oriented `key=value` text. Required fields:

```text
result=pass
mode=same-machine
model=qwen3-0.6b-coreai
manifest=<repo-relative staged manifest path>
caix_commit=<40-character git SHA>
prompt_set=docs/distributed-evidence/qwen3-0.6b-prompts.txt
prompts=<positive integer>
max_tokens=128
temperature=0
token_match=true
raw_log=<repo-relative raw log path or archive path>
```

For loopback evidence, use `mode=loopback`.

Rules:

- Same prompt set for monolithic and staged runs.
- `prompt_set=` must point to the committed prompt file used by both runs. Its non-empty line
  count must equal `prompts=`.
- Same `manifest=` value between same-machine and loopback runs.
- Evidence manifests must plan with `position_mode=full_prefix`.
- `manifest=`, `prompt_set=`, and `raw_log=` must be committed, repo-relative paths. No absolute
  paths, URLs, or untracked scratch files.
- `caix_commit=` must resolve to a commit in the local repo used for the readiness gate.
- Greedy only: `temperature=0`.
- Keep raw stdout/stderr or an archive. Do not record only a summary.
- Do not ask for Thunderbolt testing until both evidence files pass
  `scripts/check-distributed-readiness.sh`.
- The distributed readiness gate also needs a Brew-installed `caix` via `--brew-caix`.
