# Distributed Evidence

Do not add an evidence file until the test has passed. The readiness gate reads these files:

- `same-machine-qwen3-0.6b-token-match.txt`
- `loopback-qwen3-0.6b-token-match.txt`

The local same-machine parity evidence that closed the Qwen3-0.6B staged math gate is separate:

- `qwen3-0.6b-teacher-forced-fp16-128.txt`

That file records the teacher-forced HF-fp16 oracle gate for the Qwen3-0.6B noopt-first3 staged
asset: 8 prompts, 128 decode steps, 1020 exact matches, 4 accepted fp16 ties, and 0 real
divergences. The contract pins the caix commit, CoreAI Models commit, prompt set, raw XCTest log,
tie tolerance, exact-match count, accepted-tie count, and `upload_ready=false`. It is not loopback,
two-machine, load, speed, or upload-readiness evidence.
Use `scripts/check-distributed-evidence-contract.sh --require-tracked` for final publication review
so untracked local evidence files cannot satisfy the contract. The publication gate exposes the same
strict path as `scripts/check-publication-gates.sh --strict-evidence`; distributed publication review
with `scripts/check-publication-gates.sh --distributed` also enables it automatically.

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
asset_digests=<repo-relative stage asset digest file>
raw_log=<repo-relative raw log path or archive path>
```

For loopback evidence, use `mode=loopback`.

Rules:

- Same prompt set for monolithic and staged runs.
- `prompt_set=` must point to the committed prompt file used by both runs. Its non-empty line
  count must equal `prompts=`.
- Same `manifest=` value between same-machine and loopback runs.
- Evidence manifests must plan with `position_mode=full_prefix`.
- `manifest=`, `prompt_set=`, `asset_digests=`, and `raw_log=` must be committed,
  repo-relative paths. No absolute paths, URLs, or untracked scratch files.
- `asset_digests=` must list every planned stage asset and optional decode asset as
  `<sha256> <repo-relative-path>`. Hash each `.aimodel` directory deterministically before recording
  token-match evidence.
- `caix_commit=` must resolve to a commit in the local repo used for the readiness gate.
- Greedy only: `temperature=0`.
- Keep raw stdout/stderr or an archive. Do not record only a summary.
- Do not publish real-Qwen distributed claims until both evidence files pass
  `scripts/check-distributed-readiness.sh`.
- The distributed readiness gate also needs a Brew-installed `caix` via `--brew-caix`.

<sub>More open-source work: [redhillsmediafl.com/open-source](https://redhillsmediafl.com/open-source).</sub>
