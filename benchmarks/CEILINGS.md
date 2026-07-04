# Decode Ceiling Estimates

Derived no-load planning artifact. Not benchmark evidence.

- Assumed usable bandwidth: `550 GiB/s`.
- Ceiling formula: `decode_tps <= usable_bandwidth_gib_s / active_weight_gib_per_token`.
- Rows with `missing_estimate` need reviewed active-weight bytes before they can be used as a denominator.
- `gap_to_70pct_tps` is blank until a benchmark report supplies measured decode throughput.

| repo | local dir | mode | active GiB/token | ceiling tok/s | measured tok/s | util % | gap to 70% | status | evidence | source | bundle asset GiB | bundle status | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|
| redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix | qwen2.5-0.5b-instruct-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen2.5-3b-instruct-caix | qwen2.5-3b-instruct-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-0.6b-caix | qwen3-0.6b-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-1.7b-caix | qwen3-1.7b-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-4b-caix | qwen3-4b-coreai | decode | 2.30 | 239.1 | - | - | - | needs_measurement | estimate | .tmp/boss/ENGINEERING-GUIDE.md#2 | - | not_requested | Guide estimate: qwen3-4B 4-bit active weights approximately 2.3 GB/token; remeasure from bundle metadata before publishing percent-of-ceiling. |
| redhillsmediafl/rhm-qwen3-8b-caix | qwen3-8b-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-14b-caix | qwen3-14b-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-ornith-1.0-9b-caix | ornith-1.0-9b-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwythos-9b-caix | qwythos-9b-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-glm-4-9b-0414-caix | glm-4-9b-0414-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-glm-z1-9b-0414-caix | glm-z1-9b-0414-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-glm-4-32b-0414-caix | glm-4-32b-0414-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-gpt-oss-20b-caix | gpt-oss-20b-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-7b-instruct-v0.3-caix | mistral-7b-instruct-v0.3-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-nemo-instruct-2407-caix | mistral-nemo-instruct-2407-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-small-instruct-2409-caix | mistral-small-instruct-2409-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mixtral-8x7b-instruct-caix | mixtral-8x7b-instruct-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3.6-27b-caix | qwen3.6-27b-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-gemma-4-26b-a4b-caix | gemma-4-26b-a4b-coreai | decode | 2.50 | 220.0 | - | - | - | needs_measurement | estimate | .tmp/boss/ENGINEERING-GUIDE.md#2 | - | not_requested | Guide estimate: gemma-26B-A4B active expert weights approximately 2.5 GB/token; remeasure from bundle metadata before publishing percent-of-ceiling. |
| redhillsmediafl/rhm-gemma-4-31b-it-caix | gemma-4-31b-it-coreai | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-gemma-4-26b-a4b-mtp-caix | gemma-4-26b-a4b-mtp-coreai | eagle-mtp | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-4b-mtp-caix | qwen3-4b-mtp-coreai | speculative | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
