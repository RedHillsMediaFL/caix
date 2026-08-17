# Decode Ceiling Estimates

Derived no-load planning artifact. Not benchmark evidence.

- Assumed usable bandwidth: `550 GiB/s`.
- Ceiling formula: `decode_tps <= usable_bandwidth_gib_s / active_weight_gib_per_token`.
- Rows with `missing_estimate` need reviewed active-weight bytes before they can be used as a denominator.
- `gap_to_70pct_tps` is blank until a benchmark report supplies measured decode throughput.

| repo | local dir | mode | active GiB/token | ceiling tok/s | measured tok/s | util % | gap to 70% | status | evidence | source | bundle asset GiB | bundle status | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---|
| redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix | qwen2.5-0.5b-instruct-coreai-4bit-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen2.5-0.5b-instruct-bf16-caix | qwen2.5-0.5b-instruct-bf16-coreai-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen2.5-3b-instruct-caix | qwen2.5-3b-instruct-coreai-4bit-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen2.5-3b-instruct-bf16-caix | qwen2.5-3b-instruct-bf16-coreai-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-0.6b-caix | qwen3-0.6b-coreai-4bit-ctx40960 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-0.6b-bf16-caix | qwen3-0.6b-bf16-coreai-ctx40960 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-1.7b-caix | qwen3-1.7b-coreai-4bit-ctx40960 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-1.7b-bf16-caix | qwen3-1.7b-bf16-coreai-ctx40960 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-8b-caix | qwen3-8b-coreai-4bit-ctx40960 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-8b-bf16-caix | qwen3-8b-bf16-coreai-ctx40960 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-14b-caix | qwen3-14b-coreai-4bit-ctx40960 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-qwen3-14b-bf16-caix | qwen3-14b-bf16-coreai-ctx40960 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-glm-4-9b-0414-caix | glm-4-9b-0414-coreai-4bit-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-glm-4-9b-0414-bf16-caix | glm-4-9b-0414-bf16-coreai-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-glm-4-32b-0414-caix | glm-4-32b-0414-coreai-4bit-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-glm-4-32b-0414-int8-caix | glm-4-32b-0414-int8-coreai-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-gpt-oss-20b-caix | gpt-oss-20b-coreai-4bit-ctx131072 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-gpt-oss-20b-int8-caix | gpt-oss-20b-int8-coreai-ctx131072 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-7b-instruct-v0.3-caix | mistral-7b-instruct-v0.3-coreai-4bit-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-7b-instruct-v0.3-bf16-caix | mistral-7b-instruct-v0.3-bf16-coreai-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-nemo-instruct-2407-caix | mistral-nemo-instruct-2407-coreai-4bit-ctx131072 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-nemo-instruct-2407-bf16-caix | mistral-nemo-instruct-2407-bf16-coreai-ctx131072 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-small-instruct-2409-caix | mistral-small-instruct-2409-coreai-4bit-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mistral-small-instruct-2409-int8-caix | mistral-small-instruct-2409-int8-coreai-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
| redhillsmediafl/rhm-mixtral-8x7b-instruct-caix | mixtral-8x7b-instruct-coreai-4bit-ctx32768 | decode | - | - | - | - | - | missing_estimate | missing |  | - | not_requested | active-weight estimate needed |
