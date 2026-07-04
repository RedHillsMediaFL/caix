# Diffusion API Contract

Block diffusion does not produce left-to-right token deltas. The v1 caix API contract is therefore
non-streaming-only for diffusion bundles.

The machine-readable contract is `quality/diffusion_api_contract_v0.json` and is checked by
`scripts/check-diffusion-api-contract.sh`.

## v1 Behavior

- Non-streaming OpenAI chat completions may be enabled only after `diffusion_block_quality` and
  `diffusion_api_contract` evidence exists for the exact model revision.
- Streaming OpenAI chat completions for diffusion must be rejected until committed-block SSE is
  implemented and tested.
- Non-streaming Anthropic messages follow the same rule as OpenAI chat completions.
- Streaming Anthropic messages for diffusion must be rejected until committed-block SSE is implemented
  and tested.
- Token-delta streaming is not a valid diffusion claim.

## Deferred Block SSE

Committed-block SSE is the only planned streaming shape for diffusion. It must use a distinct block
event, not token-delta events. A future event must include at least:

| field | meaning |
|---|---|
| `block_index` | zero-based committed block index |
| `text` | text committed for the block |
| `accepted_token_count` | number of tokens accepted into the block |
| `denoise_step_count` | denoise iterations used for the block |
| `stop_reason` | block stop reason from the diffusion denoiser |

Until that path exists, public docs and model cards must describe diffusion serving as non-streaming.
