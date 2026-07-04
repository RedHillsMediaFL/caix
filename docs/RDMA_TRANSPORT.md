# RDMA Transport Contract

This is a design contract for future Thunderbolt 5 RDMA transport work. It is not hardware evidence
and it does not change the current socket transport.

The machine-readable contract is `docs/RDMA_TRANSPORT_CONTRACT.json` and is checked by
`scripts/check-rdma-transport-contract.sh`.

## Current Path

The shipping distributed path uses the existing worker-frame protocol over TCP:

- JSON header line
- optional raw tensor payload
- the same request id, stage id, step index, hidden-state metadata, and byte-count validation as the
  loopback path

This remains the Thunderbolt 4 and LAN fallback. It is transport evidence only until real staged
token parity exists.

## Planned RDMA Path

`rdma_verbs_tb5` is the planned transport id for future Thunderbolt 5 hardware. The contract assumes
Apple RDMA Verbs through a thin C shim called from Swift. JACCL is optional future work for
collectives; it is not required for pipeline stage handoff.

Required RDMA constraints:

| field | requirement |
|---|---|
| operation model | send/receive only |
| one-sided read/write | not used |
| maximum message bytes | `16773120` |
| queue-pair budget | at most `10` unreliable-connection queue pairs |
| work requests | at most `4095` in flight |
| buffers | page-aligned |
| memory registration | per Thunderbolt controller |

Payloads over the RDMA message limit must be chunked while preserving the worker-frame request id,
stage id, step index, hidden-state shape, dtype, and byte count.

## Negotiation

Workers and coordinators must negotiate the transport from captured peer capabilities:

- plan integrity hash
- machine identity
- caix version
- link kind
- macOS version
- Thunderbolt generation
- `rdma_ctl` state
- visible `ibv` device names
- transport capability descriptor

If any RDMA requirement is missing on any peer, the selected transport is `tcp_worker_frame` and the
fallback reason must be recorded in evidence.

## Claims

- Do not claim RDMA without a Thunderbolt 5 pair and captured negotiation evidence.
- Do not claim decode speedup without token-accurate raw evidence.
- Do not claim tensor parallelism until collectives are implemented and tested.
- Thunderbolt 4 TCP stage sharding remains capacity or transport evidence only.
