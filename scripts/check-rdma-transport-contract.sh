#!/usr/bin/env bash
# Validate the no-load RDMA transport design contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTRACT="$REPO_DIR/docs/RDMA_TRANSPORT_CONTRACT.json"
DOC="$REPO_DIR/docs/RDMA_TRANSPORT.md"
DISTRIBUTED_DOC="$REPO_DIR/docs/DISTRIBUTED_EXECUTION.md"
CLUSTER_DOC="$REPO_DIR/docs/CLUSTER.md"

[[ -f "$CONTRACT" ]] || { echo "error: RDMA contract missing: $CONTRACT" >&2; exit 1; }
[[ -f "$DOC" ]] || { echo "error: RDMA doc missing: $DOC" >&2; exit 1; }

python3 - "$CONTRACT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open() as handle:
    contract = json.load(handle)

def require(condition, message):
    if not condition:
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(1)

require(contract.get("schema") == "caix.rdma_transport_contract.v0", "unexpected RDMA schema")
require(contract.get("status") == "design_only_no_hardware_evidence", "RDMA contract must stay design-only")
require(contract.get("current_shipping_transport") == "tcp_worker_frame", "current transport must be tcp_worker_frame")

transports = {item.get("id"): item for item in contract.get("planned_transports", [])}
require("tcp_worker_frame" in transports, "tcp_worker_frame transport missing")
require("rdma_verbs_tb5" in transports, "rdma_verbs_tb5 transport missing")
require(transports["rdma_verbs_tb5"].get("api") == "apple_rdma_verbs", "RDMA API must be Apple RDMA Verbs")
require(transports["rdma_verbs_tb5"].get("swift_boundary") == "thin_c_shim", "Swift boundary must be a thin C shim")

constraints = contract.get("rdma_verbs_constraints", {})
require(constraints.get("send_recv_only") is True, "RDMA contract must use send/recv only")
require(constraints.get("one_sided_read_write") is False, "RDMA contract must reject one-sided read/write")
require(constraints.get("max_message_bytes") == 16773120, "RDMA max message bytes must stay 16773120")
require(constraints.get("max_unreliable_connection_queue_pairs") == 10, "RDMA UC queue-pair budget must stay 10")
require(constraints.get("max_work_requests_in_flight") == 4095, "RDMA work-request budget must stay 4095")
require(constraints.get("page_aligned_buffers") is True, "RDMA buffers must be page-aligned")
require(
    constraints.get("memory_registration_scope") == "per_thunderbolt_controller",
    "RDMA memory registration must be per Thunderbolt controller",
)

negotiation = set(contract.get("negotiation_inputs", []))
for field in [
    "plan_integrity_hash",
    "peer_machine_id",
    "peer_caix_version",
    "link_kind",
    "macos_version",
    "thunderbolt_generation",
    "rdma_ctl_enabled",
    "ibv_device_names",
    "transport_capabilities",
]:
    require(field in negotiation, f"negotiation input missing: {field}")

selection = set(contract.get("selection_rules", []))
for rule in [
    "rdma_verbs_tb5_requires_all_peers_tb5",
    "rdma_verbs_tb5_requires_macos_26_2_or_newer",
    "rdma_verbs_tb5_requires_rdma_ctl_enabled",
    "rdma_verbs_tb5_requires_ibv_devices",
    "fallback_to_tcp_worker_frame_when_any_requirement_is_missing",
    "record_fallback_reason_in_evidence",
]:
    require(rule in selection, f"selection rule missing: {rule}")

payload = set(contract.get("payload_rules", []))
for rule in [
    "chunk_payloads_over_max_message_bytes",
    "preserve_worker_frame_request_id",
    "preserve_stage_id_and_step_index",
    "preserve_hidden_state_metadata_shape_dtype_and_byte_count",
    "do_not_retry_forward_after_partial_send",
    "fail_request_on_transport_disconnect_or_timeout",
]:
    require(rule in payload, f"payload rule missing: {rule}")

kv = set(contract.get("kv_rules", []))
for rule in ["kv_stays_on_stage_owner", "no_mid_request_kv_migration", "recovery_requires_reprefill_from_token_zero"]:
    require(rule in kv, f"KV rule missing: {rule}")

claims = set(contract.get("claim_rules", []))
for rule in [
    "no_rdma_claim_without_tb5_pair_and_captured_negotiation",
    "no_decode_speedup_claim_without_token_accurate_raw_evidence",
    "no_tensor_parallel_claim_until_collectives_are_implemented_and_tested",
    "tb4_tcp_stage_sharding_remains_capacity_or_transport_evidence_only",
]:
    require(rule in claims, f"claim rule missing: {rule}")
PY

grep -F 'docs/RDMA_TRANSPORT_CONTRACT.json' "$DOC" >/dev/null \
  || { echo "error: RDMA doc must link the contract JSON" >&2; exit 1; }
grep -F 'not hardware evidence' "$DOC" >/dev/null \
  || { echo "error: RDMA doc must state it is not hardware evidence" >&2; exit 1; }
grep -F 'Thunderbolt 5' "$DOC" >/dev/null \
  || { echo "error: RDMA doc must name Thunderbolt 5" >&2; exit 1; }
grep -F 'tcp_worker_frame' "$DOC" >/dev/null \
  || { echo "error: RDMA doc must name the TCP fallback transport" >&2; exit 1; }
grep -F 'RDMA_TRANSPORT.md' "$DISTRIBUTED_DOC" >/dev/null \
  || { echo "error: distributed doc must link RDMA_TRANSPORT.md" >&2; exit 1; }
grep -F 'check-rdma-transport-contract.sh' "$CLUSTER_DOC" >/dev/null \
  || { echo "error: cluster doc must mention the RDMA contract checker" >&2; exit 1; }

echo "rdma transport contract ok"
