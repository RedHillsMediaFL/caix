#!/usr/bin/env bash

caix_env() {
  local lower_name="$1"
  local legacy_suffix="$2"
  local fallback="${3-}"
  local value="${!lower_name-}"
  if [[ -z "$value" ]]; then
    local legacy_name="C""AIX_$legacy_suffix"
    value="${!legacy_name-}"
  fi
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf '%s' "$fallback"
  fi
}
