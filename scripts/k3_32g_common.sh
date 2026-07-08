#!/usr/bin/env bash

# Shared helpers for local scripts that drive the K3 32GB target.
# Source this file; do not execute it directly.

k3_load_target_env() {
  K3_HOST="${K3_HOST:-${K3_32G_HOST:-}}"
  K3_USER="${K3_USER:-${K3_32G_USER:-}}"
  if [[ -z "${SSHPASS:-}" ]]; then
    if [[ -n "${K3_PASS:-}" ]]; then
      export SSHPASS="${K3_PASS}"
    elif [[ -n "${K3_32G_PASS:-}" ]]; then
      export SSHPASS="${K3_32G_PASS}"
    elif [[ -n "${K3_PASSWORD:-}" ]]; then
      export SSHPASS="${K3_PASSWORD}"
    fi
  fi
}

k3_require_target_env() {
  k3_load_target_env
  local missing=()
  [[ -n "${K3_HOST:-}" ]] || missing+=("K3_HOST or K3_32G_HOST")
  [[ -n "${K3_USER:-}" ]] || missing+=("K3_USER or K3_32G_USER")
  if [[ "${#missing[@]}" -gt 0 ]]; then
    printf 'Missing K3 connection setting(s): %s\n' "${missing[*]}" >&2
    printf 'Provide K3_HOST/K3_USER and optionally K3_PASS, K3_32G_PASS, or SSHPASS.\n' >&2
    return 2
  fi
}

k3_print_target_contract() {
  cat <<'EOF'
K3 connection contract:
  K3_HOST or K3_32G_HOST     required
  K3_USER or K3_32G_USER     required
  K3_PASS/K3_32G_PASS/SSHPASS optional; required only for password auth

Scripts must not provide real host, user, or password defaults.
EOF
}
