#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="${PROJECT_ROOT}/examples/specific_examples/metrics"
CACHE_DIR="${TMPDIR:-/tmp}/xai-metrics-example-cache"

mkdir -p "${CACHE_DIR}/matplotlib" "${CACHE_DIR}/xdg-cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_DIR}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_DIR}/xdg-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_metric_examples.sh [metric ...]

Runs all metric-specific examples in examples/specific_examples/metrics.
Pass metric names to run only a subset, for example:
  scripts/run_metric_examples.sh Faithfulness Complexity
  scripts/run_metric_examples.sh Faithfulness_examples.py
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${EXAMPLES_DIR}" ]]; then
  echo "Metric examples directory not found: ${EXAMPLES_DIR}" >&2
  exit 1
fi

resolve_example() {
  local name="$1"
  local candidate

  if [[ "${name}" == */* ]]; then
    candidate="${PROJECT_ROOT}/${name}"
  elif [[ "${name}" == *_examples.py ]]; then
    candidate="${EXAMPLES_DIR}/${name}"
  else
    candidate="${EXAMPLES_DIR}/${name}_examples.py"
  fi

  if [[ -f "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  echo "Example not found for '${name}' at ${candidate}" >&2
  return 1
}

examples=()
if [[ "$#" -gt 0 ]]; then
  for metric_name in "$@"; do
    if example_path="$(resolve_example "${metric_name}")"; then
      examples+=("${example_path}")
    else
      exit 1
    fi
  done
else
  while IFS= read -r example_path; do
    examples+=("${example_path}")
  done < <(find "${EXAMPLES_DIR}" -maxdepth 1 -type f -name '*_examples.py' | sort)
fi

if [[ "${#examples[@]}" -eq 0 ]]; then
  echo "No metric examples found in ${EXAMPLES_DIR}" >&2
  exit 1
fi

echo "Python: ${PYTHON_BIN}"
echo "Metric examples: ${#examples[@]}"
echo

failures=()
for example_path in "${examples[@]}"; do
  relative_path="${example_path#"${PROJECT_ROOT}/"}"
  echo "===== Running ${relative_path} ====="

  if (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" "${example_path}"); then
    echo "===== OK ${relative_path} ====="
  else
    status="$?"
    echo "===== FAILED ${relative_path} (exit ${status}) =====" >&2
    failures+=("${relative_path}")
  fi

  echo
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "Failed metric examples:" >&2
  printf '  - %s\n' "${failures[@]}" >&2
  exit 1
fi

echo "All metric examples finished successfully."
