#! /usr/bin/env bash

# Lints code:
#
#   # Lint lbs by default.
#   ./scripts/lint.sh
#   # Lint specific files.
#   ./scripts/lint.sh lbs/somefile/*.py

set -euo pipefail

lint() {
    pylint "$@"
}

main() {
    if [[ "$#" -eq 0 ]]; then
        lint lbs
    else
        lint "$@"
    fi
}

main "$@"
