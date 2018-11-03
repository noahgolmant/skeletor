#! /usr/bin/env bash

# Lints code:
#
#   # Lint skeletor by default.
#   ./scripts/lint.sh
#   # Lint specific files.
#   ./scripts/lint.sh skeletor/somefile/*.py

set -euo pipefail

lint() {
    pylint "$@"
}

main() {
    if [[ "$#" -eq 0 ]]; then
        lint skeletor
    else
        lint "$@"
    fi
}

main "$@"
