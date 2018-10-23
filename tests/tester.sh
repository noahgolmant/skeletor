#! /usr/bin/env bash

# Very simple invocations that validate things don't blow up in all
# command-line configurations. Doesn't do any semantic checking, but will catch
# egregious errors. Don't source this.
#
#   ./tester.sh
#   ./tester.sh --dry-run

set -eo pipefail

set -u

if [ $# -gt 1 ] || [ $# -eq 1 ] && [ "$1" != "--dry-run" ] ; then
    echo 'usage: ./scripts/tests.sh [--dry-run]' 1>&2
    exit 1
fi

if [ $# -eq 1 ] ; then
    DRY_RUN="true"
else
    DRY_RUN="false"
fi

box() {
    msg="* $1 *"
    echo "$msg" | sed 's/./\*/g'
    echo "$msg"
    echo "$msg" | sed 's/./\*/g'
}

main() {
    cmd=""
    function note_failure {
        box "${cmd}"
    }
    trap note_failure EXIT

    cmds=()
    ### Begin to add commands here!
    cmds+=("mkdir -p ./logs/test_logs")
    cmds+=("rm -rf ./logs/test_logs/*")

    cmds+=("python src/boilerplate.py")
    cmds+=("python ../examples/train.py --arch ResNet18 resnet_test --epochs 1 --batch_size 512")

    for cmd in "${cmds[@]}"; do
        box "${cmd}"
        if [ "$DRY_RUN" != "true" ] ; then
            bash -c "$cmd"
        fi
    done

    trap '' EXIT
}

main
