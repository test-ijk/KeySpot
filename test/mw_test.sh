#!/bin/bash

VALUES=()

for v in $(seq 0 0.05 1); do
    VALUES+=("$v")
    if [[ "$v" != "0" ]]; then
        VALUES+=("-$v")
    fi
done


for d in "${VALUES[@]}"; do
    echo "=== RUN disturb=$d ==="
    export ENV_TWEAK_DISTURB=$d
    bash run.sh 250729 run1rend
done

# bash run_disturb.sh