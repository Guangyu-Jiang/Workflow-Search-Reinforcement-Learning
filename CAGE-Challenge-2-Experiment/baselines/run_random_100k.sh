#!/bin/bash
# Run Random Policy baseline for 100k episodes

python baselines/random_policy.py \
    --n-episodes 100000 \
    --red-agent B_lineAgent

echo ""
echo "Results: logs/random_policy_*/"

