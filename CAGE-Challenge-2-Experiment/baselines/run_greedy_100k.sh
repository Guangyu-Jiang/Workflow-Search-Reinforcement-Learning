#!/bin/bash
# Run Greedy Heuristic baseline for 100k episodes

echo "Testing greedy heuristic with workflow: defender → enterprise → op_server → op_host → user"
echo ""

python baselines/greedy_heuristic.py \
    --n-episodes 100000 \
    --workflow defender,enterprise,op_server,op_host,user \
    --red-agent B_lineAgent

echo ""
echo "Results: logs/greedy_heuristic_*/"

