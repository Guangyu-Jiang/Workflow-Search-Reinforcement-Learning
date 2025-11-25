#!/bin/bash

# Run all baseline evaluations

echo "🚀 Running All Baselines"
echo "========================"
echo ""

# 1. Random Policy
echo "1️⃣ Random Policy Baseline"
echo "------------------------"
python baselines/random_policy.py --n-episodes 1000 --red-agent B_lineAgent
echo ""

# 2. Fixed Priority Workflows (Greedy Heuristic)
echo "2️⃣ Fixed Priority Workflows (6 workflows)"
echo "------------------------"
python baselines/fixed_priority_workflows.py --n-episodes 1000 --red-agent B_lineAgent
echo ""

echo "✅ All baselines complete!"
echo ""
echo "Results saved to:"
echo "  - logs/random_policy_*/"
echo "  - logs/greedy_heuristic_*/ (6 workflows)"
echo "  - logs/fixed_workflows_*/summary.csv (comparison)"

