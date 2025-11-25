# 📊 Baseline Implementations

This directory contains non-learning and fixed-strategy baselines for comparison with the workflow search method.

---

## Available Baselines

### **1. Random Policy** (`random_policy.py`)

**What:** Selects actions uniformly at random

**Run:**
```bash
python baselines/random_policy.py --n-episodes 1000 --red-agent B_lineAgent
```

**Expected Performance:**
- Reward: -1000 to -800
- Compliance: ~20% (by chance)
- Purpose: Lower bound

---

### **2. Greedy Heuristic** (`greedy_heuristic.py`)

**What:** Always fixes the highest-priority currently-compromised host

**Algorithm:**
1. Check which hosts are compromised
2. Find highest-priority type in workflow order
3. Execute restore action for that type
4. Repeat

**Run:**
```bash
python baselines/greedy_heuristic.py \
    --workflow defender,enterprise,op_server,op_host,user \
    --n-episodes 1000
```

**Expected Performance:**
- Reward: -50 to -150
- Compliance: ~100% (by design)
- Purpose: Strong non-learning baseline

---

### **3. Fixed Priority Workflows** (`fixed_priority_workflows.py`)

**What:** Tests 6 predefined workflows using greedy heuristic

**Workflows:**
- critical_first: defender → op_server → enterprise → op_host → user
- enterprise_focus: enterprise → defender → op_server → op_host → user
- user_priority: user → defender → enterprise → op_server → op_host
- operational_focus: op_server → op_host → defender → enterprise → user
- balanced: defender → enterprise → op_server → user → op_host
- reverse: user → op_host → enterprise → op_server → defender

**Run:**
```bash
python baselines/fixed_priority_workflows.py --n-episodes 1000
```

**Expected Performance:**
- Best workflow: -50 to -100 reward
- Worst workflow: -200 to -400 reward
- All have ~100% compliance (greedy)
- Purpose: Show GP-UCB finds better than fixed choices

---

## Run All Baselines

```bash
bash baselines/run_all_baselines.sh
```

This runs:
- Random policy (1000 episodes)
- All 6 fixed workflows (1000 episodes each)

Total: ~7000 episodes, takes ~5-10 minutes

---

## Comparison with Learning Methods

| Baseline | Learning? | Workflow? | Compliance | Reward | Time |
|----------|-----------|-----------|------------|--------|------|
| **Random** | No | No | 20% | -1000 | Instant |
| **Fixed (worst)** | No | Yes (fixed) | 100% | -400 | Fast |
| **Fixed (best)** | No | Yes (fixed) | 100% | -100 | Fast |
| **PPO Baseline** | Yes | No | 30% | -30 | Hours |
| **Fixed-Episodes** | Yes | Yes (learned) | 60-70% | -25 | Hours |
| **Adaptive (yours)** | Yes | Yes (learned) | 90% | -10 | Hours |

---

## Output

Each baseline saves:
```
logs/BASELINE_NAME_TIMESTAMP/
├── training_log.csv
└── (summary.csv for fixed workflows)
```

---

## Purpose

These baselines establish:
1. **Lower bound** (Random)
2. **Upper bound for fixed strategies** (Greedy with best workflow)
3. **Value of learning** (compare to PPO)
4. **Value of GP-UCB** (compare to fixed workflows)
5. **Value of compliance training** (compare fixed vs adaptive episodes)

Complete baseline coverage for paper! 🎯
