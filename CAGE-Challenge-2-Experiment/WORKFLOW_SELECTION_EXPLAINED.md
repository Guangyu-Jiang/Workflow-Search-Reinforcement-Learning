# 🎲 Workflow Selection Strategy Explained

## The Code You're Looking At

```python
# Select workflow
if iteration <= 5:
    canonical_dict = self.workflow_manager.get_canonical_workflows()
    candidate_orders = list(canonical_dict.values())
else:
    unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
    candidate_orders = []
    for _ in range(10):
        perm = unit_types.copy()
        np.random.shuffle(perm)
        candidate_orders.append(perm)
```

This code generates **candidate workflows** for GP-UCB to choose from.

---

## 🎯 Two-Phase Strategy

### **Phase 1: First 5 Iterations (Cold Start)**
```python
if iteration <= 5:
    canonical_dict = self.workflow_manager.get_canonical_workflows()
    candidate_orders = list(canonical_dict.values())
```

**What it does:**
Uses **predefined "canonical" workflows** - carefully designed starting points.

**The 6 Canonical Workflows:**
```python
{
    'critical_first': ['defender', 'op_server', 'enterprise', 'op_host', 'user'],
    'enterprise_focus': ['enterprise', 'defender', 'op_server', 'op_host', 'user'],
    'user_priority': ['user', 'defender', 'enterprise', 'op_server', 'op_host'],
    'operational_focus': ['op_server', 'op_host', 'defender', 'enterprise', 'user'],
    'balanced': ['defender', 'enterprise', 'op_server', 'user', 'op_host'],
    'reverse': ['user', 'op_host', 'enterprise', 'op_server', 'defender']
}
```

**Why use canonical workflows first?**
1. **Diverse initial sampling** - Cover different strategies
2. **Good GP initialization** - Provide diverse data points
3. **Known to be reasonable** - Not completely random
4. **Bootstrap the GP model** - Need some observations before GP works well

**Example:**
```
Iteration 1: GP-UCB picks from 6 canonical → selects 'critical_first'
Iteration 2: GP-UCB picks from 6 canonical → selects 'user_priority'  
Iteration 3: GP-UCB picks from 6 canonical → selects 'reverse'
...
Iteration 5: GP-UCB picks from 6 canonical → selects 'balanced'
```

---

### **Phase 2: After Iteration 5 (GP-Guided Exploration)**
```python
else:  # iteration > 5
    unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
    candidate_orders = []
    for _ in range(10):
        perm = unit_types.copy()
        np.random.shuffle(perm)
        candidate_orders.append(perm)
```

**What it does:**
Generates **10 random permutations** of the 5 unit types.

**Why random permutations?**
1. **GP has enough data** - Has 5+ observations to build a model
2. **Exploration** - Random sampling finds diverse candidates
3. **Computational efficiency** - Easier than enumerating all 120 permutations

**Example:**
```
Iteration 6:
  Generated 10 random candidates:
    1. op_host → defender → user → enterprise → op_server
    2. enterprise → op_server → defender → user → op_host
    3. user → op_host → op_server → enterprise → defender
    4. defender → enterprise → user → op_host → op_server
    ...
    10. op_server → user → defender → op_host → enterprise
  
  GP-UCB evaluates all 10, selects the one with highest UCB score
```

---

## 🧠 How GP-UCB Selects from Candidates

After generating candidates, GP-UCB scores each one:

```python
workflow_order, ucb_score, info = self.gp_search.select_next_order(
    candidate_orders,  # ← The 6 canonical or 10 random workflows
    self.workflow_manager
)
```

### **GP-UCB Formula:**
```
UCB(workflow) = mean(workflow) + beta × std(workflow)
                └─ Exploitation ─┘   └─ Exploration ─┘

Where:
- mean: GP's predicted reward for this workflow
- std: GP's uncertainty about this workflow  
- beta: Exploration parameter (default: 2.0)
```

**Selection:**
- Picks the workflow with **highest UCB score**
- Balances trying good workflows (high mean) vs uncertain workflows (high std)

---

## 📊 Example Workflow Selection

### **Iteration 1-5 (Canonical):**
```
Candidates: 6 canonical workflows
GP state: No data yet
Selection: Random from canonical (cold start)

Output:
  Selected: defender → op_server → enterprise → op_host → user
  Method: random
  Reason: initial_random
```

### **Iteration 6 (First Random):**
```
Candidates: 10 random permutations
GP state: Has 5 observations
GP model: Trained on those 5 data points

For each candidate:
  Workflow A: mean=-500, std=100 → UCB = -500 + 2×100 = -300
  Workflow B: mean=-400, std=50  → UCB = -400 + 2×50  = -300
  Workflow C: mean=-600, std=200 → UCB = -600 + 2×200 = -200 ← Pick this!
  ...

Output:
  Selected: Workflow C
  UCB Score: -200
  Mean Reward: -600
  Uncertainty: 200
```

High uncertainty (std=200) makes GP want to explore Workflow C!

### **Iteration 20 (Later Stage):**
```
Candidates: 10 random permutations
GP state: Has 19 observations  
GP model: Well-trained, confident predictions

For each candidate:
  Workflow X: mean=-200, std=10 → UCB = -200 + 2×10 = -180 ← Pick this!
  Workflow Y: mean=-500, std=50 → UCB = -500 + 2×50 = -400
  ...

Output:
  Selected: Workflow X
  UCB Score: -180
  Mean Reward: -200 (good!)
  Uncertainty: 10 (confident)
```

Low uncertainty means GP is confident. Picks based on good mean reward (exploitation).

---

## 🔍 Why This Two-Phase Approach?

### **Phase 1 (Iterations 1-5): Bootstrap**
- **Problem**: GP needs data to work
- **Solution**: Use hand-picked canonical workflows
- **Goal**: Get diverse initial observations

### **Phase 2 (Iterations 6+): Optimize**
- **Problem**: Search space is huge (120 possible workflows)
- **Solution**: Generate random candidates, let GP pick best
- **Goal**: Find optimal workflow efficiently

---

## 📈 Alternative Strategies (Not Used)

### **Enumerate All 120 Permutations:**
```python
# Could do this, but expensive:
from itertools import permutations
all_workflows = list(permutations(['defender', 'enterprise', 'op_server', 'op_host', 'user']))
# 120 workflows!
```

❌ Too many candidates for GP to evaluate efficiently

### **Always Random:**
```python
# Could do this:
candidate_orders = [random.shuffle(unit_types) for _ in range(10)]
```

❌ Might miss good starting points (canonical workflows are known to be reasonable)

### **Current Approach (Best):**
✅ Start with curated canonical workflows (diversity)
✅ Then use random sampling + GP guidance (efficiency)
✅ Balances exploration and computational cost

---

## 🎓 Key Insights

### **Why 10 Random Candidates?**
- More than 10: Too expensive to evaluate
- Fewer than 10: Might miss good options
- 10 is a good balance

### **Why Random Shuffle?**
- Simple and effective
- Covers different regions of search space
- GP learns patterns and guides selection

### **Why Not Just Try All 120?**
- 120 workflows × 100 episodes each = 12,000 episodes minimum
- With 100k budget, can explore ~1000 workflows this way
- Random + GP is smarter than brute force

---

## 💡 Summary

```
Iterations 1-5:  Use 6 canonical workflows (bootstrap GP)
Iterations 6+:   Generate 10 random workflows, GP picks best (optimize)
```

This balances:
- **Exploration** (trying new workflows)
- **Exploitation** (improving known good ones)
- **Efficiency** (not trying all 120 workflows)

The GP-UCB algorithm learns which workflow patterns work well and focuses the search on promising regions!

---

## 🔬 You Can Modify This!

### **More candidates per iteration:**
```python
for _ in range(20):  # Instead of 10
    perm = unit_types.copy()
    np.random.shuffle(perm)
    candidate_orders.append(perm)
```

### **Longer canonical phase:**
```python
if iteration <= 10:  # Instead of 5
    canonical_dict = ...
```

### **Different strategy:**
```python
# Focus search around best known workflow
best_workflow = get_best_so_far()
candidates = generate_neighbors(best_workflow)
```

The current approach works well in practice!
