# PPO Training Configuration Details

## Action Space Handling

### ✅ **FULL Action Space Used** (Not Restricted!)

Our workflow-conditioned PPO uses the **complete CAGE2 action space**:
```python
# In WorkflowConditionedPPO.__init__():
if action_space is None:
    self.action_space = list(range(145))  # Full CAGE2 action space
```

This is a **key difference** from the original training script which used a reduced action space:

#### Original train.py (Reduced):
```python
# Only 28 actions selected from 145 total
action_space = [133, 134, 135, 139]  # restore enterprise and opserver
action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
action_space += [16, 17, 18, 22]  # remove enterprise and opserer
action_space += [11, 12, 13, 14]  # analyse user hosts
action_space += [141, 142, 143, 144]  # restore user hosts
action_space += [132]  # restore defender
action_space += [2]  # analyse defender
action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts
```
- **28 actions** carefully selected
- Ignores Op_Hosts completely
- Limited to specific defense strategies

#### Our Workflow-Conditioned PPO (Full):
```python
self.action_space = list(range(145))  # All 145 actions available
```
- **145 actions** - complete action space
- Can defend ALL hosts including Op_Hosts
- No artificial limitations on strategy
- Workflow guides priority, not action availability

## Why Full Action Space Matters

1. **Generality**: Our approach doesn't assume which hosts are important
2. **Flexibility**: Different workflows may prioritize different hosts
3. **Completeness**: Can discover strategies the reduced space might miss
4. **Op_Host Defense**: Unlike original, we can defend Op_Hosts (they have Medium value!)

## PPO Architecture Details

### Network Structure

```python
class WorkflowConditionedActorCritic(nn.Module):
    def __init__(self, input_dims=52, n_actions=145, workflow_dim=8):
        # Input: state (52) + workflow (8) = 60 dimensions
        augmented_input = input_dims + workflow_dim
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(60, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 145),  # Output: full action space
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(60, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
```

### Key Differences from Original PPO

| Aspect | Original PPO | Workflow-Conditioned PPO |
|--------|-------------|-------------------------|
| Action Space | 28 actions (reduced) | 145 actions (full) |
| Input Dims | 52 (state only) | 60 (52 state + 8 workflow) |
| Network Input | State vector | Concatenated state + workflow |
| Reward | Environment reward only | Environment + alignment reward |
| Start Actions | Hardcoded sequence | None (learns from scratch) |
| Op_Host Defense | Not possible | Fully supported |

## PPO Hyperparameters

```python
# Standard PPO parameters (same as original)
lr = 0.002              # Learning rate
betas = [0.9, 0.990]    # Adam optimizer betas
gamma = 0.99            # Discount factor
K_epochs = 4            # PPO update epochs
eps_clip = 0.2          # PPO clipping parameter

# Workflow-specific parameters (new)
alignment_alpha = 0.1   # Bonus weight for aligned fixes
alignment_beta = 0.2    # Penalty weight for violations
workflow_dim = 8        # Workflow embedding dimension
```

## Training Process

### Per Episode:
1. **Reset environment** with same Red agent type
2. **Concatenate** state (52D) + workflow (8D) → 60D input
3. **Sample actions** from policy network (145-way softmax)
4. **Calculate rewards**:
   ```python
   total_reward = env_reward + alignment_reward
   
   alignment_reward = (
       alpha * aligned_fixes +    # Bonus for following priority
       -beta * violations         # Penalty for wrong order
   )
   ```
5. **Store in memory** for batch update
6. **Update policy** every K episodes using PPO loss

### Action Selection:
```python
def get_action(self, obs):
    # Augment observation with workflow
    augmented_state = torch.cat([state, workflow], dim=-1)
    
    # Get action probabilities from actor network
    action_probs = self.actor(augmented_state)  # Shape: [145]
    
    # Sample from categorical distribution
    dist = Categorical(action_probs)
    action = dist.sample()  # Range: 0-144
    
    return action.item()
```

## Advantages of Full Action Space

### 1. **Complete Defense Coverage**
- Can protect ALL 13 hosts
- No blind spots in defense strategy

### 2. **Workflow Flexibility**
- Different workflows can utilize different action subsets
- Natural emergence of specialized strategies

### 3. **Discovery Potential**
- May find novel defense patterns
- Not constrained by human preconceptions

### 4. **True Workflow Testing**
- Tests if workflow guidance alone is sufficient
- No need for action space engineering

### 5. **Fair Comparison**
- All workflows compete on equal terms
- Performance differences reflect workflow quality, not action restrictions

## Memory and Batch Processing

```python
class Memory:
    def __init__(self):
        self.actions = []      # Selected action IDs (0-144)
        self.states = []       # Augmented states (60D)
        self.logprobs = []     # Log probabilities of selected actions
        self.rewards = []      # Total rewards (env + alignment)
        self.is_terminals = [] # Episode termination flags
        
    def clear_memory(self):
        # Clear after PPO update
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
```

## Computational Considerations

Using the full action space does increase computational requirements:

- **Larger output layer**: 145 vs 28 outputs
- **More exploration needed**: Larger action space to search
- **Slower initial learning**: Takes longer to eliminate bad actions

However, this is offset by:
- **Workflow guidance**: Reduces effective search space
- **Alignment rewards**: Provides strong learning signal
- **GP-UCB efficiency**: Focuses on promising workflows

## Summary

Our PPO implementation maintains full generality by using the complete 145-action space, allowing the workflow guidance and alignment rewards to shape the policy rather than artificially constraining the available actions. This design choice ensures that our framework can discover novel defense strategies and fairly evaluate different workflow orderings.
