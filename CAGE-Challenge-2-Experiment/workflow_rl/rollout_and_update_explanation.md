# Rollout and Model Update Strategy

## Current Implementation vs Original train.py

### Original train.py Approach
```python
update_timesteps = 20000  # Update after 20,000 steps!
max_timesteps = 100      # 100 steps per episode

# Collects ~200 episodes before updating
for episode in range(max_episodes):
    for step in range(100):
        action = agent.get_action(state)
        state, reward, done = env.step(action)
        agent.store(reward, done)  # Store in buffer
        
        time_step += 1
        if time_step % 20000 == 0:  # After 200 episodes
            agent.train()      # PPO update on entire buffer
            agent.clear_memory()
            time_step = 0
```

**Key Points:**
- Collects **200 episodes** (20,000 steps) before updating
- One big batch update every 20,000 steps
- Memory cleared after update

### Our Current Implementation
```python
# Update after EVERY episode (1 episode = 1 trajectory)
for episode in range(train_episodes):
    for step in range(100):
        action = agent.get_action(obs)
        next_obs, reward, done = env.step(action)
        
        # Store transition in memory
        agent.memory.rewards.append(reward)
        agent.memory.is_terminals.append(done)
    
    # Update after each episode
    agent.update()  # PPO update on 1 episode
    agent.memory.clear_memory()
```

**Key Points:**
- Updates after **EVERY episode** (100 steps)
- Small batch updates (1 trajectory at a time)
- More frequent but smaller updates

## Comparison

| Aspect | Original train.py | Our Implementation |
|--------|------------------|-------------------|
| **Rollout Size** | 200 episodes (20,000 steps) | 1 episode (100 steps) |
| **Update Frequency** | Every 200 episodes | Every episode |
| **Batch Size** | ~20,000 transitions | ~100 transitions |
| **Memory Usage** | High (stores 200 episodes) | Low (stores 1 episode) |
| **Convergence** | Stable but slow | Faster but noisier |
| **Sample Efficiency** | High (reuses data K times) | Low (small batches) |

## Why Different Approaches?

### Original train.py Goals:
- **Stable training** over millions of steps
- **Sample efficiency** with large batches
- **Smooth convergence** with averaged gradients

### Our Workflow Search Goals:
- **Quick adaptation** to new workflows
- **Fast evaluation** of workflow effectiveness
- **Early stopping** when compliance achieved
- **Exploration** of many workflows

## Updated Parallel Implementation (NEW)

With 25 parallel environments, we now collect **25 full episodes** before each update:

```python
# Collect 25 parallel episodes (2500 transitions)
for step in range(100):  # Full episode
    actions = agent.get_actions(all_25_observations)
    all_25_observations, rewards, dones = envs.step(actions)
    buffer.store(observations, actions, rewards)

# Update on 2500 diverse transitions
agent.update()  # Much more stable gradients!
```

**Key Benefits:**
- **2500 transitions per update** (25× more than before)
- **25 different trajectories** (massive diversity)
- **Very stable gradients** (approaching original train.py's stability)
- **Still fast exploration** (parallel execution)

## Current Update Mechanism

```python
def update(self):
    """PPO update after each episode"""
    # 1. Calculate returns (Monte Carlo)
    rewards = []
    discounted_reward = 0
    for reward, is_terminal in reversed(memory):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + gamma * discounted_reward
        rewards.insert(0, discounted_reward)
    
    # 2. Normalize rewards
    rewards = (rewards - mean) / std
    
    # 3. PPO optimization (K epochs)
    for _ in range(K_epochs):  # Default K=4
        # Calculate advantages
        advantages = rewards - values
        
        # PPO clipped objective
        ratio = exp(log_prob - old_log_prob)
        clipped_ratio = clip(ratio, 1-eps, 1+eps)
        loss = -min(ratio * advantages, clipped_ratio * advantages)
        
        # Update network
        optimizer.step(loss)
```

## Pros and Cons

### Current Approach (1 Episode Updates)

**Pros:**
- ✅ Fast initial learning
- ✅ Quick workflow compliance
- ✅ Low memory footprint
- ✅ Enables early stopping
- ✅ Good for exploration

**Cons:**
- ❌ Noisy gradient estimates
- ❌ Less stable training
- ❌ Poor sample efficiency
- ❌ May overfit to single episodes

### Alternative: Mini-Batch Updates

```python
# Collect 10 episodes before updating
buffer_size = 10
episode_buffer = []

for episode in range(train_episodes):
    trajectory = collect_episode()
    episode_buffer.append(trajectory)
    
    if len(episode_buffer) >= buffer_size:
        agent.update(episode_buffer)  # Update on 10 episodes
        episode_buffer.clear()
```

**Benefits:**
- More stable gradients
- Better sample efficiency
- Still allows early stopping

## Recommendation

For workflow search, the current approach (1 episode updates) is **appropriate** because:

1. **Workflow Evaluation Priority**: We need quick assessment of whether a workflow is viable
2. **Early Stopping**: Need frequent compliance checks
3. **Exploration Focus**: Testing many workflows is more important than perfecting one
4. **Computational Budget**: Limited episodes per workflow (50 max)

However, if you wanted more stable training, you could:
- Increase to 5-10 episode batches
- Use experience replay
- Implement parallel environments

## Summary

**Current Implementation:**
- **1 trajectory (episode) per update**
- **~100 steps per trajectory**
- **Updates after every episode**
- **Optimized for quick workflow evaluation**

This is intentionally different from the original train.py's 200-episode batches because we prioritize fast workflow exploration over stable long-term training.
