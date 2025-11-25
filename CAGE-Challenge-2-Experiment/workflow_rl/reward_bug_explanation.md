# Reward Calculation Bug Explanation

## The Bug

The original code had a critical bug in how rewards were calculated:

```python
# BUGGY CODE:
actions, total_rewards = agent.step_and_store(
    observations, current_episode_rewards, episode_dones, true_states
)
```

The problem: `current_episode_rewards` is the **cumulative** reward for the episode so far, not the step reward!

## What Was Happening

1. Step 1: 
   - env_reward = -5
   - current_episode_rewards = -5 (cumulative)
   - Pass -5 to step_and_store
   - Alignment reward = -0.2
   - Total = -5 + -0.2 = -5.2

2. Step 2:
   - env_reward = -5
   - current_episode_rewards = -10 (cumulative from steps 1+2)
   - Pass **-10** to step_and_store (BUG!)
   - Alignment reward = -0.2
   - Total = -10 + -0.2 = -10.2

3. Step 3:
   - env_reward = -5
   - current_episode_rewards = -15 (cumulative)
   - Pass **-15** to step_and_store
   - Total = -15 + -0.2 = -15.2

After 100 steps with env_reward of -5 per step:
- Expected total: -500 (env) + ~-20 (alignment) = -520
- Actual (buggy): Cumulative env rewards grow to -500, and each step adds alignment to the cumulative!
- Result: Total rewards balloon to -20,000 to -30,000!

## The Fix Needed

The `step_and_store` method expects the **per-step** environment reward, not cumulative. However, there's a timing issue:

1. We need to call `step_and_store` to get actions and store in buffer
2. But we don't have env_rewards until AFTER we step the environment
3. The alignment rewards need to be computed based on the action and state transition

## Solution Options

### Option 1: Restructure the flow
- Get actions first (without storing)
- Step environment
- Then store with correct rewards

### Option 2: Store with placeholder, update later
- Store with zero rewards initially
- Step environment
- Update buffer with correct rewards

### Option 3: Delay reward association
- Store transitions without rewards
- Compute all rewards after stepping
- Update buffer

The key insight: The massive negative total rewards (-20,000 to -30,000) were caused by adding alignment penalties to cumulative episode rewards instead of per-step rewards!
