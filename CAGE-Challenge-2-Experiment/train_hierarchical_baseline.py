"""
Hierarchical RL Baseline Training (2-Level Hierarchy)
High-level: Selects which unit type to fix next
Low-level: Selects specific action for that unit type

Uses ProcessPoolExecutor for parallel collection
Full 145 action space for fair comparison with workflow search
"""

import os
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import torch
import torch.nn as nn
import numpy as np
import csv
import json
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    """Apply the same seed across Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# High-level policy: Selects unit type
class HighLevelPolicy(nn.Module):
    def __init__(self, input_dims: int = 52, n_unit_types: int = 5):
        super(HighLevelPolicy, self).__init__()
        
        # Actor: state -> unit type distribution
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_unit_types),
            nn.Softmax(dim=-1)
        )
        
        # Critic: state -> value
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )


# Low-level policy: Selects action given unit type
class LowLevelPolicy(nn.Module):
    def __init__(self, input_dims: int = 52, n_unit_types: int = 5, n_actions: int = 145):
        super(LowLevelPolicy, self).__init__()
        
        # Input: state + one-hot unit type
        combined_input = input_dims + n_unit_types
        
        # Actor: (state, unit_type) -> action distribution
        self.actor = nn.Sequential(
            nn.Linear(combined_input, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic: (state, unit_type) -> value
        self.critic = nn.Sequential(
            nn.Linear(combined_input, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )


# Unit type to action mapping
UNIT_TYPES = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
UNIT_TO_ACTIONS = {
    'defender': [15, 132],
    'enterprise': [16, 17, 18, 133, 134, 135],
    'op_host': [19, 20, 21, 136, 137, 138],
    'op_server': [22, 139],
    'user': [23, 24, 25, 26, 27, 140, 141, 142, 143, 144]
}


def collect_hierarchical_episode(worker_id: int, scenario_path: str, red_agent_type,
                                 high_level_weights: Dict, low_level_weights: Dict,
                                 max_steps: int = 100, episode_seed: Optional[int] = None):
    """
    Worker function for hierarchical episode collection
    """
    import torch
    import random
    
    # Create environment
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.get_action_space('Blue')
    
    if episode_seed is not None:
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
    
    device_local = torch.device('cpu')
    
    # Reconstruct policies
    high_policy = HighLevelPolicy(obs_dim).to(device_local)
    high_policy.load_state_dict(high_level_weights)
    high_policy.eval()
    
    low_policy = LowLevelPolicy(obs_dim).to(device_local)
    low_policy.load_state_dict(low_level_weights)
    low_policy.eval()
    
    # Collect episode
    high_level_transitions = []
    low_level_transitions = []
    
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device_local)
        
        # High-level: Select unit type
        with torch.no_grad():
            high_logits = high_policy.actor(state_tensor)
            high_dist = torch.distributions.Categorical(high_logits)
            unit_type_idx = high_dist.sample().item()
            high_log_prob = high_dist.log_prob(torch.tensor([unit_type_idx]))
            high_value = high_policy.critic(state_tensor)
        
        # Create one-hot encoding for unit type
        unit_type_onehot = torch.zeros(5)
        unit_type_onehot[unit_type_idx] = 1.0
        
        # Low-level: Select action given unit type
        combined_input = torch.cat([state_tensor, unit_type_onehot.unsqueeze(0).to(device_local)], dim=1)
        
        with torch.no_grad():
            low_logits = low_policy.actor(combined_input)
            
            # Mask actions not belonging to selected unit type
            selected_unit = UNIT_TYPES[unit_type_idx]
            valid_actions = UNIT_TO_ACTIONS[selected_unit]
            mask = torch.ones(action_dim) * -1e10
            for action_idx in valid_actions:
                mask[action_idx] = 0.0
            
            masked_logits = low_logits + mask.to(device_local)
            low_dist = torch.distributions.Categorical(logits=masked_logits)
            action = low_dist.sample().item()
            low_log_prob = low_dist.log_prob(torch.tensor([action]))
            low_value = low_policy.critic(combined_input)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Store transitions
        high_level_transitions.append({
            'state': state,
            'unit_type': unit_type_idx,
            'reward': reward,  # High-level gets full reward
            'log_prob': high_log_prob.item(),
            'value': high_value.item(),
            'done': done
        })
        
        low_level_transitions.append({
            'state': state,
            'unit_type': unit_type_idx,
            'action': action,
            'reward': reward,  # Low-level gets immediate reward
            'log_prob': low_log_prob.item(),
            'value': low_value.item(),
            'done': done
        })
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return {
        'high_level': high_level_transitions,
        'low_level': low_level_transitions,
        'total_reward': total_reward,
        'steps': len(high_level_transitions)
    }


class HierarchicalPPO:
    """Hierarchical PPO with high-level and low-level policies"""
    
    def __init__(self, input_dims: int = 52, n_actions: int = 145,
                 lr: float = 0.002, gamma: float = 0.99, K_epochs: int = 6,
                 eps_clip: float = 0.2):
        
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        # High-level policy
        self.high_policy = HighLevelPolicy(input_dims).to(device)
        self.high_policy_old = HighLevelPolicy(input_dims).to(device)
        self.high_policy_old.load_state_dict(self.high_policy.state_dict())
        
        # Low-level policy
        self.low_policy = LowLevelPolicy(input_dims).to(device)
        self.low_policy_old = LowLevelPolicy(input_dims).to(device)
        self.low_policy_old.load_state_dict(self.low_policy.state_dict())
        
        # Optimizers
        self.high_optimizer = torch.optim.Adam(self.high_policy.parameters(), lr=lr)
        self.low_optimizer = torch.optim.Adam(self.low_policy.parameters(), lr=lr)
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, high_data, low_data):
        """Update both policies"""
        # Update high-level policy
        if high_data:
            self._update_policy(
                high_data,
                self.high_policy,
                self.high_policy_old,
                self.high_optimizer,
                is_high_level=True
            )
        
        # Update low-level policy
        if low_data:
            self._update_policy(
                low_data,
                self.low_policy,
                self.low_policy_old,
                self.low_optimizer,
                is_high_level=False
            )
    
    def _update_policy(self, data, policy, policy_old, optimizer, is_high_level):
        """PPO update for one policy"""
        states = torch.FloatTensor(data['states']).to(device)
        actions = torch.LongTensor(data['actions']).to(device)
        old_logprobs = torch.FloatTensor(data['log_probs']).to(device)
        rewards = data['rewards']
        dones = data['dones']
        
        # Compute returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # PPO update
        for _ in range(self.K_epochs):
            if is_high_level:
                logits = policy.actor(states)
            else:
                # Low-level needs unit type as input
                unit_types = torch.FloatTensor(data['unit_types']).to(device)
                combined = torch.cat([states, unit_types], dim=1)
                logits = policy.actor(combined)
            
            if is_high_level:
                state_values = policy.critic(states).squeeze()
            else:
                unit_types = torch.FloatTensor(data['unit_types']).to(device)
                combined = torch.cat([states, unit_types], dim=1)
                state_values = policy.critic(combined).squeeze()
            
            dist = torch.distributions.Categorical(logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, returns) - 0.01 * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update old policy
        policy_old.load_state_dict(policy.state_dict())


def train_hierarchical(n_workers: int = 200,
                      total_episodes: int = 100000,
                      episodes_per_update: int = 200,
                      red_agent_type=B_lineAgent,
                      max_steps: int = 100,
                      scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                      seed: Optional[int] = None):
    """
    Train hierarchical RL baseline
    """
    
    print("\n" + "="*60)
    print("HIERARCHICAL RL BASELINE TRAINING")
    print("="*60)
    print(f"Workers: {n_workers}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Episodes per Update: {episodes_per_update}")
    print(f"Red Agent: {red_agent_type.__name__}")
    print(f"High-level: Selects unit type to fix")
    print(f"Low-level: Selects specific action")
    print(f"Seed: {seed if seed is not None else 'None'}")
    print("="*60 + "\n")

    if seed is not None:
        set_global_seed(seed)
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"logs/hierarchical_baseline_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'experiment_name': os.path.basename(exp_dir),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'Hierarchical PPO (2-Level)',
        'environment': {
            'n_workers': n_workers,
            'max_steps': max_steps,
            'red_agent_type': red_agent_type.__name__,
            'scenario': scenario_path
        },
        'training': {
            'total_episodes': total_episodes,
            'episodes_per_update': episodes_per_update,
            'seed': seed
        },
        'hierarchy': {
            'high_level': 'Selects unit type (5 actions)',
            'low_level': 'Selects specific action for unit type',
            'n_unit_types': 5,
            'n_actions': 145
        },
        'ppo_hyperparameters': {
            'K_epochs': 6,
            'learning_rate': 0.002,
            'eps_clip': 0.2,
            'gamma': 0.99
        }
    }
    
    config_file = os.path.join(exp_dir, 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment config: {config_file}")
    
    # Training log
    log_filename = os.path.join(exp_dir, "training_log.csv")
    log_file = open(log_filename, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['Episode', 'Avg_Reward', 'Std_Reward', 'Min_Reward', 'Max_Reward', 
                        'Collection_Time', 'Update_Time'])
    log_file.flush()
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Training log: {log_filename}\n")
    
    # Get dimensions
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.get_action_space('Blue')
    
    # Create hierarchical agent
    agent = HierarchicalPPO(input_dims=obs_dim, n_actions=action_dim)
    
    # Create ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=n_workers)
    print(f"Created ProcessPoolExecutor with {n_workers} workers\n")
    
    # Training loop
    total_episodes_collected = 0
    update_num = 0
    
    import time
    
    while total_episodes_collected < total_episodes:
        update_num += 1
        
        print(f"Update {update_num}: Collecting {episodes_per_update} episodes...")
        
        # Collect episodes in parallel
        collection_start = time.time()
        
        high_weights_cpu = {k: v.cpu() for k, v in agent.high_policy_old.state_dict().items()}
        low_weights_cpu = {k: v.cpu() for k, v in agent.low_policy_old.state_dict().items()}
        
        futures = []
        for i in range(episodes_per_update):
            episode_seed = None if seed is None else seed + total_episodes_collected + i
            future = executor.submit(
                collect_hierarchical_episode,
                worker_id=i,
                scenario_path=scenario_path,
                red_agent_type=red_agent_type,
                high_level_weights=high_weights_cpu,
                low_level_weights=low_weights_cpu,
                max_steps=max_steps,
                episode_seed=episode_seed
            )
            futures.append(future)
        
        # Collect results
        episodes = []
        collected = 0
        for future in as_completed(futures):
            episode = future.result()
            episodes.append(episode)
            collected += 1
            
            if collected % 50 == 0:
                elapsed = time.time() - collection_start
                rate = collected / elapsed
                print(f"  {collected}/{episodes_per_update} episodes ({rate:.1f} eps/sec)")
        
        collection_time = time.time() - collection_start
        rate = episodes_per_update / collection_time
        
        print(f"  Collected {len(episodes)} episodes in {collection_time:.1f}s ({rate:.1f} eps/sec)")
        
        # Aggregate transitions
        high_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': []
        }
        
        low_data = {
            'states': [],
            'unit_types': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': []
        }
        
        episode_rewards = []
        
        for ep in episodes:
            # High-level data
            for trans in ep['high_level']:
                high_data['states'].append(trans['state'])
                high_data['actions'].append(trans['unit_type'])
                high_data['rewards'].append(trans['reward'])
                high_data['dones'].append(trans['done'])
                high_data['log_probs'].append(trans['log_prob'])
            
            # Low-level data
            for trans in ep['low_level']:
                low_data['states'].append(trans['state'])
                
                # One-hot encode unit type
                unit_onehot = np.zeros(5)
                unit_onehot[trans['unit_type']] = 1.0
                low_data['unit_types'].append(unit_onehot)
                
                low_data['actions'].append(trans['action'])
                low_data['rewards'].append(trans['reward'])
                low_data['dones'].append(trans['done'])
                low_data['log_probs'].append(trans['log_prob'])
            
            episode_rewards.append(ep['total_reward'])
        
        total_episodes_collected += len(episodes)
        
        # Update both policies
        update_start = time.time()
        agent.update(high_data, low_data)
        update_time = time.time() - update_start
        
        # Statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        print(f"  PPO updates complete ({update_time:.2f}s)")
        print(f"  Episodes: {total_episodes_collected}/{total_episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print()
        
        # Log
        csv_writer.writerow([
            total_episodes_collected,
            f"{avg_reward:.2f}",
            f"{std_reward:.2f}",
            f"{min_reward:.2f}",
            f"{max_reward:.2f}",
            f"{collection_time:.2f}",
            f"{update_time:.2f}"
        ])
        log_file.flush()
        
        # Save checkpoint periodically
        if update_num % 10 == 0:
            checkpoint_path = os.path.join(exp_dir, f'hierarchical_agent_{total_episodes_collected}.pt')
            torch.save({
                'high_policy': agent.high_policy.state_dict(),
                'low_policy': agent.low_policy.state_dict()
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}\n")
    
    # Final save
    final_checkpoint = os.path.join(exp_dir, 'hierarchical_agent_final.pt')
    torch.save({
        'high_policy': agent.high_policy.state_dict(),
        'low_policy': agent.low_policy.state_dict()
    }, final_checkpoint)
    
    print("\n" + "="*60)
    print("✅ Hierarchical RL Training Complete!")
    print(f"   Total episodes: {total_episodes_collected}")
    print(f"   Final checkpoint: {final_checkpoint}")
    print("="*60)
    
    log_file.close()
    executor.shutdown(wait=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hierarchical RL Baseline Training')
    parser.add_argument('--n-workers', type=int, default=200)
    parser.add_argument('--total-episodes', type=int, default=100000)
    parser.add_argument('--episodes-per-update', type=int, default=200)
    parser.add_argument('--red-agent', type=str, default='B_lineAgent',
                       choices=['B_lineAgent', 'RedMeanderAgent', 'SleepAgent'])
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    
    args = parser.parse_args()
    
    agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    print(f"Red Agent: {args.red_agent}")
    print(f"Workers: {args.n_workers}")
    print(f"Total Episodes: {args.total_episodes}")
    print(f"Episodes per Update: {args.episodes_per_update}")
    print(f"Seed: {args.seed}")
    print(f"Algorithm: Hierarchical PPO (2-Level)")
    print(f"Action Space: Full 145 actions")
    print("="*60)
    
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    train_hierarchical(
        n_workers=args.n_workers,
        total_episodes=args.total_episodes,
        episodes_per_update=args.episodes_per_update,
        red_agent_type=agent_map[args.red_agent],
        seed=args.seed
    )


if __name__ == "__main__":
    main()
