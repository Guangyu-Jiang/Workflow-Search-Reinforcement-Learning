"""
Parallel PPO Baseline Training WITHOUT workflow conditioning
Uses ProcessPoolExecutor for fast parallel collection
Full 145 action space for fair comparison with workflow search
"""

import os
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import random
import torch
import torch.nn as nn
import numpy as np
import csv
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ActorCritic(nn.Module):
    """Simple Actor-Critic network WITHOUT workflow conditioning"""
    
    def __init__(self, input_dims: int = 52, n_actions: int = 145):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        raise NotImplementedError


def collect_episode_baseline(worker_id: int, scenario_path: str, red_agent_type,
                             policy_weights_cpu: Dict, max_steps: int = 100,
                             seed: Optional[int] = None):
    """
    Worker function to collect ONE episode (no workflow conditioning)
    """
    import torch
    
    if seed is not None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

    # Create environment
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Reconstruct policy (CPU)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.get_action_space('Blue')
    
    device_local = torch.device('cpu')
    policy = ActorCritic(obs_dim, action_dim).to(device_local)
    policy.load_state_dict(policy_weights_cpu)
    policy.eval()
    
    # Collect episode
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    episode_log_probs = []
    episode_values = []
    
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device_local)
        
        with torch.no_grad():
            logits = policy.actor(state_tensor)
            dist = torch.distributions.Categorical(logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            value = policy.critic(state_tensor)
        
        action = action_tensor.item()
        next_state, reward, done, info = env.step(action)
        
        # Store
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_dones.append(done)
        episode_log_probs.append(log_prob.item())
        episode_values.append(value.item())
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return {
        'states': episode_states,
        'actions': episode_actions,
        'rewards': episode_rewards,
        'dones': episode_dones,
        'log_probs': episode_log_probs,
        'values': episode_values,
        'total_reward': total_reward,
        'steps': len(episode_states)
    }


class ParallelBaselinePPO:
    """PPO agent for parallel baseline (no workflow conditioning)"""
    
    def __init__(self, input_dims: int = 52, n_actions: int = 145,
                 lr: float = 0.002, gamma: float = 0.99, K_epochs: int = 6,
                 eps_clip: float = 0.2):
        
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        # Networks
        self.policy = ActorCritic(input_dims, n_actions).to(device)
        self.policy_old = ActorCritic(input_dims, n_actions).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
    
    def update(self, states, actions, rewards, dones, log_probs, values):
        """PPO update"""
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
        
        # Convert to tensors
        old_states = torch.FloatTensor(states).to(device)
        old_actions = torch.LongTensor(actions).to(device)
        old_logprobs = torch.FloatTensor(log_probs).to(device)
        
        # PPO update
        for _ in range(self.K_epochs):
            logits = self.policy.actor(old_states)
            state_values = self.policy.critic(old_states).squeeze()
            
            dist = torch.distributions.Categorical(logits)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, returns) - 0.01 * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())


def train_parallel_baseline(n_workers: int = 200,
                            total_episodes: int = 20000,
                            episodes_per_update: int = 200,
                            red_agent_type=B_lineAgent,
                            max_steps: int = 100,
                            scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                            seed: Optional[int] = None):
    """
    Train baseline PPO with parallel episode collection
    NO workflow conditioning - just standard PPO
    """
    
    print("\n" + "="*60)
    print("PARALLEL BASELINE PPO TRAINING")
    print("="*60)
    print(f"Workers: {n_workers}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Episodes per Update: {episodes_per_update}")
    print(f"Red Agent: {red_agent_type.__name__}")
    print(f"NO workflow conditioning (standard PPO baseline)")
    print("="*60 + "\n")

    if seed is not None:
        set_global_seed(seed)
        print(f"Seeding enabled with seed={seed}")
        print()
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"logs/parallel_baseline_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration file
    config = {
        'experiment_name': os.path.basename(exp_dir),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'Parallel Baseline PPO (No Workflow Conditioning)',
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
        'ppo_hyperparameters': {
            'K_epochs': 6,
            'learning_rate': 0.004,
            'eps_clip': 0.2,
            'gamma': 0.99
        },
        'network': {
            'input_dims': 52,
            'hidden_dims': 64,
            'output_dims': 145,
            'architecture': 'Simple Actor-Critic (NO workflow conditioning)'
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
    
    # Create agent
    agent = ParallelBaselinePPO(input_dims=obs_dim, n_actions=action_dim)
    
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
        
        policy_weights_cpu = {k: v.cpu() for k, v in agent.policy_old.state_dict().items()}
        
        futures = []
        for i in range(episodes_per_update):
            future = executor.submit(
                collect_episode_baseline,
                worker_id=i,
                scenario_path=scenario_path,
                red_agent_type=red_agent_type,
                policy_weights_cpu=policy_weights_cpu,
                max_steps=max_steps,
                seed=seed
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
        
        # Aggregate data
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        episode_rewards = []
        
        for ep in episodes:
            all_states.extend(ep['states'])
            all_actions.extend(ep['actions'])
            all_rewards.extend(ep['rewards'])
            all_dones.extend(ep['dones'])
            all_log_probs.extend(ep['log_probs'])
            all_values.extend(ep['values'])
            episode_rewards.append(ep['total_reward'])
        
        total_episodes_collected += len(episodes)
        
        print(f"  Collected {len(episodes)} episodes in {collection_time:.1f}s ({rate:.1f} eps/sec)")
        
        # PPO update
        update_start = time.time()
        agent.update(
            np.array(all_states),
            np.array(all_actions),
            np.array(all_rewards),
            np.array(all_dones),
            np.array(all_log_probs),
            np.array(all_values)
        )
        update_time = time.time() - update_start
        
        # Statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        print(f"  PPO update complete ({update_time:.2f}s)")
        print(f"  Episodes: {total_episodes_collected}/{total_episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Reward Range: [{min_reward:.2f}, {max_reward:.2f}]")
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
            checkpoint_path = os.path.join(exp_dir, f'baseline_agent_{total_episodes_collected}.pt')
            torch.save(agent.policy.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}\n")
    
    # Final save
    final_checkpoint = os.path.join(exp_dir, 'baseline_agent_final.pt')
    torch.save(agent.policy.state_dict(), final_checkpoint)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print(f"   Total episodes: {total_episodes_collected}")
    print(f"   Final checkpoint: {final_checkpoint}")
    print("="*60)
    
    log_file.close()
    executor.shutdown(wait=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel Baseline PPO Training (No Workflow)')
    parser.add_argument('--n-workers', type=int, default=200)
    parser.add_argument('--total-episodes', type=int, default=20000)
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
    print(f"Algorithm: Standard PPO (NO workflow conditioning)")
    print(f"Action Space: Full 145 actions")
    print("="*60)
    
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    train_parallel_baseline(
        n_workers=args.n_workers,
        total_episodes=args.total_episodes,
        episodes_per_update=args.episodes_per_update,
        red_agent_type=agent_map[args.red_agent],
        seed=args.seed
    )


if __name__ == "__main__":
    main()
