"""
Parallel SAC (Soft Actor-Critic) Baseline Training WITHOUT workflow conditioning
Uses ProcessPoolExecutor for fast parallel collection
Full 145 action space for fair comparison with workflow search
"""

import os
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
from collections import deque
import random

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class SACActorCritic(nn.Module):
    """SAC Actor-Critic networks (discrete action space)"""
    
    def __init__(self, input_dims: int = 52, n_actions: int = 145):
        super(SACActorCritic, self).__init__()
        hidden = 64  # Align depth/width with PPO baselines
        
        # Actor network (outputs action probabilities)
        self.actor = nn.Sequential(
            nn.Linear(input_dims, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
        
        # Q-network 1
        self.q1 = nn.Sequential(
            nn.Linear(input_dims, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
        
        # Q-network 2
        self.q2 = nn.Sequential(
            nn.Linear(input_dims, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )


def collect_episode_sac(worker_id: int, scenario_path: str, red_agent_type,
                       policy_weights_cpu: Dict, max_steps: int = 100):
    """
    Worker function to collect ONE episode for SAC
    """
    import torch
    
    # Create environment
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    # Reconstruct policy (CPU)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.get_action_space('Blue')
    
    device_local = torch.device('cpu')
    policy_net = SACActorCritic(obs_dim, action_dim).to(device_local)
    policy_net.load_state_dict(policy_weights_cpu)
    policy_net.eval()
    
    # Collect episode
    transitions = []
    
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device_local)
        
        with torch.no_grad():
            # SAC uses softmax policy for discrete actions
            logits = policy_net.actor(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return {
        'transitions': transitions,
        'total_reward': total_reward,
        'steps': len(transitions)
    }


class ParallelSAC:
    """SAC agent for parallel training (discrete actions)"""
    
    def __init__(self, input_dims: int = 52, n_actions: int = 145,
                 lr: float = 0.0003, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, buffer_size: int = 100000):
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Entropy coefficient
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = SACActorCritic(input_dims, n_actions).to(device)
        
        # Target Q-networks
        self.target_q1 = SACActorCritic(input_dims, n_actions).to(device)
        self.target_q2 = SACActorCritic(input_dims, n_actions).to(device)
        self.target_q1.load_state_dict(self.policy_net.state_dict())
        self.target_q2.load_state_dict(self.policy_net.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy_net.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.policy_net.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.policy_net.q2.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def update(self, batch_size: int = 256):
        """SAC update step"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Update Q-networks
        with torch.no_grad():
            # Next action probabilities
            next_logits = self.policy_net.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            
            # Target Q values
            next_q1 = self.target_q1.q1(next_states)
            next_q2 = self.target_q2.q2(next_states)
            next_q = torch.min(next_q1, next_q2)
            
            # Entropy term
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.gamma * next_v
        
        # Q1 loss
        current_q1 = self.policy_net.q1(states).gather(1, actions.unsqueeze(-1)).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Q2 loss
        current_q2 = self.policy_net.q2(states).gather(1, actions.unsqueeze(-1)).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        logits = self.policy_net.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        q1_values = self.policy_net.q1(states)
        q2_values = self.policy_net.q2(states)
        q_values = torch.min(q1_values, q2_values)
        
        # Policy loss (maximize Q - alpha * log_prob)
        policy_loss = (probs * (self.alpha * log_probs - q_values)).sum(dim=-1).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.target_q1.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q2.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train_parallel_sac(n_workers: int = 200,
                      total_episodes: int = 100000,
                      episodes_per_update: int = 200,
                      batch_size: int = 250,
                      updates_per_step: int = 1,
                      red_agent_type=B_lineAgent,
                      max_steps: int = 100,
                      scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                      seed: int = 42):
    """
    Train SAC with parallel episode collection
    """
    
    print("\n" + "="*60)
    print("PARALLEL SAC BASELINE TRAINING")
    print("="*60)
    print(f"Workers: {n_workers}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Episodes per Collection: {episodes_per_update}")
    print(f"Red Agent: {red_agent_type.__name__}")
    print(f"Algorithm: SAC (Soft Actor-Critic)")
    print(f"NO workflow conditioning (standard SAC baseline)")
    print("="*60 + "\n")
    print(f"Random Seed: {seed}\n")
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"logs/parallel_sac_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'experiment_name': os.path.basename(exp_dir),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'Parallel SAC (No Workflow Conditioning)',
        'environment': {
            'n_workers': n_workers,
            'max_steps': max_steps,
            'red_agent_type': red_agent_type.__name__,
            'scenario': scenario_path
        },
        'training': {
            'total_episodes': total_episodes,
            'episodes_per_update': episodes_per_update,
            'batch_size': batch_size,
            'updates_per_step': updates_per_step,
            'seed': seed
        },
        'sac_hyperparameters': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'buffer_size': 100000
        },
        'network': {
            'input_dims': 52,
            'hidden_dims': 64,
            'output_dims': 145,
            'architecture': 'SAC with dual Q-networks (64x64 MLP)'
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
    
    # Create SAC agent
    agent = ParallelSAC(input_dims=obs_dim, n_actions=action_dim)
    
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
        
        policy_weights_cpu = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
        
        futures = []
        for i in range(episodes_per_update):
            future = executor.submit(
                collect_episode_sac,
                worker_id=i,
                scenario_path=scenario_path,
                red_agent_type=red_agent_type,
                policy_weights_cpu=policy_weights_cpu,
                max_steps=max_steps
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
        
        # Add to replay buffer and get episode rewards
        episode_rewards = []
        for ep in episodes:
            for trans in ep['transitions']:
                agent.replay_buffer.add(
                    trans['state'],
                    trans['action'],
                    trans['reward'],
                    trans['next_state'],
                    trans['done']
                )
            episode_rewards.append(ep['total_reward'])
        
        total_episodes_collected += len(episodes)
        
        print(f"  Collected {len(episodes)} episodes in {collection_time:.1f}s ({rate:.1f} eps/sec)")
        
        # SAC updates (multiple updates per collection)
        update_start = time.time()
        n_updates = len(episodes) * max_steps * updates_per_step  # One update per transition
        
        if len(agent.replay_buffer) >= batch_size:
            for _ in range(n_updates):
                agent.update(batch_size)
        
        update_time = time.time() - update_start
        
        # Statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        print(f"  SAC updates complete ({update_time:.2f}s, {n_updates} updates)")
        print(f"  Episodes: {total_episodes_collected}/{total_episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Replay Buffer: {len(agent.replay_buffer)} transitions")
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
            checkpoint_path = os.path.join(exp_dir, f'sac_agent_{total_episodes_collected}.pt')
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}\n")
    
    # Final save
    final_checkpoint = os.path.join(exp_dir, 'sac_agent_final.pt')
    torch.save(agent.policy_net.state_dict(), final_checkpoint)
    
    print("\n" + "="*60)
    print("✅ SAC Training Complete!")
    print(f"   Total episodes: {total_episodes_collected}")
    print(f"   Final checkpoint: {final_checkpoint}")
    print("="*60)
    
    log_file.close()
    executor.shutdown(wait=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel SAC Baseline Training (No Workflow)')
    parser.add_argument('--n-workers', type=int, default=50)
    parser.add_argument('--total-episodes', type=int, default=100000)
    parser.add_argument('--episodes-per-update', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--red-agent', type=str, default='B_lineAgent',
                       choices=['B_lineAgent', 'RedMeanderAgent', 'SleepAgent'])
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
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
    print(f"Algorithm: SAC (Soft Actor-Critic)")
    print(f"Action Space: Full 145 actions")
    print(f"Batch Size: {args.batch_size}")
    print(f"Random Seed: {args.seed}")
    print("="*60)
    
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    train_parallel_sac(
        n_workers=args.n_workers,
        total_episodes=args.total_episodes,
        episodes_per_update=args.episodes_per_update,
        batch_size=args.batch_size,
        red_agent_type=agent_map[args.red_agent],
        seed=args.seed
    )


if __name__ == "__main__":
    main()
