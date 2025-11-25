"""
General Hierarchical RL Baseline (Options Framework)
High-level: Selects abstract "options" (temporally extended actions)
Low-level: Executes primitive actions until option terminates

This is domain-agnostic and doesn't encode task-specific knowledge.
Based on the Options Framework (Sutton et al.)
"""

import os
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import torch
import torch.nn as nn
import random
import numpy as np
import csv
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HighLevelPolicy(nn.Module):
    """
    High-level policy: Selects abstract options
    Output: K options (temporal abstractions)
    """
    def __init__(self, input_dims: int = 52, n_options: int = 8):
        super(HighLevelPolicy, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_options),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )


class LowLevelPolicy(nn.Module):
    """
    Low-level policy: Executes actions given selected option
    Input: state + option embedding
    Output: primitive actions
    """
    def __init__(self, input_dims: int = 52, option_embed_dim: int = 8, n_actions: int = 145):
        super(LowLevelPolicy, self).__init__()
        
        combined_input = input_dims + option_embed_dim
        
        self.actor = nn.Sequential(
            nn.Linear(combined_input, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(combined_input, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )


class TerminationPolicy(nn.Module):
    """
    Termination policy: Decides when to switch options
    Output: probability of terminating current option
    """
    def __init__(self, input_dims: int = 52, option_embed_dim: int = 8):
        super(TerminationPolicy, self).__init__()
        
        combined_input = input_dims + option_embed_dim
        
        self.termination = nn.Sequential(
            nn.Linear(combined_input, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability
        )


def collect_options_episode(worker_id: int, scenario_path: str, red_agent_type,
                           high_weights: dict, low_weights: dict, term_weights: dict,
                           n_options: int = 8, option_embed_dim: int = 8,
                           option_duration: int = 10, max_steps: int = 100):
    """
    Collect episode using options framework
    """
    import torch
    
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.get_action_space('Blue')
    
    device_local = torch.device('cpu')
    
    # Reconstruct policies with passed parameters
    high_policy = HighLevelPolicy(obs_dim, n_options).to(device_local)
    high_policy.load_state_dict(high_weights)
    high_policy.eval()
    
    low_policy = LowLevelPolicy(obs_dim, option_embed_dim, action_dim).to(device_local)
    low_policy.load_state_dict(low_weights)
    low_policy.eval()
    
    term_policy = TerminationPolicy(obs_dim, option_embed_dim).to(device_local)
    term_policy.load_state_dict(term_weights)
    term_policy.eval()
    
    # Collect episode
    high_transitions = []
    low_transitions = []
    
    state = env.reset()
    total_reward = 0
    
    current_option = None
    option_start_state = None
    option_cumulative_reward = 0
    steps_in_option = 0
    
    for step in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device_local)
        
        # Check if we need to select new option
        if current_option is None or steps_in_option >= option_duration:
            # High-level: Select option
            with torch.no_grad():
                high_logits = high_policy.actor(state_tensor)
                high_dist = torch.distributions.Categorical(high_logits)
                current_option = high_dist.sample().item()
                high_log_prob = high_dist.log_prob(torch.tensor([current_option]))
                high_value = high_policy.critic(state_tensor)
            
            # If previous option finished, store transition
            if option_start_state is not None:
                high_transitions.append({
                    'state': option_start_state,
                    'option': current_option,
                    'reward': option_cumulative_reward,
                    'log_prob': high_log_prob.item(),
                    'value': high_value.item(),
                    'done': done if step > 0 else False
                })
            
            option_start_state = state
            option_cumulative_reward = 0
            steps_in_option = 0
        
        # Create option embedding (one-hot)
        option_onehot = torch.zeros(n_options)
        if current_option < n_options:  # Safety check
            option_onehot[current_option] = 1.0
        
        # Low-level: Select action given current option
        combined = torch.cat([state_tensor, option_onehot.unsqueeze(0).to(device_local)], dim=1)
        
        with torch.no_grad():
            low_logits = low_policy.actor(combined)
            low_dist = torch.distributions.Categorical(low_logits)
            action = low_dist.sample().item()
            low_log_prob = low_dist.log_prob(torch.tensor([action]))
            low_value = low_policy.critic(combined)
        
        # Execute
        next_state, reward, done, info = env.step(action)
        
        # Store low-level transition
        low_transitions.append({
            'state': state,
            'option': current_option,
            'action': action,
            'reward': reward,
            'log_prob': low_log_prob.item(),
            'value': low_value.item(),
            'done': done
        })
        
        option_cumulative_reward += reward
        total_reward += reward
        steps_in_option += 1
        state = next_state
        
        if done:
            # Store final high-level transition
            if option_start_state is not None:
                high_transitions.append({
                    'state': option_start_state,
                    'option': current_option,
                    'reward': option_cumulative_reward,
                    'log_prob': high_log_prob.item(),
                    'value': high_value.item(),
                    'done': True
                })
            break
    
    return {
        'high_level': high_transitions,
        'low_level': low_transitions,
        'total_reward': total_reward,
        'steps': len(low_transitions)
    }


class HierarchicalOptionsAgent:
    """Hierarchical agent using options framework"""
    
    def __init__(self, input_dims: int = 52, n_options: int = 8, 
                 option_embed_dim: int = 8, n_actions: int = 145,
                 lr: float = 0.002, gamma: float = 0.99, 
                 K_epochs: int = 6, eps_clip: float = 0.2):
        
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        # High-level policy (selects options)
        self.high_policy = HighLevelPolicy(input_dims, n_options).to(device)
        self.high_policy_old = HighLevelPolicy(input_dims, n_options).to(device)
        self.high_policy_old.load_state_dict(self.high_policy.state_dict())
        
        # Low-level policy (executes actions)
        self.low_policy = LowLevelPolicy(input_dims, option_embed_dim, n_actions).to(device)
        self.low_policy_old = LowLevelPolicy(input_dims, option_embed_dim, n_actions).to(device)
        self.low_policy_old.load_state_dict(self.low_policy.state_dict())
        
        # Termination policy
        self.term_policy = TerminationPolicy(input_dims, option_embed_dim).to(device)
        
        # Optimizers
        self.high_optimizer = torch.optim.Adam(self.high_policy.parameters(), lr=lr)
        self.low_optimizer = torch.optim.Adam(self.low_policy.parameters(), lr=lr)
        self.term_optimizer = torch.optim.Adam(self.term_policy.parameters(), lr=lr*0.1)
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, high_data, low_data):
        """Update all policies"""
        # Update high-level
        if high_data['states']:
            self._update_high_level(high_data)
        
        # Update low-level
        if low_data['states']:
            self._update_low_level(low_data)
    
    def _update_high_level(self, data):
        """PPO update for high-level policy"""
        states = torch.FloatTensor(np.array(data['states'])).to(device)
        options = torch.LongTensor(data['options']).to(device)
        old_logprobs = torch.FloatTensor(data['log_probs']).to(device)
        
        # Compute returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(data['rewards']), reversed(data['dones'])):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # PPO update
        for _ in range(self.K_epochs):
            logits = self.high_policy.actor(states)
            state_values = self.high_policy.critic(states).squeeze()
            
            dist = torch.distributions.Categorical(logits)
            new_logprobs = dist.log_prob(options)
            entropy = dist.entropy()
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, returns) - 0.01 * entropy.mean()
            
            self.high_optimizer.zero_grad()
            loss.backward()
            self.high_optimizer.step()
        
        self.high_policy_old.load_state_dict(self.high_policy.state_dict())
    
    def _update_low_level(self, data):
        """PPO update for low-level policy"""
        states = torch.FloatTensor(np.array(data['states'])).to(device)
        options_onehot = torch.FloatTensor(np.array(data['options_onehot'])).to(device)
        actions = torch.LongTensor(data['actions']).to(device)
        old_logprobs = torch.FloatTensor(data['log_probs']).to(device)
        
        # Compute returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(data['rewards']), reversed(data['dones'])):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # PPO update
        for _ in range(self.K_epochs):
            combined = torch.cat([states, options_onehot], dim=1)
            
            logits = self.low_policy.actor(combined)
            state_values = self.low_policy.critic(combined).squeeze()
            
            dist = torch.distributions.Categorical(logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, returns) - 0.01 * entropy.mean()
            
            self.low_optimizer.zero_grad()
            loss.backward()
            self.low_optimizer.step()
        
        self.low_policy_old.load_state_dict(self.low_policy.state_dict())


def train_hierarchical_options(n_workers: int = 50,
                               total_episodes: int = 20000,
                               episodes_per_update: int = 50,
                               n_options: int = 8,
                               option_duration: int = 10,
                               red_agent_type=RedMeanderAgent,
                               max_steps: int = 100,
                               scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml',
                               seed: int = 42):
    """
    Train hierarchical RL with options framework
    """
    
    print("\n" + "="*60)
    print("HIERARCHICAL RL BASELINE (Options Framework)")
    print("="*60)
    print(f"Workers: {n_workers}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Red Agent: {red_agent_type.__name__}")
    print(f"Hierarchy: General (domain-agnostic)")
    print(f"  High-level: Selects among {n_options} learned options")
    print(f"  Low-level: Executes primitive actions")
    print(f"  Option duration: ~{option_duration} steps")
    print(f"Random Seed: {seed}")
    print("="*60 + "\n")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"logs/hierarchical_options_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'experiment_name': os.path.basename(exp_dir),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'Hierarchical PPO (Options Framework)',
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
            'type': 'Options Framework (General)',
            'n_options': n_options,
            'option_duration': option_duration,
            'high_level': 'Selects abstract options (learned)',
            'low_level': 'Executes primitive actions',
            'domain_knowledge': False
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
    
    print(f"Training log: {log_filename}\n")
    
    # Get dimensions
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.get_action_space('Blue')
    
    # Create hierarchical agent
    agent = HierarchicalOptionsAgent(
        input_dims=obs_dim,
        n_options=n_options,
        option_embed_dim=n_options,  # Use option idx as embedding
        n_actions=action_dim
    )
    
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
        
        # Collect episodes
        collection_start = time.time()
        
        high_weights_cpu = {k: v.cpu() for k, v in agent.high_policy_old.state_dict().items()}
        low_weights_cpu = {k: v.cpu() for k, v in agent.low_policy_old.state_dict().items()}
        term_weights_cpu = {k: v.cpu() for k, v in agent.term_policy.state_dict().items()}
        
        futures = []
        for i in range(episodes_per_update):
            future = executor.submit(
                collect_options_episode,
                worker_id=i,
                scenario_path=scenario_path,
                red_agent_type=red_agent_type,
                high_weights=high_weights_cpu,
                low_weights=low_weights_cpu,
                term_weights=term_weights_cpu,
                n_options=n_options,
                option_embed_dim=n_options,  # Use same dim as n_options
                option_duration=option_duration,
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
        
        print(f"  Collected {len(episodes)} episodes in {collection_time:.1f}s ({rate:.1f} eps/sec)")
        
        # Aggregate data
        high_level_data = {
            'states': [],
            'options': [],
            'rewards': [],
            'dones': [],
            'log_probs': []
        }
        
        low_level_data = {
            'states': [],
            'options_onehot': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': []
        }
        
        episode_rewards = []
        
        for ep in episodes:
            # High-level
            for trans in ep['high_level']:
                high_level_data['states'].append(trans['state'])
                high_level_data['options'].append(trans['option'])
                high_level_data['rewards'].append(trans['reward'])
                high_level_data['dones'].append(trans['done'])
                high_level_data['log_probs'].append(trans['log_prob'])
            
            # Low-level
            for trans in ep['low_level']:
                low_level_data['states'].append(trans['state'])
                
                option_onehot = np.zeros(n_options)
                option_onehot[trans['option']] = 1.0
                low_level_data['options_onehot'].append(option_onehot)
                
                low_level_data['actions'].append(trans['action'])
                low_level_data['rewards'].append(trans['reward'])
                low_level_data['dones'].append(trans['done'])
                low_level_data['log_probs'].append(trans['log_prob'])
            
            episode_rewards.append(ep['total_reward'])
        
        total_episodes_collected += len(episodes)
        
        # Update both policies
        update_start = time.time()
        agent.update(high_level_data, low_level_data)
        update_time = time.time() - update_start
        
        # Statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        print(f"  Hierarchical updates complete ({update_time:.2f}s)")
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
            checkpoint_path = os.path.join(exp_dir, f'hierarchical_{total_episodes_collected}.pt')
            torch.save({
                'high_policy': agent.high_policy.state_dict(),
                'low_policy': agent.low_policy.state_dict(),
                'term_policy': agent.term_policy.state_dict()
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}\n")
    
    # Final save
    final_checkpoint = os.path.join(exp_dir, 'hierarchical_final.pt')
    torch.save({
        'high_policy': agent.high_policy.state_dict(),
        'low_policy': agent.low_policy.state_dict(),
        'term_policy': agent.term_policy.state_dict()
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
    
    parser = argparse.ArgumentParser(description='General Hierarchical RL Baseline')
    parser.add_argument('--n-workers', type=int, default=200)
    parser.add_argument('--total-episodes', type=int, default=20000)
    parser.add_argument('--episodes-per-update', type=int, default=200)
    parser.add_argument('--n-options', type=int, default=8,
                       help='Number of high-level options (abstract actions)')
    parser.add_argument('--option-duration', type=int, default=10,
                       help='Average steps per option before switching')
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
    print(f"Algorithm: Hierarchical RL (Options Framework)")
    print(f"N Options: {args.n_options}")
    print(f"Option Duration: ~{args.option_duration} steps")
    print(f"Domain Knowledge: None (general hierarchy)")
    print(f"Random Seed: {args.seed}")
    print("="*60)
    
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    train_hierarchical_options(
        n_workers=args.n_workers,
        total_episodes=args.total_episodes,
        episodes_per_update=args.episodes_per_update,
        n_options=args.n_options,
        option_duration=args.option_duration,
        red_agent_type=agent_map[args.red_agent],
        seed=args.seed
    )


if __name__ == "__main__":
    main()
