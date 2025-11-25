"""
Random Policy Baseline
Selects actions uniformly at random - lower bound on performance
"""

import os
import sys
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

import numpy as np
import csv
from datetime import datetime
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2


def run_random_policy(n_episodes: int = 1000,
                     max_steps: int = 100,
                     red_agent_type=B_lineAgent,
                     scenario_path: str = '/home/ubuntu/CAGE2/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'):
    """
    Run random policy baseline
    """
    
    print("\n" + "="*60)
    print("RANDOM POLICY BASELINE")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Red Agent: {red_agent_type.__name__}")
    print(f"Policy: Uniform random action selection")
    print("="*60 + "\n")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"logs/random_policy_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create environment
    cyborg = CybORG(scenario_path, 'sim', agents={'Red': red_agent_type})
    env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    action_space = env.get_action_space('Blue')
    
    # Logging
    log_file = open(os.path.join(exp_dir, 'training_log.csv'), 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['Episode', 'Reward', 'Steps'])
    
    print(f"Experiment directory: {exp_dir}\n")
    
    # Run episodes
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Random action selection
            action = np.random.randint(0, action_space)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        csv_writer.writerow([episode + 1, f"{episode_reward:.2f}", step + 1])
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{n_episodes}: Avg Reward (last 100) = {avg_reward:.2f}")
    
    log_file.close()
    
    # Final statistics
    print("\n" + "="*60)
    print("✅ Random Policy Complete!")
    print(f"   Episodes: {n_episodes}")
    print(f"   Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"   Std Reward: {np.std(episode_rewards):.2f}")
    print(f"   Min Reward: {np.min(episode_rewards):.2f}")
    print(f"   Max Reward: {np.max(episode_rewards):.2f}")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Policy Baseline')
    parser.add_argument('--n-episodes', type=int, default=1000)
    parser.add_argument('--red-agent', type=str, default='B_lineAgent',
                       choices=['B_lineAgent', 'RedMeanderAgent', 'SleepAgent'])
    
    args = parser.parse_args()
    
    agent_map = {
        'B_lineAgent': B_lineAgent,
        'RedMeanderAgent': RedMeanderAgent,
        'SleepAgent': SleepAgent
    }
    
    run_random_policy(
        n_episodes=args.n_episodes,
        red_agent_type=agent_map[args.red_agent]
    )


if __name__ == "__main__":
    main()

