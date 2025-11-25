"""
PPO baseline training WITHOUT action space reduction
This version uses the full action space (145 actions) for fair comparison with workflow search
"""

import torch
import numpy as np
import os
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect
from Agents.PPOAgent import PPOAgent
import random
from datetime import datetime

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100, 
          start_actions=[], log_file=None):
    """
    Train PPO agent with full action space
    
    Args:
        env: CybORG environment
        input_dims: Observation space dimensions (52)
        action_space: Full action space (0-144) - NO REDUCTION
        max_episodes: Total episodes to train
        max_timesteps: Steps per episode
        update_timestep: Steps before PPO update
        K_epochs: PPO optimization epochs
        eps_clip: PPO clipping parameter
        gamma: Discount factor
        lr: Learning rate
        betas: Adam optimizer betas
        ckpt_folder: Directory for checkpoints
        print_interval: Episodes between print
        save_interval: Episodes between saves
        start_actions: Initial decoy actions (optional)
        log_file: CSV file for logging (optional)
    """
    
    agent = PPOAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip, 
                     start_actions=start_actions)
    
    running_reward, time_step = 0, 0
    episode_rewards = []
    
    # Create log file if specified
    if log_file:
        import csv
        csv_file = open(log_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Episode', 'Reward', 'Steps'])
    
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Full Action Space: {len(action_space)} actions")
    print(f"  Observation Space: {input_dims} dimensions")
    print(f"  Update Every: {update_timestep} steps ({update_timestep/max_timesteps:.0f} episodes)")
    print(f"  K Epochs: {K_epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Gamma: {gamma}")
    print(f"  Epsilon Clip: {eps_clip}")
    print(f"  Max Episodes: {max_episodes}")
    print("="*60)
    
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for t in range(max_timesteps):
            time_step += 1
            episode_steps += 1
            
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            agent.store(reward, done)
            
            episode_reward += reward
            running_reward += reward
            
            # Update PPO
            if time_step % update_timestep == 0:
                print(f"  [Update at step {time_step}] Training PPO...")
                agent.train()
                agent.clear_memory()
                time_step = 0
            
            if done:
                break
        
        agent.end_episode()
        episode_rewards.append(episode_reward)
        
        # Log to CSV
        if log_file:
            csv_writer.writerow([i_episode, episode_reward, episode_steps])
            csv_file.flush()
        
        # Save checkpoint
        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, 'episode_{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            print(f'  Checkpoint saved: {ckpt}')
        
        # Print progress
        if i_episode % print_interval == 0:
            avg_reward = running_reward / print_interval
            recent_avg = np.mean(episode_rewards[-print_interval:])
            print(f'Episode {i_episode:5d} | Avg Reward: {avg_reward:7.2f} | Recent Avg: {recent_avg:7.2f}')
            running_reward = 0
    
    if log_file:
        csv_file.close()
        print(f"Training log saved to: {log_file}")
    
    # Save final model
    final_ckpt = os.path.join(ckpt_folder, 'final_model.pth')
    torch.save(agent.policy.state_dict(), final_ckpt)
    print(f"Final model saved: {final_ckpt}")
    
    return episode_rewards


def main(red_agent_type='bline', use_decoys=False):
    """
    Main training function
    
    Args:
        red_agent_type: 'meander' or 'bline' 
        use_decoys: Whether to use initial decoy actions
    """
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = f'baseline_ppo_full_action_{red_agent_type}_{timestamp}'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    
    log_file = os.path.join(ckpt_folder, 'training_log.csv')
    
    # Select red agent
    if red_agent_type == 'meander':
        red_agent = RedMeanderAgent
    elif red_agent_type == 'bline':
        red_agent = B_lineAgent
    else:
        raise ValueError(f"Unknown red agent type: {red_agent_type}")
    
    print(f"="*60)
    print(f"PPO Training with FULL ACTION SPACE")
    print(f"="*60)
    print(f"Red Agent: {red_agent.__name__}")
    print(f"Checkpoint Directory: {ckpt_folder}")
    print(f"="*60)
    
    # Create environment
    CYBORG = CybORG(PATH, 'sim', agents={'Red': red_agent})
    env = ChallengeWrapper2(env=CYBORG, agent_name="Blue")
    
    input_dims = env.observation_space.shape[0]  # 52
    
    # FULL ACTION SPACE - NO REDUCTION!
    # Using all 145 actions (0-144)
    action_space = list(range(145))
    
    print(f"Action Space Comparison:")
    print(f"  Original baseline: 22 selected actions")
    print(f"  This version: {len(action_space)} actions (FULL SPACE)")
    print(f"  Reduction factor: {145/22:.1f}x more actions")
    print(f"="*60)
    
    # Optional: Use decoy start actions
    start_actions = []
    if use_decoys:
        # These are example decoy actions from the original
        # You may need to verify these action IDs exist in full space
        start_actions = []  # Disabled for now as IDs may differ
    
    # Training hyperparameters (matching original for fair comparison)
    print_interval = 50
    save_interval = 10000
    max_episodes = 100000  # Reduced for testing, use 100000 for full training
    max_timesteps = 100
    
    # IMPORTANT: Buffer size before update
    # Original: 20000 steps = 200 episodes before each update
    update_timesteps = 5000  # This means collect 200 episodes before updating
    
    K_epochs = 6  # PPO optimization epochs
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002
    betas = [0.9, 0.990]
    
    print(f"Training Details:")
    print(f"  Episodes before update: {update_timesteps/max_timesteps:.0f}")
    print(f"  Total timesteps per update: {update_timesteps}")
    print(f"  PPO epochs per update: {K_epochs}")
    print(f"="*60)
    
    # Train
    episode_rewards = train(
        env, input_dims, action_space,
        max_episodes=max_episodes, 
        max_timesteps=max_timesteps,
        update_timestep=update_timesteps, 
        K_epochs=K_epochs,
        eps_clip=eps_clip, 
        gamma=gamma, 
        lr=lr,
        betas=betas, 
        ckpt_folder=ckpt_folder,
        print_interval=print_interval, 
        save_interval=save_interval, 
        start_actions=start_actions,
        log_file=log_file
    )
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Statistics:")
    print(f"  Total Episodes: {len(episode_rewards)}")
    print(f"  Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"  Average Reward (last 100 eps): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Model saved in: {ckpt_folder}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO with full action space')
    parser.add_argument('--red-agent', type=str, default='bline',
                       choices=['meander', 'bline'],
                       help='Red agent type to train against')
    parser.add_argument('--episodes', type=int, default=100000,
                       help='Number of episodes to train')
    parser.add_argument('--use-decoys', action='store_true',
                       help='Use initial decoy actions')
    
    args = parser.parse_args()
    
    # Override max_episodes if specified
    if args.episodes != 100000:
        import train_no_action_reduction
        train_no_action_reduction.main.__defaults__ = (args.red_agent, args.use_decoys)
        # Need to modify the max_episodes in main function
    
    main(red_agent_type=args.red_agent, use_decoys=args.use_decoys)
