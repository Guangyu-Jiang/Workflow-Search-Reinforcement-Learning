#!/usr/bin/env python3
"""
General Workflow Search-Based PPO Framework for CAGE2
A hierarchical approach where workflows guide PPO training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import inspect
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from PPO.ActorCritic import ActorCritic
from PPO.Memory import Memory

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WorkflowSpace:
    """
    Defines the workflow search space
    General enough to work with any environment
    """
    def __init__(self, dim: int = 8, bounds: Tuple[float, float] = (-1.0, 1.0)):
        self.dim = dim
        self.bounds = bounds
        self.workflow_history = []
        self.performance_history = []
        
    def sample(self, method: str = 'uniform') -> np.ndarray:
        """Sample a workflow from the space"""
        if method == 'uniform':
            return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        elif method == 'gaussian' and len(self.workflow_history) > 0:
            # Sample around best workflow
            best_idx = np.argmax(self.performance_history)
            best_workflow = self.workflow_history[best_idx]
            noise = np.random.randn(self.dim) * 0.2
            return np.clip(best_workflow + noise, self.bounds[0], self.bounds[1])
        else:
            return self.sample('uniform')
    
    def update(self, workflow: np.ndarray, performance: float):
        """Update history with evaluation result"""
        self.workflow_history.append(workflow.copy())
        self.performance_history.append(performance)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get best workflow found so far"""
        if not self.performance_history:
            return None, float('-inf')
        best_idx = np.argmax(self.performance_history)
        return self.workflow_history[best_idx], self.performance_history[best_idx]


class WorkflowActorCritic(nn.Module):
    """
    Modified Actor-Critic that takes workflow embedding as additional input
    """
    def __init__(self, state_dim: int, action_dim: int, workflow_dim: int = 8):
        super(WorkflowActorCritic, self).__init__()
        
        # Combine state and workflow features
        combined_dim = state_dim + workflow_dim
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor, workflow: torch.Tensor):
        """Forward pass with state and workflow"""
        combined = torch.cat([state, workflow], dim=-1)
        return self.actor(combined), self.critic(combined)
    
    def act(self, state: torch.Tensor, workflow: torch.Tensor, memory=None, 
            deterministic: bool = False):
        """Select action given state and workflow"""
        combined = torch.cat([state, workflow], dim=-1)
        action_probs = self.actor(combined)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            if memory is not None:
                memory.states.append(combined)
                memory.actions.append(action)
                memory.logprobs.append(dist.log_prob(action))
        
        return action.item() if action.dim() == 0 else action
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update"""
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy


class WorkflowPPOAgent:
    """
    General PPO agent guided by workflow search
    Handles special settings like reduced action space
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_space: List[int],
                 workflow_dim: int = 8,
                 lr: float = 0.002,
                 betas: List[float] = [0.9, 0.990],
                 gamma: float = 0.99,
                 K_epochs: int = 4,
                 eps_clip: float = 0.2,
                 start_actions: List[int] = None,
                 use_decoys: bool = True):
        
        # Core parameters
        self.state_dim = state_dim
        self.action_space = action_space
        self.workflow_dim = workflow_dim
        self.n_actions = len(action_space)
        
        # PPO parameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        # Special settings
        self.start_actions = start_actions or []
        self.use_decoys = use_decoys
        
        # Initialize workflow
        self.current_workflow = None
        self.workflow_features = None
        
        # Initialize networks
        self.policy = WorkflowActorCritic(state_dim, self.n_actions, workflow_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.old_policy = WorkflowActorCritic(state_dim, self.n_actions, workflow_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Memory for PPO
        self.memory = Memory()
        
        # Loss function
        self.MSE_loss = nn.MSELoss()
        
        # Episode tracking
        self.episode_step = 0
        self.start_actions_used = []
        
        # Decoy management (if enabled)
        if use_decoys:
            self._init_decoy_system()
    
    def _init_decoy_system(self):
        """Initialize decoy management system from original implementation"""
        self.current_decoys = {
            1000: [], 1001: [], 1002: [],  # Enterprise 0-2
            1003: [], 1004: [], 1005: [], 1006: [],  # User 1-4
            1007: [],  # Defender
            1008: []   # OpServer
        }
        
        self.greedy_decoys = {
            1000: [55, 107, 120, 29],  # Enterprise0
            1001: [43],  # Enterprise1
            1002: [44],  # Enterprise2
            1003: [37, 115, 76, 102],  # User1
            1004: [51, 116, 38, 90],  # User2
            1005: [130, 91],  # User3
            1006: [131],  # User4
            1007: [54, 106, 28, 119],  # Defender
            1008: [61, 35, 113, 126]  # OpServer
        }
        
        # Extend action space with decoy IDs
        self.decoy_ids = list(range(1000, 1009))
        self.extended_action_space = self.action_space + self.decoy_ids
        self.n_actions_extended = len(self.extended_action_space)
    
    def set_workflow(self, workflow: np.ndarray):
        """Set the current workflow for this agent"""
        self.current_workflow = workflow
        self.workflow_features = self._extract_workflow_features(workflow)
        
    def _extract_workflow_features(self, workflow: np.ndarray) -> Dict:
        """Extract interpretable features from workflow vector"""
        return {
            'fortify_early': workflow[0] > 0.5,
            'fortify_intensity': workflow[1],
            'analyse_freq': workflow[2],
            'prefer_restore': workflow[3] > 0,
            'immediate_response': workflow[4] > 0.5,
            'subnet_focus': workflow[5:8] / np.sum(workflow[5:8] + 1e-8)
        }
    
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """Get action based on observation and workflow"""
        # Handle start actions
        if len(self.start_actions_used) < len(self.start_actions):
            action = self.start_actions[len(self.start_actions_used)]
            self.start_actions_used.append(action)
            return action
        
        # Prepare state
        state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
        workflow = torch.FloatTensor(self.current_workflow.reshape(1, -1)).to(device)
        
        # Get action from policy
        with torch.no_grad():
            action_idx = self.old_policy.act(state, workflow, self.memory, deterministic)
        
        # Map to actual action
        if self.use_decoys and action_idx >= len(self.action_space):
            # Handle decoy action
            decoy_host = self.extended_action_space[action_idx]
            action = self._select_decoy(decoy_host, observation)
        else:
            action = self.action_space[action_idx] if action_idx < len(self.action_space) else 0
        
        self.episode_step += 1
        return action
    
    def _select_decoy(self, host: int, observation: np.ndarray) -> int:
        """Select appropriate decoy for host (from original implementation)"""
        if host not in self.greedy_decoys:
            return self.action_space[0]  # Default action
            
        available_decoys = [d for d in self.greedy_decoys[host] 
                           if d not in self.current_decoys[host]]
        
        if available_decoys:
            decoy = available_decoys[0]
            self.current_decoys[host].append(decoy)
            return decoy
        else:
            # All decoys used, return a different action based on workflow
            if self.workflow_features['prefer_restore']:
                # Return a restore action
                restore_actions = [a for a in self.action_space if 132 <= a <= 144]
                return np.random.choice(restore_actions) if restore_actions else 0
            else:
                # Return a remove action
                remove_actions = [a for a in self.action_space if 15 <= a <= 27]
                return np.random.choice(remove_actions) if remove_actions else 0
    
    def store_transition(self, reward: float, done: bool):
        """Store transition for PPO training"""
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)
    
    def update(self):
        """PPO update step"""
        # Calculate discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), 
                                      reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert memory to tensors
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        
        # PPO epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            
            # Calculate ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Calculate advantages
            advantages = rewards - state_values.detach()
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate final loss
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = 0.5 * self.MSE_loss(state_values, rewards)
            entropy_bonus = -0.01 * dist_entropy
            
            loss = actor_loss + critic_loss + entropy_bonus
            
            # Update networks
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights to old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
    
    def clear_memory(self):
        """Clear memory after update"""
        self.memory.clear_memory()
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_step = 0
        self.start_actions_used = []
        if self.use_decoys:
            # Reset decoy tracking
            for key in self.current_decoys:
                self.current_decoys[key] = []


class WorkflowSearchPPO:
    """
    Main class that combines workflow search with PPO training
    """
    
    def __init__(self,
                 env_config: Dict,
                 ppo_config: Dict,
                 workflow_config: Dict):
        
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.workflow_config = workflow_config
        
        # Initialize workflow space
        self.workflow_space = WorkflowSpace(
            dim=workflow_config.get('dim', 8),
            bounds=workflow_config.get('bounds', (-1.0, 1.0))
        )
        
        # Track training progress
        self.training_history = defaultdict(list)
        
    def train(self, 
              num_workflows: int = 50,
              episodes_per_workflow: int = 10,
              update_frequency: int = 5):
        """
        Main training loop
        """
        print("="*80)
        print("WORKFLOW SEARCH-BASED PPO TRAINING")
        print("="*80)
        
        best_workflow = None
        best_performance = float('-inf')
        
        for workflow_idx in range(num_workflows):
            # Sample or select workflow
            if workflow_idx < 3:
                # Start with some predefined workflows
                workflow = self._get_initial_workflow(workflow_idx)
            else:
                # Use exploration strategy
                workflow = self.workflow_space.sample(
                    'gaussian' if workflow_idx > 10 else 'uniform'
                )
            
            print(f"\n--- Workflow {workflow_idx + 1}/{num_workflows} ---")
            print(f"Testing workflow: {workflow[:4]}...")
            
            # Create PPO agent with this workflow
            agent = WorkflowPPOAgent(
                state_dim=self.ppo_config['state_dim'],
                action_space=self.ppo_config['action_space'],
                workflow_dim=self.workflow_config['dim'],
                **self.ppo_config['hyperparams']
            )
            agent.set_workflow(workflow)
            
            # Train with this workflow
            workflow_performance = self._train_workflow(
                agent, 
                episodes_per_workflow, 
                update_frequency
            )
            
            # Update workflow space
            self.workflow_space.update(workflow, workflow_performance)
            
            # Track best
            if workflow_performance > best_performance:
                best_performance = workflow_performance
                best_workflow = workflow
                # Save best model
                self._save_model(agent, 'best_workflow_ppo.pth')
                print(f"New best workflow! Performance: {workflow_performance:.2f}")
            
            # Log progress
            self.training_history['workflows'].append(workflow.tolist())
            self.training_history['performances'].append(workflow_performance)
            
        # Final results
        self._print_results(best_workflow, best_performance)
        return best_workflow, best_performance
    
    def _train_workflow(self, 
                       agent: WorkflowPPOAgent,
                       num_episodes: int,
                       update_frequency: int) -> float:
        """Train PPO agent with specific workflow"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Create environment
            red_agent = B_lineAgent if episode % 2 == 0 else RedMeanderAgent
            cyborg = CybORG(PATH, 'sim', agents={'Red': red_agent})
            env = ChallengeWrapper2(env=cyborg, agent_name="Blue")
            
            # Run episode
            obs = env.reset()
            agent.reset_episode()
            episode_reward = 0
            
            for step in range(self.env_config['max_steps']):
                action = agent.get_action(obs)
                next_obs, reward, done, _ = env.step(action)
                
                agent.store_transition(reward, done)
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # PPO update
            if (episode + 1) % update_frequency == 0:
                agent.update()
                agent.clear_memory()
            
            if episode % 3 == 0:
                print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        return np.mean(episode_rewards)
    
    def _get_initial_workflow(self, idx: int) -> np.ndarray:
        """Get predefined initial workflows"""
        initial_workflows = [
            np.array([0.2, 0.1, 0.2, -0.5, 0.8, 0.6, 0.3, 0.1]),  # Reactive
            np.array([0.9, 0.4, 0.5, 0.5, 0.9, 0.3, 0.4, 0.3]),   # Proactive
            np.array([0.5, 0.3, 0.4, 0.0, 0.7, 0.4, 0.4, 0.2])    # Balanced
        ]
        return initial_workflows[idx % 3]
    
    def _save_model(self, agent: WorkflowPPOAgent, filename: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'workflow': agent.current_workflow
        }, filename)
    
    def _print_results(self, best_workflow: np.ndarray, best_performance: float):
        """Print final results"""
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"\nBest workflow: {best_workflow}")
        print(f"Best performance: {best_performance:.2f}")
        
        # Save history
        with open('workflow_training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print("\nTraining history saved to workflow_training_history.json")


if __name__ == "__main__":
    # Configuration
    env_config = {
        'max_steps': 50
    }
    
    ppo_config = {
        'state_dim': 52,
        'action_space': [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22,
                        11, 12, 13, 14, 141, 142, 143, 144, 132, 2, 15, 24, 25, 26, 27],
        'hyperparams': {
            'lr': 0.002,
            'betas': [0.9, 0.990],
            'gamma': 0.99,
            'K_epochs': 4,
            'eps_clip': 0.2,
            'start_actions': [],  # Can add initial decoy placements
            'use_decoys': False   # Simplified for demo
        }
    }
    
    workflow_config = {
        'dim': 8,
        'bounds': (-1.0, 1.0)
    }
    
    # Create and run trainer
    trainer = WorkflowSearchPPO(env_config, ppo_config, workflow_config)
    
    # Train (reduced for demo)
    best_workflow, best_performance = trainer.train(
        num_workflows=5,
        episodes_per_workflow=3,
        update_frequency=2
    )
