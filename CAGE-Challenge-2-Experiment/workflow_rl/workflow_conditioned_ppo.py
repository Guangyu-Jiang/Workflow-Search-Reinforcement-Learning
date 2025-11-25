#!/usr/bin/env python3
"""
Workflow-Conditioned PPO for CAGE2
Implements PPO with alignment rewards for following defense priority workflows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional
import copy

from CybORG.Agents import BaseAgent
from PPO.Memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WorkflowConditionedActorCritic(nn.Module):
    """Actor-Critic network conditioned on workflow"""
    
    def __init__(self, state_dim: int = 52, action_dim: int = 145, workflow_dim: int = 8):
        super().__init__()
        
        # Shared encoder for state + workflow
        input_dim = state_dim + workflow_dim
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def act(self, state: torch.Tensor, workflow: torch.Tensor, memory: Memory, 
            deterministic: bool = False):
        """Select action given state and workflow"""
        
        # Concatenate state and workflow
        if workflow.dim() == 1:
            workflow = workflow.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        augmented_state = torch.cat([state, workflow], dim=-1)
        
        # Get action probabilities
        action_probs = self.actor(augmented_state)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            action = dist.sample()
        
        # Store in memory for training
        if memory is not None and not deterministic:
            memory.states.append(augmented_state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, states: torch.Tensor, workflows: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update"""
        
        # Ensure workflows match batch size
        if workflows.dim() == 1:
            workflows = workflows.unsqueeze(0).repeat(states.shape[0], 1)
        
        # Concatenate states and workflows
        augmented_states = torch.cat([states, workflows], dim=-1)
        
        # Get action probabilities and values
        action_probs = self.actor(augmented_states)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        state_values = self.critic(augmented_states)
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy


class WorkflowConditionedPPO(BaseAgent):
    """PPO agent that follows workflow priorities with alignment rewards"""
    
    def __init__(self, input_dims: int = 52, action_space: List[int] = None,
                 lr: float = 0.002, betas: List[float] = [0.9, 0.990], 
                 gamma: float = 0.99, K_epochs: int = 4, eps_clip: float = 0.2,
                 workflow: np.ndarray = None, workflow_dim: int = 8,
                 alignment_alpha: float = 0.1, alignment_beta: float = 0.2,
                 deterministic: bool = False, training: bool = True):
        
        self.input_dims = input_dims
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.deterministic = deterministic
        self.training = training
        
        # Workflow-specific parameters
        self.workflow = torch.FloatTensor(workflow).to(device) if workflow is not None else None
        self.workflow_dim = workflow_dim
        self.alignment_alpha = alignment_alpha  # Bonus for correct fixes
        self.alignment_beta = alignment_beta    # Penalty for violations
        
        # Use full action space (no reduction)
        if action_space is None:
            self.action_space = list(range(145))  # Full CAGE2 action space
        else:
            self.action_space = action_space
        
        self.n_actions = len(self.action_space)
        
        # Initialize networks
        self.policy = WorkflowConditionedActorCritic(
            input_dims, self.n_actions, workflow_dim
        ).to(device)
        
        self.policy_old = WorkflowConditionedActorCritic(
            input_dims, self.n_actions, workflow_dim
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss()
        
        # Memory
        self.memory = Memory()
        
        # For tracking alignment
        self.last_action = None
        self.last_true_state = None
    
    def set_workflow(self, workflow: np.ndarray):
        """Set the workflow for this agent"""
        self.workflow = torch.FloatTensor(workflow).to(device)
    
    def get_action(self, observation: np.ndarray, action_space: List[int] = None) -> int:
        """Get action following workflow priority"""
        
        if self.workflow is None:
            raise ValueError("Workflow not set for agent")
        
        state = torch.FloatTensor(observation).to(device)
        
        # Get action from policy
        action_idx = self.policy_old.act(
            state, self.workflow, self.memory, 
            deterministic=self.deterministic
        )
        
        # Map to actual action
        action = self.action_space[action_idx]
        self.last_action = action
        
        return action
    
    def compute_alignment_reward(self, action: int, true_state: Dict, 
                                workflow_order: List[str]) -> float:
        """
        Compute alignment reward based on whether action follows workflow priority
        
        Args:
            action: The action taken
            true_state: True environment state showing compromised hosts
            workflow_order: Priority order of unit types
        
        Returns:
            Alignment reward (positive for compliance, negative for violation)
        """
        
        # Map action to target unit type
        target_type = self._get_action_unit_type(action)
        if target_type is None:
            return 0.0  # Non-fix actions get no alignment reward
        
        # Get compromised units by type
        compromised_by_type = self._get_compromised_by_type(true_state)
        
        # Check if there are higher priority compromised units
        target_priority = workflow_order.index(target_type) if target_type in workflow_order else 999
        
        violation = False
        for unit_type, units in compromised_by_type.items():
            if units and unit_type in workflow_order:
                type_priority = workflow_order.index(unit_type)
                if type_priority < target_priority:
                    violation = True
                    break
        
        # Return alignment reward
        if violation:
            return -self.alignment_beta  # Penalty for violation
        else:
            return self.alignment_alpha   # Bonus for compliance
    
    def _get_action_unit_type(self, action: int) -> Optional[str]:
        """Map action ID to unit type"""
        
        # Only consider fix actions (Analyse, Remove, Restore)
        if action in [2, 15, 132]:  # Defender
            return 'defender'
        elif action in [3, 4, 5, 16, 17, 18, 133, 134, 135]:  # Enterprise
            return 'enterprise'
        elif action in [9, 22, 139]:  # Op_Server
            return 'op_server'
        elif action in [6, 7, 8, 19, 20, 21, 136, 137, 138]:  # Op_Host
            return 'op_host'
        elif action in [10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 140, 141, 142, 143, 144]:  # User
            return 'user'
        else:
            return None  # Not a fix action
    
    def _get_compromised_by_type(self, true_state: Dict) -> Dict[str, List[str]]:
        """Extract compromised hosts grouped by type from true state"""
        
        compromised_by_type = {
            'defender': [],
            'enterprise': [],
            'op_server': [],
            'op_host': [],
            'user': []
        }
        
        for hostname, host_info in true_state.items():
            if hostname == 'success':
                continue
                
            # Check if host is compromised (has Red sessions)
            if 'Sessions' in host_info:
                for session in host_info['Sessions']:
                    if session.get('Agent') == 'Red':
                        # Categorize by type
                        if hostname == 'Defender':
                            compromised_by_type['defender'].append(hostname)
                        elif 'Enterprise' in hostname:
                            compromised_by_type['enterprise'].append(hostname)
                        elif hostname == 'Op_Server0':
                            compromised_by_type['op_server'].append(hostname)
                        elif 'Op_Host' in hostname:
                            compromised_by_type['op_host'].append(hostname)
                        elif 'User' in hostname:
                            compromised_by_type['user'].append(hostname)
                        break
        
        return compromised_by_type
    
    def update(self, reward: float, done: bool):
        """Store reward and update if done"""
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)
        
        if done and self.training:
            self.train()
            self.memory.clear_memory()
    
    def train(self):
        """PPO training update"""
        
        # Monte Carlo estimate of rewards
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
        
        # Convert lists to tensors
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        
        # Extract states and workflows from augmented states
        state_dim = self.input_dims
        old_state_only = old_states[:, :state_dim]
        old_workflows = old_states[:, state_dim:]
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_state_only, old_workflows[0], old_actions
            )
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Loss
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'workflow': self.workflow.cpu().numpy() if self.workflow is not None else None
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['workflow'] is not None:
            self.workflow = torch.FloatTensor(checkpoint['workflow']).to(device)
    
    def end_episode(self):
        """Reset episode-specific variables"""
        self.last_action = None
        self.last_true_state = None
    
    def set_initial_values(self, action_space, observation):
        """Required by CybORG interface"""
        pass
