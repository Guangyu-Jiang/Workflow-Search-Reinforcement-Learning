#!/usr/bin/env python3
"""
Order-Conditioned PPO Agent for CAGE2
PPO agent that follows workflow priorities defined as orders
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    """PPO Memory Buffer"""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class OrderConditionedActorCritic(nn.Module):
    """
    Actor-Critic network conditioned on workflow order
    """
    
    def __init__(self, input_dims: int = 52, n_actions: int = 145, order_dims: int = 25):
        """
        Args:
            input_dims: State observation dimensions
            n_actions: Number of actions (full action space)
            order_dims: Order representation dimensions (25 for one-hot encoding of 5x5)
        """
        super(OrderConditionedActorCritic, self).__init__()
        
        # Total input is state + order representation
        augmented_input = input_dims + order_dims
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(augmented_input, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(augmented_input, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def act(self, state: torch.Tensor, order_encoding: torch.Tensor, 
            memory: Optional[Memory] = None, deterministic: bool = False):
        """Select action given state and workflow order"""
        
        # Ensure correct dimensions
        if order_encoding.dim() == 1:
            order_encoding = order_encoding.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Concatenate state and order
        augmented_state = torch.cat([state, order_encoding], dim=-1)
        
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
    
    def evaluate(self, states: torch.Tensor, order_encodings: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update"""
        
        # Ensure order encodings match batch size
        if order_encodings.dim() == 1:
            order_encodings = order_encodings.unsqueeze(0).repeat(states.shape[0], 1)
        
        # Concatenate states and orders
        augmented_states = torch.cat([states, order_encodings], dim=-1)
        
        # Get action probabilities and values
        action_probs = self.actor(augmented_states)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        state_values = self.critic(augmented_states)
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy


class OrderConditionedPPO:
    """PPO agent that follows workflow order priorities"""
    
    def __init__(self, input_dims: int = 52, action_space: List[int] = None,
                 lr: float = 0.002, betas: List[float] = [0.9, 0.990], 
                 gamma: float = 0.99, K_epochs: int = 4, eps_clip: float = 0.2,
                 workflow_order: List[str] = None, workflow_manager = None,
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
        self.workflow_order = workflow_order
        self.workflow_manager = workflow_manager
        self.alignment_alpha = alignment_alpha  # Bonus for correct fixes
        self.alignment_beta = alignment_beta    # Penalty for violations
        
        # Convert order to encoding
        if workflow_order and workflow_manager:
            self.order_encoding = torch.FloatTensor(
                workflow_manager.order_to_onehot(workflow_order)
            ).to(device)
        else:
            # Default order
            default_order = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
            self.order_encoding = torch.zeros(25).to(device)
            # Set default one-hot encoding
            for i, unit in enumerate(default_order):
                unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
                idx = unit_types.index(unit)
                self.order_encoding[i * 5 + idx] = 1.0
        
        # Use full action space (no reduction)
        if action_space is None:
            self.action_space = list(range(145))  # Full CAGE2 action space
        else:
            self.action_space = action_space
        
        self.n_actions = len(self.action_space)
        
        # Initialize networks
        self.policy = OrderConditionedActorCritic(
            input_dims, self.n_actions, order_dims=25
        ).to(device)
        
        self.policy_old = OrderConditionedActorCritic(
            input_dims, self.n_actions, order_dims=25
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss()
        
        # Memory
        self.memory = Memory()
        
        # For tracking alignment
        self.recent_fixes = deque(maxlen=10)
    
    def get_action(self, observation):
        """Get action from policy"""
        state = torch.FloatTensor(observation).to(device)
        
        if self.training:
            action = self.policy_old.act(
                state, self.order_encoding, self.memory, self.deterministic
            )
        else:
            action = self.policy.act(
                state, self.order_encoding, None, self.deterministic
            )
        
        return action
    
    def compute_alignment_reward(self, action: int, true_state: Dict, 
                                 prev_true_state: Optional[Dict] = None) -> float:
        """
        Compute alignment reward based on workflow order compliance
        
        Returns positive reward for fixes following the order,
        negative for violations
        """
        
        if not self.workflow_order or not self.workflow_manager:
            return 0.0
        
        if prev_true_state is None:
            return 0.0
        
        # Detect if a fix occurred by comparing states
        fixed_hosts = []
        
        for host_name, host_info in true_state.items():
            if host_name == 'success':
                continue
                
            if host_name in prev_true_state:
                prev_info = prev_true_state[host_name]
                
                # Check if host was compromised and is now clean
                was_compromised = (
                    prev_info.get('System info', {}).get('Compromised', False) or
                    prev_info.get('Interface', [{}])[0].get('Compromised', False) if prev_info.get('Interface') else False
                )
                
                is_clean = not (
                    host_info.get('System info', {}).get('Compromised', False) or
                    host_info.get('Interface', [{}])[0].get('Compromised', False) if host_info.get('Interface') else False
                )
                
                if was_compromised and is_clean:
                    fixed_hosts.append(host_name)
        
        if not fixed_hosts:
            return 0.0
        
        # Check if fixes follow the workflow order
        alignment_reward = 0.0
        
        for fixed_host in fixed_hosts:
            if fixed_host in self.workflow_manager.host_to_type:
                fixed_type = self.workflow_manager.host_to_type[fixed_host]
                fixed_priority = self.workflow_order.index(fixed_type)
                
                # Check against other compromised hosts
                violation = False
                for other_host, other_info in true_state.items():
                    if other_host == 'success' or other_host == fixed_host:
                        continue
                    
                    # Check if other host is still compromised
                    is_compromised = (
                        other_info.get('System info', {}).get('Compromised', False) or
                        other_info.get('Interface', [{}])[0].get('Compromised', False) if other_info.get('Interface') else False
                    )
                    
                    if is_compromised and other_host in self.workflow_manager.host_to_type:
                        other_type = self.workflow_manager.host_to_type[other_host]
                        other_priority = self.workflow_order.index(other_type)
                        
                        # Violation if we fixed lower priority while higher exists
                        if other_priority < fixed_priority:
                            violation = True
                            break
                
                if violation:
                    alignment_reward -= self.alignment_beta
                else:
                    alignment_reward += self.alignment_alpha
                
                # Track recent fixes
                self.recent_fixes.append((fixed_host, fixed_type, violation))
        
        return alignment_reward
    
    def update(self):
        """PPO update"""
        if not self.training:
            return
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), 
                                      reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert lists to tensors
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.tensor(self.memory.actions, dtype=torch.long).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states[:, :self.input_dims],  # Extract original state
                self.order_encoding,
                old_actions
            )
            
            # Find ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Find Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory.clear_memory()
    
    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'workflow_order': self.workflow_order,
            'order_encoding': self.order_encoding.cpu().numpy()
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'workflow_order' in checkpoint:
            self.workflow_order = checkpoint['workflow_order']
        if 'order_encoding' in checkpoint:
            self.order_encoding = torch.FloatTensor(checkpoint['order_encoding']).to(device)
