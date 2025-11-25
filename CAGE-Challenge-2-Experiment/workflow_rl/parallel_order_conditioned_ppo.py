"""
Parallel version of OrderConditionedPPO that works with vectorized environments
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque

from workflow_rl.order_conditioned_ppo import OrderConditionedActorCritic
from workflow_rl.parallel_env_wrapper import ParallelTrajectoryBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ParallelOrderConditionedPPO:
    """PPO agent for parallel environments with workflow conditioning"""
    
    def __init__(self, input_dims: int = 52, n_envs: int = 25,
                 lr: float = 0.002, betas: List[float] = [0.9, 0.990],
                 gamma: float = 0.99, K_epochs: int = 4, eps_clip: float = 0.2,
                 workflow_order: List[str] = None, workflow_manager = None,
                 alignment_lambda: float = 10.0,
                 update_steps: int = 100,
                 compliant_bonus_scale: float = 0.0,
                 violation_penalty_scale: float = 0.0):  # Update every 100 steps (full episode) = 2500 transitions with 25 envs
        """
        Initialize parallel PPO agent
        
        Args:
            n_envs: Number of parallel environments
            update_steps: Number of steps before PPO update
            alignment_lambda: Single parameter for alignment reward (S = λ * compliance_rate)
        """
        
        self.input_dims = input_dims
        self.n_envs = n_envs
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.alignment_lambda = alignment_lambda
        self.update_steps = update_steps
        self.compliant_bonus_scale = compliant_bonus_scale
        self.violation_penalty_scale = violation_penalty_scale
        
        # Workflow setup
        self.workflow_order = workflow_order
        self.workflow_manager = workflow_manager
        
        # Convert order to encoding
        if workflow_order and workflow_manager:
            self.order_encoding = torch.FloatTensor(
                workflow_manager.order_to_onehot(workflow_order)
            ).to(device)
        else:
            # Default order
            default_order = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
            self.order_encoding = torch.zeros(25).to(device)
            for i, unit in enumerate(default_order):
                unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
                idx = unit_types.index(unit)
                self.order_encoding[i * 5 + idx] = 1.0
        
        # Full action space
        self.action_space = list(range(145))
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
        
        # Trajectory buffer for parallel environments
        self.buffer = ParallelTrajectoryBuffer(n_envs)
        
        # Step counter for updates
        self.step_count = 0
        
        # Previous true states for each environment (for alignment reward)
        self.prev_true_states = [None] * n_envs
        
        # Compliance tracking for each environment
        self.env_compliant_actions = np.zeros(n_envs)
        self.env_total_fix_actions = np.zeros(n_envs)
        # Track which types have been fixed in each environment
        self.env_fixed_types = [set() for _ in range(n_envs)]
        # Track last compliance-based alignment score for potential shaping
        self.env_last_alignment_scores = np.zeros(n_envs)
        
    def get_actions(self, observations: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Get actions for all parallel environments
        
        Args:
            observations: (n_envs, obs_dim) array
            deterministic: If True, take argmax instead of sampling
            
        Returns:
            actions: (n_envs,) array of actions
            log_probs: (n_envs,) tensor of log probabilities
            values: (n_envs,) tensor of value estimates
        """
        states = torch.FloatTensor(observations).to(device)
        n_envs_batch = states.shape[0]  # Use actual batch size, not self.n_envs
        
        # Expand order encoding for all environments in this batch
        order_batch = self.order_encoding.unsqueeze(0).expand(n_envs_batch, -1)
        
        with torch.no_grad():
            # Concatenate state and order encoding
            augmented_states = torch.cat([states, order_batch], dim=-1)
            
            # Get action distribution and values
            action_probs = self.policy_old.actor(augmented_states)
            dist = torch.distributions.Categorical(action_probs)
            values = self.policy_old.critic(augmented_states)
            
            if deterministic:
                actions = torch.argmax(action_probs, dim=1)
            else:
                actions = dist.sample()
            
            log_probs = dist.log_prob(actions)
        
        return actions.cpu().numpy(), log_probs, values
    
    def compute_alignment_rewards(self, actions: np.ndarray, true_states: List[Dict],
                                 prev_true_states: List[Optional[Dict]], dones: np.ndarray = None) -> np.ndarray:
        """
        Compute normalized alignment rewards for all environments
        
        Two modes:
        1. Step-wise tracking: Just track compliance (no reward yet)
        2. Episode-end reward: Give normalized reward based on compliance rate
        
        Args:
            actions: (n_envs,) array of actions
            true_states: List of true state dicts for each env
            prev_true_states: List of previous true state dicts
            dones: (n_envs,) array indicating episode end
            
        Returns:
            alignment_rewards: (n_envs,) array of alignment rewards
        """
        alignment_rewards = np.zeros(self.n_envs)
        
        if not self.workflow_order or not self.workflow_manager:
            return alignment_rewards
        
        # Define Remove (15-27) and Restore (132-144) action ranges
        remove_action_range = range(15, 28)  # Remove actions
        restore_action_range = range(132, 145)  # Restore actions
        
        # Map action indices to host types
        action_to_host_type = {
            # Remove actions
            15: 'defender', 16: 'enterprise', 17: 'enterprise', 18: 'enterprise',
            19: 'op_host', 20: 'op_host', 21: 'op_host', 22: 'op_server',
            23: 'user', 24: 'user', 25: 'user', 26: 'user', 27: 'user',
            # Restore actions  
            132: 'defender', 133: 'enterprise', 134: 'enterprise', 135: 'enterprise',
            136: 'op_host', 137: 'op_host', 138: 'op_host', 139: 'op_server',
            140: 'user', 141: 'user', 142: 'user', 143: 'user', 144: 'user'
        }
        
        for env_idx in range(self.n_envs):
            action = int(actions[env_idx])
            violation = False
            step_bonus = 0.0
            
            # Only track Remove and Restore actions
            if action in remove_action_range or action in restore_action_range:
                # Get the target host type
                target_type = action_to_host_type.get(action)
                
                if target_type:
                    # Check if this fix violates the workflow order
                    target_priority = self.workflow_order.index(target_type)
                    
                    # Check if any higher priority type hasn't been fixed yet
                    for priority_idx in range(target_priority):
                        priority_type = self.workflow_order[priority_idx]
                        if priority_type not in self.env_fixed_types[env_idx]:
                            violation = True
                            break
                    
                    self.env_total_fix_actions[env_idx] += 1
                    
                    if not violation:
                        self.env_compliant_actions[env_idx] += 1
                        step_bonus = self.alignment_lambda * self.compliant_bonus_scale
                    else:
                        step_bonus = -self.alignment_lambda * self.violation_penalty_scale
                    
                    # Mark this type as fixed
                    self.env_fixed_types[env_idx].add(target_type)
            
            # Compute compliance-based score and potential shaping delta
            # Formula: S = λ * compliance_rate
            if self.env_total_fix_actions[env_idx] > 0:
                compliance_rate = self.env_compliant_actions[env_idx] / self.env_total_fix_actions[env_idx]
                current_score = self.alignment_lambda * compliance_rate
            else:
                current_score = 0.0
            
            # Per-step reward is the delta: λ * (new_rate - old_rate)
            alignment_rewards[env_idx] = current_score - self.env_last_alignment_scores[env_idx]
            self.env_last_alignment_scores[env_idx] = current_score
            alignment_rewards[env_idx] += step_bonus
            
            # If episode ends with no fixes detected, give a small penalty
            if dones is not None and dones[env_idx] and self.env_total_fix_actions[env_idx] == 0:
                alignment_rewards[env_idx] -= self.alignment_lambda * 0.2
        
        return alignment_rewards
    
    def _detect_fixes(self, true_state: Dict, prev_true_state: Dict) -> List[str]:
        """Detect which hosts were fixed"""
        fixed_hosts = []
        
        for host_name, host_info in true_state.items():
            if host_name == 'success' or host_name not in prev_true_state:
                continue
            
            prev_info = prev_true_state[host_name]
            
            # Check if host was compromised and is now clean
            was_compromised = (
                prev_info.get('System info', {}).get('Compromised', False) or
                (prev_info.get('Interface', [{}])[0].get('Compromised', False) 
                 if prev_info.get('Interface') else False)
            )
            
            is_clean = not (
                host_info.get('System info', {}).get('Compromised', False) or
                (host_info.get('Interface', [{}])[0].get('Compromised', False) 
                 if host_info.get('Interface') else False)
            )
            
            if was_compromised and is_clean:
                fixed_hosts.append(host_name)
        
        return fixed_hosts
    
    def _get_host_type(self, host_name: str) -> Optional[str]:
        """Get unit type from host name"""
        if 'defender' in host_name.lower():
            return 'defender'
        elif 'enterprise' in host_name.lower():
            return 'enterprise'
        elif 'op_server' in host_name.lower():
            return 'op_server'
        elif 'op_host' in host_name.lower():
            return 'op_host'
        elif 'user' in host_name.lower():
            return 'user'
        return None
    
    def _check_violation(self, fixed_type: str, true_state: Dict) -> bool:
        """Check if fixing this type violates workflow order"""
        if fixed_type not in self.workflow_order:
            return False
        
        fixed_priority = self.workflow_order.index(fixed_type)
        
        # Check if higher priority types are still compromised
        for priority_idx in range(fixed_priority):
            priority_type = self.workflow_order[priority_idx]
            
            # Check if any host of this type is compromised
            for host_name, host_info in true_state.items():
                if host_name == 'success':
                    continue
                
                if self._get_host_type(host_name) == priority_type:
                    is_compromised = (
                        host_info.get('System info', {}).get('Compromised', False) or
                        (host_info.get('Interface', [{}])[0].get('Compromised', False) 
                         if host_info.get('Interface') else False)
                    )
                    
                    if is_compromised:
                        return True  # Violation: higher priority still compromised
        
        return False
    
    def step_and_store(self, observations: np.ndarray, env_rewards: np.ndarray,
                      dones: np.ndarray, true_states: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a step and store in buffer
        
        Returns:
            (actions, total_rewards)
        """
        # Get actions
        actions, log_probs, values = self.get_actions(observations)
        
        # Compute alignment rewards
        alignment_rewards = self.compute_alignment_rewards(
            actions, true_states, self.prev_true_states
        )
        
        # Total rewards
        total_rewards = env_rewards + alignment_rewards
        
        # Store in buffer
        self.buffer.add(
            observations, actions, total_rewards, dones,
            log_probs.cpu().numpy(), values.cpu().numpy()
        )
        
        # Update previous states
        self.prev_true_states = true_states.copy()
        
        # Reset compliance tracking for done environments
        for env_idx in range(self.n_envs):
            if dones[env_idx]:
                self.prev_true_states[env_idx] = None
                self.env_compliant_actions[env_idx] = 0
                self.env_total_fix_actions[env_idx] = 0
        
        self.step_count += 1
        
        return actions, total_rewards
    
    def should_update(self) -> bool:
        """Check if we should run PPO update"""
        return self.step_count >= self.update_steps
    
    def update(self):
        """Run PPO update on collected trajectories"""
        if not self.should_update():
            return
        
        # Get trajectories from buffer
        trajectories = self.buffer.get_trajectories()
        
        # Compute returns
        returns = self.buffer.compute_returns(self.gamma)
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(trajectories['states']).to(device)
        actions = torch.LongTensor(trajectories['actions']).to(device)
        old_log_probs = torch.FloatTensor(trajectories['log_probs']).to(device)
        
        # Expand order encoding for batch
        batch_size = states.shape[0]
        order_batch = self.order_encoding.unsqueeze(0).expand(batch_size, -1)
        
        # PPO optimization
        for _ in range(self.K_epochs):
            # Concatenate state and order encoding
            augmented_states = torch.cat([states, order_batch], dim=-1)
            
            # Evaluate actions
            action_probs = self.policy.actor(augmented_states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            values = self.policy.critic(augmented_states).squeeze()
            entropy = dist.entropy()
            
            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Calculate advantages
            advantages = returns - values.detach()
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * self.MseLoss(values, returns)
            
            # Entropy bonus
            entropy_loss = -0.01 * entropy.mean()
            
            # Total loss
            loss = actor_loss + value_loss + entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer and reset counter
        self.buffer.reset()
        self.step_count = 0
    
    def get_compliance_rates(self) -> np.ndarray:
        """Get compliance rates for all environments"""
        n_envs = len(self.env_total_fix_actions)  # Use actual array size
        rates = np.zeros(n_envs)
        for i in range(n_envs):
            if self.env_total_fix_actions[i] > 0:
                rates[i] = self.env_compliant_actions[i] / self.env_total_fix_actions[i]
            else:
                # No fix actions detected yet - return 0.5 (neutral) instead of 1.0
                # This prevents early stopping when no fixes have occurred
                rates[i] = 0.5
        return rates
    
    def reset_episode_compliance(self, env_idx: int):
        """Reset compliance tracking for a specific environment when episode ends"""
        self.env_compliant_actions[env_idx] = 0
        self.env_total_fix_actions[env_idx] = 0
        self.env_fixed_types[env_idx] = set()
        self.env_last_alignment_scores[env_idx] = 0.0
    
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
            self.order_encoding = torch.FloatTensor(
                checkpoint['order_encoding']
            ).to(device)
