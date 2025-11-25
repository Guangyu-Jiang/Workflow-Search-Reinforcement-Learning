#!/usr/bin/env python3
"""
Learning Meaningful Workflow Representations from Scratch
No behavioral cloning - discover workflow semantics through exploration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# APPROACH 1: EMERGENT SEMANTICS THROUGH CONSISTENT CONDITIONING
# ============================================================================

class EmergentWorkflowSemantics:
    """
    Let the network discover what workflows mean through consistent exposure
    Key: Same workflow vector across many episodes → network learns its meaning
    """
    
    def __init__(self, workflow_dim: int = 8):
        self.workflow_dim = workflow_dim
        
    def create_diverse_workflows(self, n: int = 20) -> List[np.ndarray]:
        """
        Create maximally diverse workflows to encourage distinct behaviors
        Use orthogonal or well-separated vectors
        """
        workflows = []
        
        # Method 1: Orthogonal basis vectors (for first 8)
        if n >= self.workflow_dim:
            for i in range(self.workflow_dim):
                w = np.zeros(self.workflow_dim)
                w[i] = 1.0
                workflows.append(w)
            
            # Negative basis vectors
            for i in range(min(self.workflow_dim, n - self.workflow_dim)):
                w = np.zeros(self.workflow_dim)
                w[i] = -1.0
                workflows.append(w)
        
        # Method 2: Random with minimum separation
        while len(workflows) < n:
            candidate = np.random.uniform(-1, 1, self.workflow_dim)
            
            # Ensure minimum distance from existing workflows
            if workflows:
                min_dist = min(np.linalg.norm(candidate - w) for w in workflows)
                if min_dist < 0.5:  # Too close to existing
                    continue
            
            workflows.append(candidate)
        
        return workflows[:n]
    
    def train_with_consistency(self, policy, workflow, num_episodes: int = 100):
        """
        Train policy with SAME workflow for many episodes
        This consistency allows the network to learn what this workflow means
        """
        # Key insight: By keeping workflow fixed for many episodes,
        # the network learns to associate this particular vector with
        # the behaviors that work well when conditioned on it
        
        for episode in range(num_episodes):
            # Train with this workflow
            # The network will gradually learn that this workflow vector
            # should produce certain behaviors
            pass
        
        return policy


# ============================================================================
# APPROACH 2: DISCOVERY THROUGH REGULARIZATION
# ============================================================================

class RegularizedWorkflowLearning(nn.Module):
    """
    Use regularization to ensure different workflows produce different behaviors
    No pre-training needed - behaviors emerge from the regularization
    """
    
    def __init__(self, state_dim: int = 52, workflow_dim: int = 8, action_dim: int = 27):
        super().__init__()
        self.state_dim = state_dim
        self.workflow_dim = workflow_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim + workflow_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Diversity regularizer: ensures different workflows → different policies
        self.workflow_memory = []
        self.policy_memory = []
        
    def forward(self, state: torch.Tensor, workflow: torch.Tensor):
        combined = torch.cat([state, workflow], dim=-1)
        return self.policy(combined)
    
    def compute_diversity_loss(self, workflow: torch.Tensor, 
                              policy_output: torch.Tensor) -> torch.Tensor:
        """
        Regularization loss that encourages different workflows to produce different behaviors
        This creates meaningful distinctions without needing pre-training
        """
        
        if len(self.workflow_memory) < 2:
            return torch.tensor(0.0).to(device)
        
        diversity_loss = 0.0
        
        # Compare current workflow/policy pair with memory
        for past_workflow, past_policy in zip(self.workflow_memory[-10:], 
                                              self.policy_memory[-10:]):
            # Workflow similarity
            workflow_sim = F.cosine_similarity(workflow, past_workflow, dim=-1)
            
            # Policy similarity (KL divergence)
            policy_sim = -F.kl_div(
                torch.log(policy_output + 1e-8), 
                past_policy, 
                reduction='batchmean'
            )
            
            # Loss: similar workflows should have similar policies
            # Different workflows should have different policies
            diversity_loss += (workflow_sim - policy_sim).pow(2)
        
        return diversity_loss / len(self.workflow_memory[-10:])
    
    def update_memory(self, workflow: torch.Tensor, policy_output: torch.Tensor):
        """Store recent workflow-policy pairs"""
        self.workflow_memory.append(workflow.detach())
        self.policy_memory.append(policy_output.detach())
        
        # Keep memory bounded
        if len(self.workflow_memory) > 100:
            self.workflow_memory.pop(0)
            self.policy_memory.pop(0)


# ============================================================================
# APPROACH 3: IMPLICIT SEMANTICS THROUGH ACTION MASKING
# ============================================================================

class ImplicitWorkflowSemantics:
    """
    Create meaningful workflows by having them implicitly affect action availability
    No pre-training needed - semantics emerge from constraints
    """
    
    def __init__(self, workflow_dim: int = 8, action_dim: int = 27):
        self.workflow_dim = workflow_dim
        self.action_dim = action_dim
        
        # Learn a mapping from workflow to action preferences
        self.preference_network = nn.Sequential(
            nn.Linear(workflow_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()  # Output [0,1] preference for each action
        ).to(device)
        
    def get_action_weights(self, workflow: torch.Tensor) -> torch.Tensor:
        """
        Convert workflow to action preferences (soft masking)
        This creates implicit semantics without pre-training
        """
        preferences = self.preference_network(workflow)
        
        # Add noise during training to encourage exploration
        if self.training:
            noise = torch.randn_like(preferences) * 0.1
            preferences = torch.clamp(preferences + noise, 0, 1)
        
        return preferences
    
    def modulate_policy(self, policy_output: torch.Tensor, 
                       workflow: torch.Tensor) -> torch.Tensor:
        """
        Modulate policy output based on workflow preferences
        This creates meaningful workflow effects without pre-training
        """
        action_weights = self.get_action_weights(workflow)
        
        # Soft masking: multiply probabilities by preferences
        modulated = policy_output * action_weights
        
        # Renormalize
        modulated = modulated / (modulated.sum(dim=-1, keepdim=True) + 1e-8)
        
        return modulated


# ============================================================================
# APPROACH 4: SELF-SUPERVISED WORKFLOW DISCOVERY
# ============================================================================

class SelfSupervisedWorkflowDiscovery:
    """
    Discover workflow semantics through self-supervised learning
    No demonstrations needed - learn from environment interactions
    """
    
    def __init__(self, workflow_dim: int = 8):
        self.workflow_dim = workflow_dim
        self.discovered_patterns = defaultdict(list)
        
    def discover_pattern(self, workflow: np.ndarray, 
                        episode_trajectory: List[Tuple]) -> Dict:
        """
        Analyze episode trajectory to discover what this workflow does
        """
        patterns = {
            'early_actions': [],
            'late_actions': [],
            'common_sequences': [],
            'state_visitation': [],
            'average_reward': 0
        }
        
        # Analyze early vs late behavior
        if len(episode_trajectory) > 20:
            early = episode_trajectory[:10]
            late = episode_trajectory[-10:]
            
            patterns['early_actions'] = [a for _, a, _, _ in early]
            patterns['late_actions'] = [a for _, a, _, _ in late]
        
        # Find common action sequences
        actions = [a for _, a, _, _ in episode_trajectory]
        patterns['common_sequences'] = self._find_common_sequences(actions)
        
        # Track state visitation patterns
        states = [s for s, _, _, _ in episode_trajectory]
        patterns['state_visitation'] = self._cluster_states(states)
        
        # Average reward
        rewards = [r for _, _, r, _ in episode_trajectory]
        patterns['average_reward'] = np.mean(rewards) if rewards else 0
        
        # Store discovered pattern
        workflow_key = tuple(workflow.round(2))  # Discretize for grouping
        self.discovered_patterns[workflow_key].append(patterns)
        
        return patterns
    
    def _find_common_sequences(self, actions: List[int], min_length: int = 2) -> List:
        """Find frequently occurring action sequences"""
        sequences = defaultdict(int)
        
        for i in range(len(actions) - min_length + 1):
            seq = tuple(actions[i:i+min_length])
            sequences[seq] += 1
        
        # Return most common sequences
        common = sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:5]
        return [list(seq) for seq, count in common if count > 1]
    
    def _cluster_states(self, states: List[np.ndarray]) -> Dict:
        """Simple clustering of visited states"""
        if not states:
            return {}
        
        # Simple: just track mean and std of state features
        states_array = np.array(states)
        return {
            'mean': states_array.mean(axis=0),
            'std': states_array.std(axis=0)
        }
    
    def create_semantic_embedding(self, workflow: np.ndarray) -> np.ndarray:
        """
        Create semantic embedding based on discovered patterns
        This gives meaning to arbitrary workflow vectors
        """
        workflow_key = tuple(workflow.round(2))
        
        if workflow_key not in self.discovered_patterns:
            return np.zeros(16)  # Default embedding
        
        patterns_list = self.discovered_patterns[workflow_key]
        
        # Aggregate discovered patterns into semantic features
        semantic_features = []
        
        # Feature 1-4: Action type distribution in early phase
        early_action_types = [0, 0, 0, 0]  # fortify, analyze, remove, restore
        for patterns in patterns_list:
            for action in patterns['early_actions']:
                if 1000 <= action <= 1008:
                    early_action_types[0] += 1
                elif action in [2, 3, 4, 5, 9, 11, 12, 13, 14]:
                    early_action_types[1] += 1
                elif action in [15, 16, 17, 18, 22, 24, 25, 26, 27]:
                    early_action_types[2] += 1
                elif action in [132, 133, 134, 135, 139, 141, 142, 143, 144]:
                    early_action_types[3] += 1
        
        total = sum(early_action_types) + 1e-8
        semantic_features.extend([x/total for x in early_action_types])
        
        # Feature 5-8: Action type distribution in late phase
        late_action_types = [0, 0, 0, 0]
        for patterns in patterns_list:
            for action in patterns['late_actions']:
                if 1000 <= action <= 1008:
                    late_action_types[0] += 1
                elif action in [2, 3, 4, 5, 9, 11, 12, 13, 14]:
                    late_action_types[1] += 1
                elif action in [15, 16, 17, 18, 22, 24, 25, 26, 27]:
                    late_action_types[2] += 1
                elif action in [132, 133, 134, 135, 139, 141, 142, 143, 144]:
                    late_action_types[3] += 1
        
        total = sum(late_action_types) + 1e-8
        semantic_features.extend([x/total for x in late_action_types])
        
        # Feature 9-12: Common sequence indicators
        # (simplified - just count presence of certain patterns)
        semantic_features.extend([0, 0, 0, 0])  # Placeholder
        
        # Feature 13-16: Performance statistics
        avg_rewards = [p['average_reward'] for p in patterns_list]
        if avg_rewards:
            semantic_features.extend([
                np.mean(avg_rewards),
                np.std(avg_rewards),
                np.min(avg_rewards),
                np.max(avg_rewards)
            ])
        else:
            semantic_features.extend([0, 0, 0, 0])
        
        return np.array(semantic_features)


# ============================================================================
# APPROACH 5: PROGRESSIVE DIFFERENTIATION
# ============================================================================

class ProgressiveDifferentiation:
    """
    Start with random workflows, progressively differentiate them based on performance
    No pre-training - workflows gain meaning through evolution
    """
    
    def __init__(self, workflow_dim: int = 8, population_size: int = 20):
        self.workflow_dim = workflow_dim
        self.population_size = population_size
        
        # Initialize random population
        self.population = [
            np.random.uniform(-1, 1, workflow_dim) 
            for _ in range(population_size)
        ]
        self.performance = [0.0] * population_size
        self.generation = 0
        
    def evolve_workflows(self, performance_scores: List[float]):
        """
        Evolve workflows to be more distinct based on performance differences
        """
        self.performance = performance_scores
        self.generation += 1
        
        # Sort by performance
        sorted_indices = np.argsort(performance_scores)[::-1]
        
        # Keep top performers
        elite_size = self.population_size // 4
        new_population = [self.population[i] for i in sorted_indices[:elite_size]]
        
        # Create variants of good performers
        while len(new_population) < self.population_size:
            # Select parent
            parent_idx = np.random.choice(elite_size)
            parent = new_population[parent_idx]
            
            # Create child with mutation
            child = parent + np.random.randn(self.workflow_dim) * 0.1
            
            # Push child away from similar workflows (differentiation)
            for other in new_population:
                if np.linalg.norm(child - other) < 0.3:
                    # Too similar - push away
                    direction = child - other
                    child += direction * 0.1
            
            # Clip to bounds
            child = np.clip(child, -1, 1)
            new_population.append(child)
        
        self.population = new_population
        
    def get_diverse_workflows(self, n: int) -> List[np.ndarray]:
        """
        Return n maximally diverse workflows from current population
        """
        if n >= len(self.population):
            return self.population.copy()
        
        # Greedy selection for diversity
        selected = []
        remaining = self.population.copy()
        
        # Start with random workflow
        idx = np.random.choice(len(remaining))
        selected.append(remaining.pop(idx))
        
        # Iteratively select most distant workflow
        while len(selected) < n and remaining:
            max_min_dist = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                min_dist = min(np.linalg.norm(candidate - s) for s in selected)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected


# ============================================================================
# MAIN TRAINING FRAMEWORK WITHOUT BEHAVIORAL CLONING
# ============================================================================

class WorkflowTrainingFromScratch:
    """
    Complete framework for learning workflows from scratch
    """
    
    def __init__(self, state_dim: int = 52, workflow_dim: int = 8, action_dim: int = 27):
        self.state_dim = state_dim
        self.workflow_dim = workflow_dim
        self.action_dim = action_dim
        
        # Main policy with diversity regularization
        self.policy = RegularizedWorkflowLearning(state_dim, workflow_dim, action_dim)
        
        # Implicit semantics through action masking
        self.implicit_semantics = ImplicitWorkflowSemantics(workflow_dim, action_dim)
        
        # Self-supervised discovery
        self.discovery = SelfSupervisedWorkflowDiscovery(workflow_dim)
        
        # Progressive differentiation
        self.evolution = ProgressiveDifferentiation(workflow_dim)
        
    def train_step(self, state: torch.Tensor, workflow: torch.Tensor,
                   action: torch.Tensor, reward: float, 
                   next_state: torch.Tensor, done: bool,
                   optimizer: torch.optim.Optimizer) -> Dict:
        """
        Single training step without any pre-training
        """
        
        # Get policy output
        policy_output = self.policy(state, workflow)
        
        # Apply implicit semantics (soft action masking)
        modulated_output = self.implicit_semantics.modulate_policy(policy_output, workflow)
        
        # Compute losses
        # 1. Standard RL loss
        selected_prob = modulated_output.gather(1, action.unsqueeze(1))
        rl_loss = -torch.log(selected_prob + 1e-8) * reward
        
        # 2. Diversity regularization (ensures different workflows → different behaviors)
        diversity_loss = self.policy.compute_diversity_loss(workflow, modulated_output)
        
        # 3. Entropy bonus (exploration)
        entropy = -(modulated_output * torch.log(modulated_output + 1e-8)).sum(dim=-1)
        entropy_loss = -0.01 * entropy  # Negative because we want to maximize entropy
        
        # Total loss
        total_loss = rl_loss.mean() + 0.1 * diversity_loss + entropy_loss.mean()
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update policy memory for diversity tracking
        self.policy.update_memory(workflow, modulated_output)
        
        return {
            'rl_loss': rl_loss.mean().item(),
            'diversity_loss': diversity_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item()
        }
    
    def discover_semantics(self, workflow: np.ndarray, 
                          episode_data: List[Tuple]) -> np.ndarray:
        """
        Discover what a workflow means from episode data
        """
        patterns = self.discovery.discover_pattern(workflow, episode_data)
        semantic_embedding = self.discovery.create_semantic_embedding(workflow)
        return semantic_embedding
    
    def evolve_population(self, performance_scores: List[float]) -> List[np.ndarray]:
        """
        Evolve workflow population based on performance
        """
        self.evolution.evolve_workflows(performance_scores)
        return self.evolution.get_diverse_workflows(10)


# ============================================================================
# KEY INSIGHT: HOW SEMANTICS EMERGE WITHOUT PRE-TRAINING
# ============================================================================

def explain_emergent_semantics():
    """
    Explain how workflows gain meaning without behavioral cloning
    """
    
    explanation = """
    HOW WORKFLOWS GAIN MEANING FROM SCRATCH:
    ========================================
    
    WITHOUT BEHAVIORAL CLONING, semantics emerge through:
    
    1. CONSISTENCY:
       - Same workflow for many episodes
       - Network learns: "This vector → these behaviors work"
       - Meaning emerges from consistent conditioning
    
    2. DIVERSITY REGULARIZATION:
       - Loss function ensures different workflows → different behaviors
       - Similar workflows → similar behaviors
       - This creates meaningful distinctions naturally
    
    3. IMPLICIT CONSTRAINTS:
       - Workflows modulate action preferences (soft masking)
       - Different workflows make different actions more likely
       - Semantics emerge from these constraints
    
    4. SELF-SUPERVISED DISCOVERY:
       - Track what behaviors each workflow produces
       - Build semantic understanding from observations
       - No pre-training needed - learn from experience
    
    5. EVOLUTIONARY DIFFERENTIATION:
       - Start with random workflows
       - Evolve them to be more distinct
       - Good workflows survive and differentiate
    
    THE LEARNING PROCESS:
    ====================
    
    Early Stage (Random Exploration):
    - Workflows are random vectors
    - Policy explores randomly
    - Discovery system tracks patterns
    
    Middle Stage (Pattern Formation):
    - Certain workflows consistently produce certain behaviors
    - Diversity loss ensures workflows differentiate
    - Patterns start to stabilize
    
    Late Stage (Semantic Stability):
    - Each workflow has consistent meaning
    - Network has learned workflow → behavior mapping
    - GP-UCB can effectively search workflow space
    
    KEY: We don't need to know what workflows mean a priori.
    The meaning emerges from:
    - Consistent use (same workflow → many episodes)
    - Diversity pressure (different workflows must differ)
    - Performance feedback (good patterns survive)
    
    PRACTICAL IMPLEMENTATION:
    ========================
    
    1. Start with diverse random workflows
    2. For each workflow:
       - Train for N episodes (consistency)
       - Apply diversity regularization
       - Track discovered patterns
    3. Evolve population based on performance
    4. GP-UCB learns which regions of workflow space are good
    5. Over time, semantics stabilize and become meaningful
    
    No demonstrations needed - just exploration and consistency!
    """
    
    print(explanation)


if __name__ == "__main__":
    print("Learning Workflows from Scratch - No Behavioral Cloning Needed!")
    print("=" * 70)
    
    # Initialize system
    trainer = WorkflowTrainingFromScratch()
    
    # Create initial diverse workflows
    initial_workflows = EmergentWorkflowSemantics(8).create_diverse_workflows(10)
    
    print(f"\nCreated {len(initial_workflows)} diverse initial workflows")
    print("First workflow:", initial_workflows[0][:4], "...")
    
    # Explain the approach
    explain_emergent_semantics()
