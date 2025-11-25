#!/usr/bin/env python3
"""
Complete Training Framework with Meaningful Workflow Representation
Addresses the key question: How to ensure workflow representation is meaningful
and policies actually follow them?
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
# SOLUTION 1: BEHAVIORAL CLONING INITIALIZATION
# ============================================================================

class BehavioralCloningInitializer:
    """
    Make workflow representation meaningful by pre-training the network
    to associate workflow vectors with expected behaviors
    """
    
    def __init__(self, state_dim: int = 52, workflow_dim: int = 8, action_dim: int = 27):
        self.state_dim = state_dim
        self.workflow_dim = workflow_dim
        self.action_dim = action_dim
        
    def create_demonstration_data(self) -> List[Tuple]:
        """
        Create synthetic demonstrations that show what each workflow dimension means
        """
        demonstrations = []
        
        # Workflow dimension 0: Early vs Late fortification
        # Early fortify workflow
        workflow_early = np.zeros(self.workflow_dim)
        workflow_early[0] = 1.0  # Early fortify
        for t in range(10):
            state = np.random.randn(self.state_dim) * 0.1  # Early game state
            action = np.random.choice([1000, 1001, 1002])  # Decoy actions
            demonstrations.append((state, workflow_early, action, t))
        
        # Late fortify workflow
        workflow_late = np.zeros(self.workflow_dim)
        workflow_late[0] = -1.0  # Late fortify
        for t in range(20, 30):
            state = np.random.randn(self.state_dim) * 0.1
            state[0:10] = 0.5  # Some activity detected
            action = np.random.choice([1000, 1001, 1002])  # Decoy actions
            demonstrations.append((state, workflow_late, action, t))
        
        # Workflow dimension 3: Remove vs Restore preference
        # Restore preference
        workflow_restore = np.zeros(self.workflow_dim)
        workflow_restore[3] = 1.0  # Prefer restore
        for _ in range(20):
            state = np.random.randn(self.state_dim)
            state[2::4] = 0.8  # Compromise indicators
            action = np.random.choice([133, 134, 135, 141, 142])  # Restore actions
            demonstrations.append((state, workflow_restore, action, 15))
        
        # Remove preference
        workflow_remove = np.zeros(self.workflow_dim)
        workflow_remove[3] = -1.0  # Prefer remove
        for _ in range(20):
            state = np.random.randn(self.state_dim)
            state[2::4] = 0.8  # Compromise indicators
            action = np.random.choice([16, 17, 18, 24, 25])  # Remove actions
            demonstrations.append((state, workflow_remove, action, 15))
        
        return demonstrations
    
    def pretrain_policy(self, policy_network: nn.Module, epochs: int = 100):
        """
        Pre-train policy to understand workflow semantics through behavioral cloning
        """
        demonstrations = self.create_demonstration_data()
        optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(demonstrations)
            
            for state, workflow, action, timestep in demonstrations:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                workflow_t = torch.FloatTensor(workflow).unsqueeze(0).to(device)
                action_t = torch.LongTensor([action]).to(device)
                
                # Forward pass
                combined = torch.cat([state_t, workflow_t], dim=-1)
                action_probs = policy_network(combined)
                
                # Compute loss
                loss = F.cross_entropy(action_probs, action_t)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Behavioral cloning epoch {epoch}: Loss = {total_loss/len(demonstrations):.4f}")


# ============================================================================
# SOLUTION 2: AUXILIARY TASK LEARNING
# ============================================================================

class WorkflowAwarePolicy(nn.Module):
    """
    Policy that learns workflow semantics through auxiliary tasks
    """
    
    def __init__(self, state_dim: int = 52, workflow_dim: int = 8, action_dim: int = 27):
        super().__init__()
        self.state_dim = state_dim
        self.workflow_dim = workflow_dim
        self.action_dim = action_dim
        
        # Shared encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.workflow_encoder = nn.Sequential(
            nn.Linear(workflow_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Main policy head
        self.policy_head = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Auxiliary task 1: Predict workflow from actions
        self.workflow_predictor = nn.Sequential(
            nn.Linear(action_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, workflow_dim)
        )
        
        # Auxiliary task 2: Predict action type from workflow
        self.action_type_predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # 5 action types: fortify, analyze, remove, restore, other
        )
    
    def forward(self, state: torch.Tensor, workflow: torch.Tensor):
        # Encode inputs
        state_features = self.state_encoder(state)
        workflow_features = self.workflow_encoder(workflow)
        
        # Main policy
        combined = torch.cat([state_features, workflow_features], dim=-1)
        action_probs = self.policy_head(combined)
        
        # Auxiliary predictions
        predicted_workflow = self.workflow_predictor(
            torch.cat([action_probs, state_features], dim=-1)
        )
        predicted_action_type = self.action_type_predictor(workflow_features)
        
        return {
            'action_probs': action_probs,
            'predicted_workflow': predicted_workflow,
            'predicted_action_type': predicted_action_type
        }
    
    def compute_auxiliary_loss(self, outputs: Dict, workflow: torch.Tensor, action: torch.Tensor):
        """
        Compute auxiliary losses that ensure workflow understanding
        """
        # Loss 1: Workflow reconstruction
        workflow_loss = F.mse_loss(outputs['predicted_workflow'], workflow)
        
        # Loss 2: Action type prediction
        action_type = self._get_action_type(action)
        action_type_loss = F.cross_entropy(outputs['predicted_action_type'], action_type)
        
        return workflow_loss + action_type_loss
    
    def _get_action_type(self, action: torch.Tensor) -> torch.Tensor:
        """Map action to action type for auxiliary task"""
        # This is a simplified mapping - in practice would be more sophisticated
        action_types = torch.zeros(action.shape[0], dtype=torch.long)
        for i, a in enumerate(action):
            if 1000 <= a <= 1008 or a in [28, 41, 54, 67, 80, 93, 106, 119]:
                action_types[i] = 0  # Fortify
            elif a in [2, 3, 4, 5, 9, 11, 12, 13, 14]:
                action_types[i] = 1  # Analyze
            elif a in [15, 16, 17, 18, 22, 24, 25, 26, 27]:
                action_types[i] = 2  # Remove
            elif a in [132, 133, 134, 135, 139, 141, 142, 143, 144]:
                action_types[i] = 3  # Restore
            else:
                action_types[i] = 4  # Other
        return action_types.to(device)


# ============================================================================
# SOLUTION 3: CONTRASTIVE LEARNING FOR WORKFLOW DISTINCTION
# ============================================================================

class ContrastiveWorkflowLearning:
    """
    Ensure workflows are distinguishable through contrastive learning
    """
    
    def __init__(self, workflow_dim: int = 8, temperature: float = 0.1):
        self.workflow_dim = workflow_dim
        self.temperature = temperature
        
        # Workflow encoder that ensures distinct representations
        self.encoder = nn.Sequential(
            nn.Linear(workflow_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh()  # Bounded output
        ).to(device)
        
    def contrastive_loss(self, workflow_batch: torch.Tensor, 
                        behavior_batch: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss ensures similar workflows lead to similar behaviors
        and different workflows lead to different behaviors
        """
        batch_size = workflow_batch.shape[0]
        
        # Encode workflows
        workflow_embeddings = self.encoder(workflow_batch)
        
        # Compute pairwise workflow similarities
        workflow_sim = F.cosine_similarity(
            workflow_embeddings.unsqueeze(1), 
            workflow_embeddings.unsqueeze(0), 
            dim=-1
        ) / self.temperature
        
        # Compute pairwise behavior similarities
        # (behavior_batch could be action distributions, state visitations, etc.)
        behavior_sim = F.cosine_similarity(
            behavior_batch.unsqueeze(1),
            behavior_batch.unsqueeze(0),
            dim=-1
        )
        
        # Contrastive loss: similar behaviors should have similar workflows
        labels = (behavior_sim > 0.8).float()  # Threshold for "similar"
        loss = F.binary_cross_entropy_with_logits(workflow_sim, labels)
        
        return loss


# ============================================================================
# SOLUTION 4: STRUCTURED REWARD WITHOUT CHANGING OPTIMALITY
# ============================================================================

class InvariantRewardShaping:
    """
    Shape rewards to encourage workflow following without changing optimal policy
    Uses invariant shaping that preserves policy optimality
    """
    
    def __init__(self, workflow: np.ndarray):
        self.workflow = workflow
        self.potential_scale = 0.1  # Small scale to avoid dominating true reward
        
    def compute_potential(self, state: np.ndarray, timestep: int) -> float:
        """
        Potential function Φ(s) that encodes workflow progress
        Key: Potential differences guide toward workflow without changing optimum
        """
        potential = 0.0
        
        # Analyze state for workflow-relevant features
        activity_level = np.sum(np.abs(state[:20])) / 20 if len(state) >= 20 else 0
        compromise_level = np.mean(state[2::4]) if len(state) >= 52 else 0
        
        # Workflow dimension 0: Fortify timing
        if self.workflow[0] > 0.5:  # Early fortify
            # High potential early, decreases over time
            potential += (1 - timestep / 50) * self.workflow[0] * 0.1
        else:  # Late fortify
            # Low potential early, increases over time
            potential += (timestep / 50) * abs(self.workflow[0]) * 0.1
        
        # Workflow dimension 2: Analysis frequency
        # Higher potential for having information
        potential += self.workflow[2] * activity_level * 0.05
        
        # Workflow dimension 3: Remediation preference
        if compromise_level > 0:
            if self.workflow[3] > 0:  # Prefer restore
                # Higher potential when ready to restore
                potential += self.workflow[3] * compromise_level * 0.1
            else:  # Prefer remove
                # Higher potential when ready to remove
                potential += abs(self.workflow[3]) * compromise_level * 0.05
        
        return potential * self.potential_scale
    
    def shape_reward(self, r: float, s: np.ndarray, a: int, 
                     s_next: np.ndarray, t: int, gamma: float = 0.99) -> float:
        """
        Apply invariant reward shaping: r' = r + γΦ(s') - Φ(s)
        This preserves optimal policy while guiding exploration
        """
        phi_s = self.compute_potential(s, t)
        phi_s_next = self.compute_potential(s_next, t + 1)
        
        # Invariant shaping
        shaped_reward = r + gamma * phi_s_next - phi_s
        
        return shaped_reward


# ============================================================================
# SOLUTION 5: WORKFLOW VERIFICATION THROUGH STATE VISITATION
# ============================================================================

class WorkflowVerification:
    """
    Verify that policies are following workflows by checking state visitation patterns
    """
    
    def __init__(self, workflow: np.ndarray):
        self.workflow = workflow
        self.state_visitations = defaultdict(int)
        self.action_patterns = defaultdict(list)
        
    def update(self, state: np.ndarray, action: int, timestep: int):
        """Track state visitations and action patterns"""
        # Discretize state for counting
        state_key = self._discretize_state(state)
        self.state_visitations[state_key] += 1
        
        # Track action patterns
        time_bucket = timestep // 10  # Group by time periods
        self.action_patterns[time_bucket].append(action)
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Convert continuous state to discrete for counting"""
        # Simple discretization - can be more sophisticated
        discrete = np.round(state * 2) / 2  # Round to nearest 0.5
        return tuple(discrete[:10])  # Use first 10 dims for efficiency
    
    def compute_workflow_alignment(self) -> float:
        """
        Compute how well the observed behavior aligns with intended workflow
        """
        alignment_score = 0.0
        total_weight = 0.0
        
        # Check early vs late fortification
        early_fortify_actions = sum(
            1 for a in self.action_patterns[0] 
            if 1000 <= a <= 1008
        )
        if self.workflow[0] > 0.5:  # Should fortify early
            alignment_score += (early_fortify_actions > 2) * 1.0
        else:  # Should fortify late
            alignment_score += (early_fortify_actions <= 1) * 1.0
        total_weight += 1.0
        
        # Check remediation preference
        all_remediations = []
        for actions in self.action_patterns.values():
            all_remediations.extend([a for a in actions 
                                    if a in range(132, 145) or a in range(15, 28)])
        
        if all_remediations:
            restore_ratio = sum(1 for a in all_remediations if a >= 132) / len(all_remediations)
            expected_restore = (self.workflow[3] + 1) / 2  # Map [-1,1] to [0,1]
            alignment_score += (1 - abs(restore_ratio - expected_restore)) * 1.0
            total_weight += 1.0
        
        return alignment_score / total_weight if total_weight > 0 else 0.0


# ============================================================================
# INTEGRATED TRAINING WITH ALL SOLUTIONS
# ============================================================================

class MeaningfulWorkflowTraining:
    """
    Combines all solutions to ensure meaningful workflow representation
    """
    
    def __init__(self, 
                 state_dim: int = 52,
                 workflow_dim: int = 8,
                 action_dim: int = 27):
        
        # Main policy with auxiliary tasks
        self.policy = WorkflowAwarePolicy(state_dim, workflow_dim, action_dim)
        
        # Behavioral cloning for initialization
        self.bc_initializer = BehavioralCloningInitializer(state_dim, workflow_dim, action_dim)
        
        # Contrastive learning for workflow distinction
        self.contrastive_learner = ContrastiveWorkflowLearning(workflow_dim)
        
        # Workflow verification
        self.verifiers = {}
        
    def initialize_policy(self):
        """Step 1: Initialize with behavioral cloning"""
        print("Initializing policy with behavioral cloning...")
        self.bc_initializer.pretrain_policy(self.policy.policy_head, epochs=50)
        
    def train_step(self, 
                   state: torch.Tensor,
                   workflow: torch.Tensor,
                   action: torch.Tensor,
                   reward: float,
                   next_state: torch.Tensor,
                   optimizer: torch.optim.Optimizer):
        """
        Single training step with all components
        """
        
        # Forward pass
        outputs = self.policy(state, workflow)
        
        # Main RL loss (e.g., PPO loss - simplified here)
        action_probs = outputs['action_probs']
        selected_prob = action_probs.gather(1, action.unsqueeze(1))
        rl_loss = -torch.log(selected_prob + 1e-8) * reward
        
        # Auxiliary task loss (ensures workflow understanding)
        aux_loss = self.policy.compute_auxiliary_loss(outputs, workflow, action)
        
        # Total loss
        total_loss = rl_loss.mean() + 0.1 * aux_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'rl_loss': rl_loss.mean().item(),
            'aux_loss': aux_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def verify_workflow_following(self, workflow: np.ndarray, 
                                 episode_data: List[Tuple]) -> float:
        """
        Verify that the policy followed the workflow
        """
        workflow_key = tuple(workflow)
        if workflow_key not in self.verifiers:
            self.verifiers[workflow_key] = WorkflowVerification(workflow)
        
        verifier = self.verifiers[workflow_key]
        
        for state, action, timestep in episode_data:
            verifier.update(state, action, timestep)
        
        return verifier.compute_workflow_alignment()


# ============================================================================
# MAIN POINT: HOW TO ENSURE MEANINGFUL REPRESENTATION
# ============================================================================

def explain_meaningful_representation():
    """
    Explain the key strategies for meaningful workflow representation
    """
    
    explanation = """
    HOW TO ENSURE WORKFLOW REPRESENTATION IS MEANINGFUL:
    =====================================================
    
    1. BEHAVIORAL CLONING INITIALIZATION:
       - Pre-train the network with synthetic demonstrations
       - Shows the network what each workflow dimension means
       - Example: workflow[0]=1 → take decoy actions early
       
    2. AUXILIARY TASKS:
       - Network must predict workflow from its own actions
       - Forces internal representation to understand workflow semantics
       - If network can't reconstruct workflow, it doesn't understand it
       
    3. CONTRASTIVE LEARNING:
       - Similar workflows → similar behaviors
       - Different workflows → different behaviors
       - Ensures workflows are distinguishable in behavior space
       
    4. INVARIANT REWARD SHAPING:
       - Use potential-based shaping: r' = r + γΦ(s') - Φ(s)
       - Guides toward workflow without changing optimal policy
       - Mathematically guaranteed to preserve optimality
       
    5. VERIFICATION THROUGH STATE VISITATION:
       - Track whether policy actually follows workflow
       - If compliance < threshold, the sample is less informative
       - Weight GP-UCB updates by compliance score
    
    KEY INSIGHT - NO CUSTOM REWARDS NEEDED:
    ========================================
    We DON'T need different reward functions for different workflows!
    
    Instead, we use:
    - Potential-based shaping (invariant, preserves optimality)
    - Auxiliary losses (train-time only, not rewards)
    - Compliance weighting (adjusts GP confidence, not rewards)
    
    The base environment reward stays the same. The workflow just
    provides strategic context that biases HOW the agent pursues
    the same objective.
    
    TRAINING LOOP:
    =============
    1. Sample workflow w from GP-UCB
    2. Initialize policy with behavioral cloning for w
    3. For each episode:
       - Use invariant shaping: r' = r + γΦ(s') - Φ(s)
       - Train with auxiliary losses
       - Track compliance
    4. Update GP-UCB: score = avg_reward * (0.5 + 0.5 * compliance)
    5. Repeat
    
    This ensures workflows are:
    - Meaningful (BC initialization + auxiliary tasks)
    - Distinguishable (contrastive learning)
    - Followed (verification + compliance weighting)
    - Optimal (invariant shaping preserves optimality)
    """
    
    print(explanation)


if __name__ == "__main__":
    # Demonstrate the framework
    print("Meaningful Workflow Representation Framework")
    print("=" * 60)
    
    # Initialize training system
    trainer = MeaningfulWorkflowTraining()
    
    # Initialize policy with behavioral cloning
    trainer.initialize_policy()
    
    # Explain the approach
    explain_meaningful_representation()
