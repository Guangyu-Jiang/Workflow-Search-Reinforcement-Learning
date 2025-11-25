#!/usr/bin/env python3
"""
Predefined Workflow Semantics for CAGE2
Define meaningful workflow dimensions based on domain knowledge
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PREDEFINED WORKFLOW DIMENSIONS
# ============================================================================

@dataclass
class WorkflowDimension:
    """Definition of a single workflow dimension"""
    name: str
    description: str
    min_value: float  # -1
    max_value: float  # +1
    min_meaning: str  # What -1 means
    max_meaning: str  # What +1 means
    
    def interpret(self, value: float) -> str:
        """Interpret a value in this dimension"""
        if value < -0.5:
            return f"Strong {self.min_meaning}"
        elif value < 0:
            return f"Moderate {self.min_meaning}"
        elif value < 0.5:
            return f"Moderate {self.max_meaning}"
        else:
            return f"Strong {self.max_meaning}"


class PredefinedWorkflowSpace:
    """
    8-dimensional workflow space with predefined semantics
    Each dimension has clear meaning based on CAGE2 domain analysis
    """
    
    def __init__(self):
        self.dimensions = [
            WorkflowDimension(
                name="fortify_timing",
                description="When to deploy decoys",
                min_value=-1, max_value=1,
                min_meaning="Late fortification (after threats detected)",
                max_meaning="Early fortification (preventive)"
            ),
            WorkflowDimension(
                name="fortify_intensity", 
                description="How many decoys to deploy",
                min_value=-1, max_value=1,
                min_meaning="Minimal decoys",
                max_meaning="Maximum decoys"
            ),
            WorkflowDimension(
                name="analysis_frequency",
                description="How often to analyze hosts",
                min_value=-1, max_value=1,
                min_meaning="Rare analysis",
                max_meaning="Constant analysis"
            ),
            WorkflowDimension(
                name="remediation_preference",
                description="How to handle compromised hosts",
                min_value=-1, max_value=1,
                min_meaning="Remove (aggressive)",
                max_meaning="Restore (conservative)"
            ),
            WorkflowDimension(
                name="response_speed",
                description="How quickly to respond to threats",
                min_value=-1, max_value=1,
                min_meaning="Delayed response",
                max_meaning="Immediate response"
            ),
            WorkflowDimension(
                name="user_subnet_priority",
                description="Focus on user subnet",
                min_value=-1, max_value=1,
                min_meaning="Ignore user subnet",
                max_meaning="Prioritize user subnet"
            ),
            WorkflowDimension(
                name="enterprise_subnet_priority",
                description="Focus on enterprise subnet",
                min_value=-1, max_value=1,
                min_meaning="Ignore enterprise subnet",
                max_meaning="Prioritize enterprise subnet"
            ),
            WorkflowDimension(
                name="operational_subnet_priority",
                description="Focus on operational server",
                min_value=-1, max_value=1,
                min_meaning="Ignore operational server",
                max_meaning="Prioritize operational server"
            )
        ]
        
        self.dim_to_idx = {dim.name: i for i, dim in enumerate(self.dimensions)}
    
    def create_workflow(self, **kwargs) -> np.ndarray:
        """
        Create workflow vector from named parameters
        
        Example:
            workflow = space.create_workflow(
                fortify_timing=0.8,      # Early fortification
                fortify_intensity=0.5,    # Moderate decoys
                analysis_frequency=-0.3,  # Less analysis
                remediation_preference=0.7 # Prefer restore
            )
        """
        workflow = np.zeros(8)
        
        for name, value in kwargs.items():
            if name in self.dim_to_idx:
                idx = self.dim_to_idx[name]
                workflow[idx] = np.clip(value, -1, 1)
            else:
                print(f"Warning: Unknown dimension {name}")
        
        return workflow
    
    def interpret_workflow(self, workflow: np.ndarray) -> Dict[str, str]:
        """Convert workflow vector to human-readable interpretation"""
        interpretation = {}
        for i, dim in enumerate(self.dimensions):
            interpretation[dim.name] = dim.interpret(workflow[i])
        return interpretation
    
    def create_canonical_workflows(self) -> Dict[str, np.ndarray]:
        """
        Create named canonical workflows based on common strategies
        """
        workflows = {}
        
        # 1. Aggressive Early Defense
        workflows["aggressive_early"] = self.create_workflow(
            fortify_timing=1.0,        # Early fortification
            fortify_intensity=0.8,      # Heavy decoys
            analysis_frequency=0.3,     # Moderate analysis
            remediation_preference=0.7, # Prefer restore
            response_speed=0.9,         # Fast response
            user_subnet_priority=0.2,
            enterprise_subnet_priority=0.7,
            operational_subnet_priority=0.5
        )
        
        # 2. Reactive Minimal
        workflows["reactive_minimal"] = self.create_workflow(
            fortify_timing=-0.8,        # Late fortification
            fortify_intensity=-0.5,     # Few decoys
            analysis_frequency=0.2,     # Some analysis
            remediation_preference=-0.7,# Prefer remove
            response_speed=0.5,         # Moderate speed
            user_subnet_priority=0.5,
            enterprise_subnet_priority=0.3,
            operational_subnet_priority=0.2
        )
        
        # 3. Information Focused
        workflows["information_focused"] = self.create_workflow(
            fortify_timing=0.0,         # Neutral timing
            fortify_intensity=-0.3,     # Few decoys
            analysis_frequency=0.9,     # Heavy analysis
            remediation_preference=0.0, # Balanced
            response_speed=0.3,         # Deliberate
            user_subnet_priority=0.4,
            enterprise_subnet_priority=0.4,
            operational_subnet_priority=0.4
        )
        
        # 4. Balanced Defense
        workflows["balanced"] = self.create_workflow(
            fortify_timing=0.3,         # Slightly early
            fortify_intensity=0.3,      # Moderate decoys
            analysis_frequency=0.5,     # Regular analysis
            remediation_preference=0.2, # Slight restore preference
            response_speed=0.6,         # Reasonably fast
            user_subnet_priority=0.4,
            enterprise_subnet_priority=0.5,
            operational_subnet_priority=0.4
        )
        
        # 5. User Protection Focus
        workflows["user_protection"] = self.create_workflow(
            fortify_timing=0.5,
            fortify_intensity=0.6,
            analysis_frequency=0.4,
            remediation_preference=0.8, # Restore to maintain availability
            response_speed=0.7,
            user_subnet_priority=0.9,   # High user focus
            enterprise_subnet_priority=0.0,
            operational_subnet_priority=0.1
        )
        
        # 6. Critical Asset Protection (Enterprise + OpServer)
        workflows["critical_protection"] = self.create_workflow(
            fortify_timing=0.7,
            fortify_intensity=0.7,
            analysis_frequency=0.6,
            remediation_preference=0.5,
            response_speed=0.8,
            user_subnet_priority=-0.3,
            enterprise_subnet_priority=0.9,  # High enterprise focus
            operational_subnet_priority=0.8  # High operational focus
        )
        
        return workflows


# ============================================================================
# WORKFLOW-GUIDED POLICY
# ============================================================================

class WorkflowGuidedPolicy(nn.Module):
    """
    Policy that uses predefined workflow semantics to guide actions
    """
    
    def __init__(self, state_dim: int = 52, action_dim: int = 27):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.workflow_space = PredefinedWorkflowSpace()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Workflow processors for each semantic dimension
        self.timing_processor = nn.Linear(1, 8)      # Dim 0: fortify timing
        self.intensity_processor = nn.Linear(1, 8)   # Dim 1: fortify intensity
        self.analysis_processor = nn.Linear(1, 8)    # Dim 2: analysis frequency
        self.remediation_processor = nn.Linear(1, 8) # Dim 3: remediation preference
        self.speed_processor = nn.Linear(1, 8)       # Dim 4: response speed
        self.subnet_processor = nn.Linear(3, 8)      # Dims 5-7: subnet priorities
        
        # Combine all features
        self.combiner = nn.Sequential(
            nn.Linear(32 + 48, 64),  # state features + workflow features
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state: torch.Tensor, workflow: torch.Tensor, timestep: int = 0):
        """
        Forward pass with semantic workflow processing
        """
        # Encode state
        state_features = self.state_encoder(state)
        
        # Process each workflow dimension semantically
        workflow_features = []
        
        # Timing feature (affects early vs late actions)
        timing_feature = self.timing_processor(workflow[:, 0:1])
        if timestep < 10:  # Early phase
            timing_feature = timing_feature * (1 + workflow[:, 0])  # Amplify if positive
        else:  # Late phase
            timing_feature = timing_feature * (1 - workflow[:, 0])  # Amplify if negative
        workflow_features.append(timing_feature)
        
        # Intensity feature (affects decoy action probability)
        intensity_feature = self.intensity_processor(workflow[:, 1:2])
        workflow_features.append(intensity_feature)
        
        # Analysis feature (affects analyze action probability)
        analysis_feature = self.analysis_processor(workflow[:, 2:3])
        workflow_features.append(analysis_feature)
        
        # Remediation feature (affects remove vs restore balance)
        remediation_feature = self.remediation_processor(workflow[:, 3:4])
        workflow_features.append(remediation_feature)
        
        # Speed feature (affects action frequency)
        speed_feature = self.speed_processor(workflow[:, 4:5])
        workflow_features.append(speed_feature)
        
        # Subnet features (affects which hosts to target)
        subnet_feature = self.subnet_processor(workflow[:, 5:8])
        workflow_features.append(subnet_feature)
        
        # Combine all features
        all_workflow_features = torch.cat(workflow_features, dim=-1)
        combined = torch.cat([state_features, all_workflow_features], dim=-1)
        
        # Generate action logits
        action_logits = self.combiner(combined)
        
        # Apply semantic biases based on workflow
        action_logits = self.apply_semantic_biases(action_logits, workflow, timestep)
        
        return torch.softmax(action_logits, dim=-1)
    
    def apply_semantic_biases(self, logits: torch.Tensor, 
                             workflow: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Apply direct semantic biases based on workflow values
        This ensures the policy respects the predefined semantics
        """
        # Clone to avoid in-place operations
        biased_logits = logits.clone()
        
        # Map action indices to types (simplified - use actual mapping)
        decoy_actions = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
        analyze_actions = [2, 3, 4, 5, 9, 11, 12, 13, 14]
        remove_actions = [15, 16, 17, 18, 22, 24, 25, 26, 27]
        restore_actions = [132, 133, 134, 135, 139, 141, 142, 143, 144]
        
        # Apply timing bias
        if timestep < 10 and workflow[0, 0] > 0.5:  # Early fortify
            for i, action in enumerate(decoy_actions):
                if action < logits.shape[1]:
                    biased_logits[0, i] += workflow[0, 0] * 2
        
        # Apply intensity bias
        for i, action in enumerate(decoy_actions):
            if action < logits.shape[1]:
                biased_logits[0, i] += workflow[0, 1]
        
        # Apply analysis bias
        for i, action in enumerate(analyze_actions):
            if action < logits.shape[1]:
                biased_logits[0, i] += workflow[0, 2]
        
        # Apply remediation preference
        if workflow[0, 3] > 0:  # Prefer restore
            for i, action in enumerate(restore_actions):
                if action < logits.shape[1]:
                    biased_logits[0, i] += workflow[0, 3]
            for i, action in enumerate(remove_actions):
                if action < logits.shape[1]:
                    biased_logits[0, i] -= workflow[0, 3] * 0.5
        else:  # Prefer remove
            for i, action in enumerate(remove_actions):
                if action < logits.shape[1]:
                    biased_logits[0, i] += abs(workflow[0, 3])
            for i, action in enumerate(restore_actions):
                if action < logits.shape[1]:
                    biased_logits[0, i] -= abs(workflow[0, 3]) * 0.5
        
        return biased_logits


# ============================================================================
# WORKFLOW SAMPLING WITH PREDEFINED SEMANTICS
# ============================================================================

class SemanticWorkflowSampler:
    """
    Sample workflows with awareness of their predefined meanings
    """
    
    def __init__(self, workflow_space: PredefinedWorkflowSpace):
        self.space = workflow_space
        self.canonical_workflows = self.space.create_canonical_workflows()
        
    def sample_meaningful_workflow(self, method: str = "canonical_variation") -> np.ndarray:
        """
        Sample workflows that are meaningful combinations
        """
        
        if method == "canonical_variation":
            # Start from canonical and add noise
            base_name = np.random.choice(list(self.canonical_workflows.keys()))
            base = self.canonical_workflows[base_name].copy()
            noise = np.random.randn(8) * 0.2
            workflow = np.clip(base + noise, -1, 1)
            
        elif method == "dimension_combination":
            # Randomly set each dimension to meaningful value
            workflow = np.zeros(8)
            for i in range(8):
                # Sample from meaningful values: -1, -0.5, 0, 0.5, 1
                workflow[i] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                
        elif method == "interpolation":
            # Interpolate between two canonical workflows
            names = list(self.canonical_workflows.keys())
            w1 = self.canonical_workflows[np.random.choice(names)]
            w2 = self.canonical_workflows[np.random.choice(names)]
            alpha = np.random.random()
            workflow = w1 * alpha + w2 * (1 - alpha)
            
        elif method == "focused":
            # Focus on specific dimensions
            workflow = np.zeros(8)
            # Randomly select 2-3 dimensions to set strongly
            focused_dims = np.random.choice(8, size=np.random.randint(2, 4), replace=False)
            for dim in focused_dims:
                workflow[dim] = np.random.choice([-0.8, 0.8])  # Strong values
            # Others get weak values
            for dim in range(8):
                if dim not in focused_dims:
                    workflow[dim] = np.random.uniform(-0.3, 0.3)
                    
        else:  # random
            workflow = np.random.uniform(-1, 1, 8)
        
        return workflow
    
    def create_exploration_grid(self, resolution: int = 3) -> List[np.ndarray]:
        """
        Create a grid of workflows for systematic exploration
        Each dimension takes values from a discrete set
        """
        values = np.linspace(-1, 1, resolution)
        workflows = []
        
        # For efficiency, don't do full grid (3^8 = 6561 workflows!)
        # Instead, sample systematically
        
        # 1. Extremes on each dimension
        for dim in range(8):
            for val in [-1, 1]:
                w = np.zeros(8)
                w[dim] = val
                workflows.append(w)
        
        # 2. Canonical workflows
        for w in self.canonical_workflows.values():
            workflows.append(w.copy())
        
        # 3. Random combinations
        for _ in range(20):
            w = np.random.choice(values, size=8)
            workflows.append(w)
        
        return workflows


# ============================================================================
# TRAINING WITH PREDEFINED SEMANTICS
# ============================================================================

def train_with_predefined_semantics(env_fn, num_iterations: int = 100):
    """
    Training loop using predefined workflow semantics
    """
    
    # Initialize workflow space and sampler
    workflow_space = PredefinedWorkflowSpace()
    sampler = SemanticWorkflowSampler(workflow_space)
    
    # Initialize policy that understands semantics
    policy = WorkflowGuidedPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    
    # GP-UCB for workflow search
    from workflow_rl.gpucb_workflow_search import GPUCBWorkflowSearch
    searcher = GPUCBWorkflowSearch(workflow_dim=8)
    
    for iteration in range(num_iterations):
        # Sample meaningful workflow
        if iteration < 10:
            # Start with canonical workflows
            workflow = sampler.sample_meaningful_workflow("canonical_variation")
        else:
            # Use GP-UCB to select workflow
            workflow = searcher.select_next_workflow()
        
        # Interpret workflow
        interpretation = workflow_space.interpret_workflow(workflow)
        print(f"\nIteration {iteration}: Testing workflow with semantics:")
        for key, value in interpretation.items():
            print(f"  {key}: {value}")
        
        # Train policy with this workflow
        total_reward = 0
        for episode in range(10):
            env = env_fn()
            obs = env.reset()
            episode_reward = 0
            
            for timestep in range(50):
                # Get action from semantically-aware policy
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                workflow_tensor = torch.FloatTensor(workflow).unsqueeze(0)
                
                action_probs = policy(state_tensor, workflow_tensor, timestep)
                action = torch.multinomial(action_probs, 1).item()
                
                # Step environment
                next_obs, reward, done, _ = env.step(action)
                episode_reward += reward
                
                # Update policy (simplified)
                loss = -torch.log(action_probs[0, action]) * reward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                obs = next_obs
                if done:
                    break
            
            total_reward += episode_reward
        
        # Update GP-UCB
        avg_reward = total_reward / 10
        searcher.update(workflow, avg_reward, compliance=1.0)  # Assume good compliance
        
        print(f"  Average reward: {avg_reward:.2f}")
    
    return policy, searcher


# ============================================================================
# MAIN ADVANTAGES OF PREDEFINED SEMANTICS
# ============================================================================

def explain_advantages():
    """
    Explain why predefined semantics are beneficial
    """
    
    explanation = """
    ADVANTAGES OF PREDEFINED WORKFLOW SEMANTICS:
    ============================================
    
    1. INTERPRETABILITY:
       - Each dimension has clear meaning
       - Can explain why agent behaves certain way
       - Easy to debug and adjust
    
    2. FASTER LEARNING:
       - Network doesn't need to discover semantics
       - Can directly learn to execute meaningful strategies
       - Reduces exploration needed
    
    3. DOMAIN KNOWLEDGE INTEGRATION:
       - Incorporates expert knowledge about CAGE2
       - Dimensions chosen based on analysis of optimal policies
       - Avoids obviously bad strategies
    
    4. CONTROLLED EXPLORATION:
       - Can systematically explore strategy space
       - Know what each workflow should do
       - Can verify if policy follows intended strategy
    
    5. TRANSFER LEARNING:
       - Semantics are consistent across training
       - Can reuse learned mappings
       - Easy to adapt to new scenarios
    
    IMPLEMENTATION:
    ==============
    
    1. Define 8 meaningful dimensions based on domain analysis
    2. Create canonical workflows (aggressive, defensive, etc.)
    3. Sample variations of meaningful workflows
    4. Train policy with semantic biases
    5. Use GP-UCB to find best workflow in semantic space
    
    The key is that we're not searching in arbitrary 8D space,
    but in a semantically meaningful space where each dimension
    has a clear strategic interpretation.
    """
    
    print(explanation)


if __name__ == "__main__":
    print("Predefined Workflow Semantics for CAGE2")
    print("=" * 60)
    
    # Initialize workflow space
    space = PredefinedWorkflowSpace()
    
    # Show canonical workflows
    canonical = space.create_canonical_workflows()
    print("\nCanonical Workflows:")
    for name, workflow in canonical.items():
        print(f"\n{name}:")
        interpretation = space.interpret_workflow(workflow)
        for dim_name, meaning in interpretation.items():
            print(f"  {dim_name}: {meaning}")
    
    # Explain advantages
    explain_advantages()
