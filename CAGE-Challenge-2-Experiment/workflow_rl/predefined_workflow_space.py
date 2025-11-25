#!/usr/bin/env python3
"""
Predefined Meaningful Workflow Space for CAGE2
Define workflow dimensions with clear semantics BEFORE training starts
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PREDEFINED WORKFLOW DIMENSIONS WITH CLEAR SEMANTICS
# ============================================================================

@dataclass
class WorkflowDimension:
    """Define what each dimension means"""
    name: str
    description: str
    low_value_meaning: str  # What -1 means
    high_value_meaning: str  # What +1 means
    action_mapping: Dict[str, List[int]]  # Which actions this dimension affects


class PredefinedWorkflowSpace:
    """
    8-dimensional workflow space with predefined semantics
    Each dimension has clear meaning defined BEFORE training
    """
    
    def __init__(self):
        # Define what each dimension means
        self.dimensions = [
            WorkflowDimension(
                name="fortify_timing",
                description="When to deploy decoys",
                low_value_meaning="Deploy decoys late (after detection)",
                high_value_meaning="Deploy decoys early (preventive)",
                action_mapping={
                    'decoy_actions': [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
                    'timing': 'early_vs_late'
                }
            ),
            WorkflowDimension(
                name="fortify_intensity", 
                description="How many decoys to deploy",
                low_value_meaning="Minimal decoys (conservative)",
                high_value_meaning="Maximum decoys (aggressive)",
                action_mapping={
                    'decoy_actions': [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
                    'frequency': 'low_vs_high'
                }
            ),
            WorkflowDimension(
                name="analysis_frequency",
                description="How often to analyze hosts",
                low_value_meaning="Analyze rarely (trust-based)",
                high_value_meaning="Analyze constantly (paranoid)",
                action_mapping={
                    'analyze_actions': [2, 3, 4, 5, 9, 11, 12, 13, 14],
                    'frequency': 'low_vs_high'
                }
            ),
            WorkflowDimension(
                name="remediation_preference",
                description="How to handle compromised hosts",
                low_value_meaning="Prefer Remove (aggressive cleanup)",
                high_value_meaning="Prefer Restore (preserve availability)",
                action_mapping={
                    'remove_actions': [15, 16, 17, 18, 22, 24, 25, 26, 27],
                    'restore_actions': [132, 133, 134, 135, 139, 141, 142, 143, 144],
                    'preference': 'remove_vs_restore'
                }
            ),
            WorkflowDimension(
                name="response_speed",
                description="How quickly to respond to threats",
                low_value_meaning="Delayed response (gather info first)",
                high_value_meaning="Immediate response (act fast)",
                action_mapping={
                    'all_defensive': 'all',
                    'speed': 'slow_vs_fast'
                }
            ),
            WorkflowDimension(
                name="user_subnet_priority",
                description="Priority for user subnet defense",
                low_value_meaning="Low priority (focus elsewhere)",
                high_value_meaning="High priority (protect users)",
                action_mapping={
                    'user_hosts': [11, 12, 13, 14, 24, 25, 26, 27, 141, 142, 143, 144, 1003, 1004, 1005, 1006],
                    'priority': 'low_vs_high'
                }
            ),
            WorkflowDimension(
                name="enterprise_subnet_priority",
                description="Priority for enterprise subnet defense",
                low_value_meaning="Low priority (focus elsewhere)",
                high_value_meaning="High priority (protect enterprise)",
                action_mapping={
                    'enterprise_hosts': [3, 4, 5, 16, 17, 18, 133, 134, 135, 1000, 1001, 1002],
                    'priority': 'low_vs_high'
                }
            ),
            WorkflowDimension(
                name="operational_subnet_priority",
                description="Priority for operational subnet defense",
                low_value_meaning="Low priority (focus elsewhere)",
                high_value_meaning="High priority (protect operational)",
                action_mapping={
                    'operational_hosts': [9, 22, 139, 1007, 1008],
                    'priority': 'low_vs_high'
                }
            )
        ]
        
        # Validate we have 8 dimensions
        assert len(self.dimensions) == 8, f"Expected 8 dimensions, got {len(self.dimensions)}"
    
    def create_canonical_workflows(self) -> Dict[str, np.ndarray]:
        """
        Create interpretable reference workflows based on known strategies
        """
        workflows = {
            "early_fortification": np.array([
                1.0,   # fortify_timing: early
                0.8,   # fortify_intensity: high
                0.3,   # analysis_frequency: moderate
                0.0,   # remediation_preference: balanced
                0.5,   # response_speed: moderate
                0.3,   # user_subnet_priority: moderate
                0.5,   # enterprise_subnet_priority: moderate  
                0.2    # operational_subnet_priority: low
            ]),
            
            "reactive_defense": np.array([
                -0.8,  # fortify_timing: late
                0.2,   # fortify_intensity: low
                0.7,   # analysis_frequency: high
                -0.5,  # remediation_preference: prefer remove
                0.9,   # response_speed: fast
                0.5,   # user_subnet_priority: moderate
                0.4,   # enterprise_subnet_priority: moderate
                0.1    # operational_subnet_priority: low
            ]),
            
            "information_focused": np.array([
                0.0,   # fortify_timing: neutral
                0.0,   # fortify_intensity: neutral
                1.0,   # analysis_frequency: maximum
                0.0,   # remediation_preference: balanced
                -0.5,  # response_speed: slow (gather info)
                0.4,   # user_subnet_priority: moderate
                0.4,   # enterprise_subnet_priority: moderate
                0.2    # operational_subnet_priority: low
            ]),
            
            "availability_focused": np.array([
                0.5,   # fortify_timing: moderate early
                0.6,   # fortify_intensity: moderate high
                0.4,   # analysis_frequency: moderate
                0.9,   # remediation_preference: strongly prefer restore
                0.7,   # response_speed: fast
                0.3,   # user_subnet_priority: moderate
                0.7,   # enterprise_subnet_priority: high (keep enterprise up)
                0.5    # operational_subnet_priority: moderate
            ]),
            
            "user_protection": np.array([
                0.3,   # fortify_timing: slightly early
                0.5,   # fortify_intensity: moderate
                0.6,   # analysis_frequency: moderate high
                0.3,   # remediation_preference: slight restore preference
                0.6,   # response_speed: moderate fast
                0.9,   # user_subnet_priority: high
                0.2,   # enterprise_subnet_priority: low
                0.1    # operational_subnet_priority: low
            ]),
            
            "enterprise_protection": np.array([
                0.4,   # fortify_timing: moderate early
                0.7,   # fortify_intensity: high
                0.5,   # analysis_frequency: moderate
                0.5,   # remediation_preference: restore preference
                0.7,   # response_speed: fast
                0.2,   # user_subnet_priority: low
                0.9,   # enterprise_subnet_priority: high
                0.3    # operational_subnet_priority: low
            ])
        }
        
        return workflows
    
    def sample_workflow(self, strategy: str = "random") -> np.ndarray:
        """
        Sample a workflow vector with known semantics
        """
        if strategy == "random":
            # Pure random in [-1, 1]
            return np.random.uniform(-1, 1, 8)
        
        elif strategy == "gaussian":
            # Gaussian around neutral (0)
            return np.clip(np.random.randn(8) * 0.5, -1, 1)
        
        elif strategy == "mixture":
            # Mix two canonical workflows
            canonicals = self.create_canonical_workflows()
            keys = list(canonicals.keys())
            w1 = canonicals[np.random.choice(keys)]
            w2 = canonicals[np.random.choice(keys)]
            alpha = np.random.random()
            return w1 * alpha + w2 * (1 - alpha)
        
        elif strategy == "perturbed_canonical":
            # Start from canonical, add noise
            canonicals = self.create_canonical_workflows()
            base = canonicals[np.random.choice(list(canonicals.keys()))]
            noise = np.random.randn(8) * 0.2
            return np.clip(base + noise, -1, 1)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def interpret_workflow(self, workflow: np.ndarray) -> Dict[str, str]:
        """
        Convert workflow vector to human-readable interpretation
        """
        interpretation = {}
        
        for i, (value, dim) in enumerate(zip(workflow, self.dimensions)):
            if value < -0.5:
                interpretation[dim.name] = dim.low_value_meaning
            elif value > 0.5:
                interpretation[dim.name] = dim.high_value_meaning
            else:
                interpretation[dim.name] = f"Balanced {dim.description}"
        
        return interpretation


# ============================================================================
# WORKFLOW-GUIDED ACTION SELECTION
# ============================================================================

class WorkflowGuidedPolicy(nn.Module):
    """
    Policy that uses predefined workflow semantics to guide action selection
    """
    
    def __init__(self, state_dim: int = 52, workflow_dim: int = 8, action_dim: int = 27):
        super().__init__()
        self.state_dim = state_dim
        self.workflow_dim = workflow_dim
        self.action_dim = action_dim
        
        # Define action groups based on workflow dimensions
        self.action_groups = {
            'decoy': [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            'analyze': [2, 3, 4, 5, 9, 11, 12, 13, 14],
            'remove': [15, 16, 17, 18, 22, 24, 25, 26, 27],
            'restore': [132, 133, 134, 135, 139, 141, 142, 143, 144],
            'sleep': [0, 1]  # Sleep/Monitor
        }
        
        # Base policy network
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Workflow processing network
        self.workflow_network = nn.Sequential(
            nn.Linear(workflow_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Combined decision network
        self.decision_network = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state: torch.Tensor, workflow: torch.Tensor, timestep: int = 0):
        """
        Generate action probabilities guided by workflow semantics
        """
        # Process state and workflow
        state_features = self.base_network(state)
        workflow_features = self.workflow_network(workflow)
        
        # Combine features
        combined = torch.cat([state_features, workflow_features], dim=-1)
        action_logits = self.decision_network(combined)
        
        # Apply workflow-based biases (using predefined semantics)
        biased_logits = self.apply_workflow_bias(action_logits, workflow, timestep)
        
        # Convert to probabilities
        action_probs = torch.softmax(biased_logits, dim=-1)
        
        return action_probs
    
    def apply_workflow_bias(self, logits: torch.Tensor, workflow: torch.Tensor, timestep: int):
        """
        Apply biases based on predefined workflow semantics
        """
        batch_size = logits.shape[0]
        biases = torch.zeros_like(logits)
        
        # Dimension 0: Fortify timing
        fortify_timing = workflow[:, 0]  # Shape: [batch_size]
        if timestep < 10:  # Early game
            # High value → boost decoy actions
            for action in self.action_groups['decoy']:
                if action < self.action_dim:
                    biases[:, action] += fortify_timing * 2.0
        else:  # Late game
            # Low value → boost decoy actions
            for action in self.action_groups['decoy']:
                if action < self.action_dim:
                    biases[:, action] += (-fortify_timing) * 2.0
        
        # Dimension 1: Fortify intensity
        fortify_intensity = workflow[:, 1]
        for action in self.action_groups['decoy']:
            if action < self.action_dim:
                biases[:, action] += fortify_intensity * 1.5
        
        # Dimension 2: Analysis frequency
        analysis_freq = workflow[:, 2]
        for action in self.action_groups['analyze']:
            if action < self.action_dim:
                biases[:, action] += analysis_freq * 1.5
        
        # Dimension 3: Remediation preference (-1: remove, +1: restore)
        remediation_pref = workflow[:, 3]
        for action in self.action_groups['remove']:
            if action < self.action_dim:
                biases[:, action] += (-remediation_pref) * 1.5  # Negative pref boosts remove
        for action in self.action_groups['restore']:
            if action < self.action_dim:
                biases[:, action] += remediation_pref * 1.5  # Positive pref boosts restore
        
        # Dimension 4: Response speed
        response_speed = workflow[:, 4]
        # Fast response → reduce sleep actions
        for action in self.action_groups['sleep']:
            if action < self.action_dim:
                biases[:, action] -= response_speed * 1.0
        
        # Dimensions 5-7: Subnet priorities
        user_priority = workflow[:, 5]
        enterprise_priority = workflow[:, 6]
        operational_priority = workflow[:, 7]
        
        # Map actions to subnets and apply priority biases
        user_actions = [11, 12, 13, 14, 24, 25, 26, 27, 141, 142, 143, 144, 1003, 1004, 1005, 1006]
        enterprise_actions = [3, 4, 5, 16, 17, 18, 133, 134, 135, 1000, 1001, 1002]
        operational_actions = [9, 22, 139, 1007, 1008]
        
        for action in user_actions:
            if action < self.action_dim:
                biases[:, action] += user_priority * 1.0
        
        for action in enterprise_actions:
            if action < self.action_dim:
                biases[:, action] += enterprise_priority * 1.0
        
        for action in operational_actions:
            if action < self.action_dim:
                biases[:, action] += operational_priority * 1.0
        
        return logits + biases


# ============================================================================
# WORKFLOW COMPLIANCE CHECKER WITH PREDEFINED SEMANTICS
# ============================================================================

class SemanticComplianceChecker:
    """
    Check if policy follows the predefined workflow semantics
    """
    
    def __init__(self, workflow: np.ndarray, workflow_space: PredefinedWorkflowSpace):
        self.workflow = workflow
        self.workflow_space = workflow_space
        self.action_history = []
        self.compliance_scores = {}
    
    def update(self, action: int, timestep: int):
        """Record action for compliance checking"""
        self.action_history.append((action, timestep))
    
    def compute_compliance(self) -> float:
        """
        Compute compliance with predefined workflow semantics
        """
        if not self.action_history:
            return 0.0
        
        scores = []
        
        # Check dimension 0: Fortify timing
        early_decoys = sum(1 for a, t in self.action_history 
                          if t < 10 and 1000 <= a <= 1008)
        late_decoys = sum(1 for a, t in self.action_history 
                         if t >= 30 and 1000 <= a <= 1008)
        
        if self.workflow[0] > 0.5:  # Should fortify early
            scores.append(min(1.0, early_decoys / 3))
        elif self.workflow[0] < -0.5:  # Should fortify late
            scores.append(min(1.0, late_decoys / 3))
        else:
            scores.append(0.5)  # Neutral
        
        # Check dimension 2: Analysis frequency
        analyze_actions = sum(1 for a, _ in self.action_history 
                            if a in [2, 3, 4, 5, 9, 11, 12, 13, 14])
        total_actions = len(self.action_history)
        analyze_ratio = analyze_actions / (total_actions + 1e-8)
        
        expected_ratio = (self.workflow[2] + 1) / 4  # Map [-1,1] to [0,0.5]
        scores.append(1.0 - abs(analyze_ratio - expected_ratio))
        
        # Check dimension 3: Remediation preference
        remove_actions = sum(1 for a, _ in self.action_history 
                           if a in range(15, 28))
        restore_actions = sum(1 for a, _ in self.action_history 
                            if a in range(132, 145))
        
        if remove_actions + restore_actions > 0:
            actual_restore_ratio = restore_actions / (remove_actions + restore_actions)
            expected_restore_ratio = (self.workflow[3] + 1) / 2
            scores.append(1.0 - abs(actual_restore_ratio - expected_restore_ratio))
        
        # Check subnet priorities (dimensions 5-7)
        user_actions = sum(1 for a, _ in self.action_history 
                          if a in [11, 12, 13, 14, 24, 25, 26, 27, 141, 142, 143, 144])
        enterprise_actions = sum(1 for a, _ in self.action_history 
                               if a in [3, 4, 5, 16, 17, 18, 133, 134, 135])
        operational_actions = sum(1 for a, _ in self.action_history 
                                if a in [9, 22, 139])
        
        total_subnet = user_actions + enterprise_actions + operational_actions + 1e-8
        
        if total_subnet > 0:
            actual_ratios = np.array([user_actions, enterprise_actions, operational_actions]) / total_subnet
            priority_weights = np.array([self.workflow[5], self.workflow[6], self.workflow[7]])
            priority_weights = (priority_weights + 1) / 2  # Map to [0,1]
            priority_weights = priority_weights / (priority_weights.sum() + 1e-8)
            
            subnet_score = 1.0 - np.abs(actual_ratios - priority_weights).mean()
            scores.append(subnet_score)
        
        return np.mean(scores) if scores else 0.0


# ============================================================================
# EXAMPLE: HOW TO USE PREDEFINED WORKFLOWS
# ============================================================================

def demonstrate_predefined_workflows():
    """
    Show how to use predefined workflow semantics
    """
    print("="*70)
    print("PREDEFINED WORKFLOW SPACE DEMONSTRATION")
    print("="*70)
    
    # Initialize workflow space
    workflow_space = PredefinedWorkflowSpace()
    
    # Create canonical workflows
    canonicals = workflow_space.create_canonical_workflows()
    
    print("\n1. CANONICAL WORKFLOWS:")
    print("-" * 40)
    for name, workflow in canonicals.items():
        print(f"\n{name}:")
        interpretation = workflow_space.interpret_workflow(workflow)
        for dim_name, meaning in list(interpretation.items())[:3]:  # Show first 3
            print(f"  - {dim_name}: {meaning}")
    
    print("\n2. WORKFLOW-GUIDED POLICY:")
    print("-" * 40)
    
    # Create policy
    policy = WorkflowGuidedPolicy()
    
    # Test with different workflows
    state = torch.randn(1, 52)
    
    for name, workflow in list(canonicals.items())[:2]:
        workflow_tensor = torch.FloatTensor(workflow).unsqueeze(0)
        action_probs = policy(state, workflow_tensor, timestep=5)
        
        # Get top 3 actions
        top3_probs, top3_actions = torch.topk(action_probs[0], 3)
        
        print(f"\n{name} workflow top actions:")
        for prob, action in zip(top3_probs, top3_actions):
            action_type = "unknown"
            if 1000 <= action <= 1008:
                action_type = "decoy"
            elif action in [2, 3, 4, 5, 9, 11, 12, 13, 14]:
                action_type = "analyze"
            elif action in range(15, 28):
                action_type = "remove"
            elif action in range(132, 145):
                action_type = "restore"
            
            print(f"  Action {action:3d} ({action_type:8s}): {prob:.3f}")
    
    print("\n3. COMPLIANCE CHECKING:")
    print("-" * 40)
    
    # Test compliance
    workflow = canonicals["early_fortification"]
    checker = SemanticComplianceChecker(workflow, workflow_space)
    
    # Simulate some actions
    for t in range(20):
        if t < 5:
            action = np.random.choice([1000, 1001, 1002])  # Early decoys
        else:
            action = np.random.choice([3, 4, 5])  # Analyze
        checker.update(action, t)
    
    compliance = checker.compute_compliance()
    print(f"Compliance with 'early_fortification' workflow: {compliance:.2f}")


def explain_predefined_approach():
    """
    Explain the advantages of predefined semantics
    """
    explanation = """
    
    ADVANTAGES OF PREDEFINED WORKFLOW SEMANTICS:
    ============================================
    
    1. INTERPRETABILITY:
       - Each dimension has clear meaning
       - Can explain why agent behaves certain way
       - Easy to debug and tune
    
    2. FASTER LEARNING:
       - Network doesn't discover semantics from scratch
       - Biases guide exploration in meaningful directions
       - Reduces sample complexity
    
    3. DOMAIN KNOWLEDGE:
       - Incorporates expert knowledge about defense strategies
       - Dimensions based on analysis of optimal policies
       - Covers known important strategic choices
    
    4. CONTROLLABILITY:
       - Can manually create workflows for specific scenarios
       - Easy to test specific strategies
       - Can interpolate between known good strategies
    
    5. GP-UCB BENEFITS:
       - Distance metric is meaningful from the start
       - Similar workflows will behave similarly
       - Search space has semantic structure
    
    HOW IT WORKS IN PRACTICE:
    =========================
    
    1. Define workflow dimensions based on domain knowledge
    2. Create canonical workflows representing known strategies
    3. Sample workflows from this structured space
    4. Use workflow to bias action selection (not hard constraints)
    5. Check compliance to ensure workflow is being followed
    6. GP-UCB searches this semantically meaningful space
    
    The key: Workflows have MEANING from day 1, not discovered over time!
    """
    print(explanation)


if __name__ == "__main__":
    # Demonstrate the system
    demonstrate_predefined_workflows()
    
    # Explain the approach
    explain_predefined_approach()
