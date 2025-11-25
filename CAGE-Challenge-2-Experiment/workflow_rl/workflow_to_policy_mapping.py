#!/usr/bin/env python3
"""
Demonstration of how workflow embeddings translate to policies
Shows different mechanisms for workflow → action mapping
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# METHOD 1: CONCATENATION - Workflow as additional input features
# ============================================================================

class ConcatenationMethod(nn.Module):
    """
    Simplest approach: Concatenate workflow embedding with state
    The neural network learns to use workflow features to bias decisions
    """
    
    def __init__(self, state_dim=52, workflow_dim=8, action_dim=27):
        super().__init__()
        
        # Combine state and workflow
        combined_dim = state_dim + workflow_dim  # 52 + 8 = 60
        
        # Actor network sees both state and workflow
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state, workflow):
        # Simply concatenate
        combined = torch.cat([state, workflow], dim=-1)
        action_probs = self.actor(combined)
        return action_probs
    
    def get_action(self, state, workflow):
        """
        The network learns that:
        - High workflow[0] (early fortify) → increase P(decoy actions) early
        - High workflow[3] (restore pref) → increase P(restore) vs P(remove)
        - etc.
        """
        action_probs = self.forward(state, workflow)
        action = torch.multinomial(action_probs, 1)
        return action.item()


# ============================================================================
# METHOD 2: MODULATION - Workflow modulates action probabilities
# ============================================================================

class ModulationMethod(nn.Module):
    """
    Workflow embedding modulates the base policy's action distribution
    More explicit control over how workflow affects actions
    """
    
    def __init__(self, state_dim=52, workflow_dim=8, action_dim=27):
        super().__init__()
        
        # Base policy (state → actions)
        self.base_policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Workflow → modulation weights
        self.modulator = nn.Sequential(
            nn.Linear(workflow_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()  # Output 0-2 multipliers
        )
        
    def forward(self, state, workflow):
        # Get base action logits
        base_logits = self.base_policy(state)
        
        # Get modulation from workflow
        modulation = self.modulator(workflow) * 2  # Scale to [0, 2]
        
        # Modulate the logits
        modulated_logits = base_logits * modulation
        
        # Convert to probabilities
        action_probs = torch.softmax(modulated_logits, dim=-1)
        return action_probs
    
    def get_action(self, state, workflow):
        """
        Workflow directly scales action probabilities:
        - Fortify workflow → amplifies decoy action probabilities
        - Restore workflow → amplifies restore action probabilities
        """
        action_probs = self.forward(state, workflow)
        action = torch.multinomial(action_probs, 1)
        return action.item()


# ============================================================================
# METHOD 3: HIERARCHICAL - Workflow selects sub-policies
# ============================================================================

class HierarchicalMethod(nn.Module):
    """
    Workflow embedding selects between specialized sub-policies
    Most structured approach with clear separation of concerns
    """
    
    def __init__(self, state_dim=52, workflow_dim=8, action_dim=27):
        super().__init__()
        
        # Multiple sub-policies for different strategies
        self.fortify_policy = nn.Linear(state_dim, action_dim)
        self.analyze_policy = nn.Linear(state_dim, action_dim)
        self.remediate_policy = nn.Linear(state_dim, action_dim)
        
        # Workflow determines mixing weights
        self.strategy_selector = nn.Sequential(
            nn.Linear(workflow_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 3 sub-policies
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state, workflow):
        # Get outputs from each sub-policy
        fortify_logits = self.fortify_policy(state)
        analyze_logits = self.analyze_policy(state)
        remediate_logits = self.remediate_policy(state)
        
        # Stack sub-policy outputs
        all_logits = torch.stack([fortify_logits, analyze_logits, remediate_logits], dim=1)
        
        # Get mixing weights from workflow
        weights = self.strategy_selector(workflow)  # [batch, 3]
        
        # Mix sub-policies based on workflow
        mixed_logits = torch.sum(all_logits * weights.unsqueeze(-1), dim=1)
        
        action_probs = torch.softmax(mixed_logits, dim=-1)
        return action_probs
    
    def get_action(self, state, workflow):
        """
        Workflow selects which sub-policy to use:
        - Early fortify workflow → fortify_policy dominates
        - High analysis workflow → analyze_policy dominates
        - etc.
        """
        action_probs = self.forward(state, workflow)
        action = torch.multinomial(action_probs, 1)
        return action.item()


# ============================================================================
# METHOD 4: RULE-AUGMENTED - Workflow defines soft rules
# ============================================================================

class RuleAugmentedMethod:
    """
    Workflow embedding is interpreted as soft rules that bias action selection
    Combines learned policy with interpretable workflow rules
    """
    
    def __init__(self, base_policy, action_space):
        self.base_policy = base_policy
        self.action_space = action_space
        self.workflow_params = None
        
    def set_workflow(self, workflow_embedding):
        """Decode workflow into interpretable parameters"""
        self.workflow_params = {
            'fortify_early': workflow_embedding[0] > 0.5,
            'fortify_prob': workflow_embedding[1],
            'analyze_prob': workflow_embedding[2],
            'prefer_restore': workflow_embedding[3] > 0,
            'immediate': workflow_embedding[4] > 0.5,
            'subnet_weights': workflow_embedding[5:8]
        }
        
    def get_action(self, state, timestep):
        """Apply workflow rules to bias action selection"""
        
        # Rule 1: Early fortification
        if timestep < 10 and self.workflow_params['fortify_early']:
            if np.random.random() < self.workflow_params['fortify_prob']:
                return self._select_decoy_action()
        
        # Rule 2: Information gathering
        if np.random.random() < self.workflow_params['analyze_prob'] * 0.3:
            return self._select_analyze_action()
        
        # Rule 3: Threat response
        if self._detect_threat(state):
            if self.workflow_params['immediate']:
                if self.workflow_params['prefer_restore']:
                    return self._select_restore_action()
                else:
                    return self._select_remove_action()
        
        # Default: Use base policy
        return self.base_policy.get_action(state)
    
    def _select_decoy_action(self):
        """Select decoy based on subnet weights"""
        subnet_probs = self.workflow_params['subnet_weights'] / np.sum(self.workflow_params['subnet_weights'])
        subnet = np.random.choice(['user', 'enterprise', 'operational'], p=subnet_probs)
        
        decoy_actions = {
            'user': [1003, 1004, 1005, 1006],
            'enterprise': [1000, 1001, 1002],
            'operational': [1008]
        }
        return np.random.choice(decoy_actions[subnet])
    
    def _select_analyze_action(self):
        """Select analyze action based on subnet focus"""
        subnet_probs = self.workflow_params['subnet_weights'] / np.sum(self.workflow_params['subnet_weights'])
        subnet = np.random.choice(['user', 'enterprise', 'operational'], p=subnet_probs)
        
        analyze_actions = {
            'user': [11, 12, 13, 14],
            'enterprise': [3, 4, 5],
            'operational': [9]
        }
        return np.random.choice(analyze_actions[subnet])
    
    def _detect_threat(self, state):
        """Simple threat detection from state"""
        # Check activity indicators in state vector
        for i in range(0, min(len(state), 52), 4):
            if state[i] > 0:  # Activity detected
                return True
        return False
    
    def _select_restore_action(self):
        """Select restore action"""
        return np.random.choice([133, 134, 135, 141, 142, 143, 144])
    
    def _select_remove_action(self):
        """Select remove action"""
        return np.random.choice([16, 17, 18, 24, 25, 26, 27])


# ============================================================================
# DEMONSTRATION: How workflow affects action distribution
# ============================================================================

def demonstrate_workflow_effect():
    """Show how different workflow embeddings affect action selection"""
    
    print("="*80)
    print("WORKFLOW → POLICY MAPPING DEMONSTRATION")
    print("="*80)
    
    # Create example state and workflows
    state = torch.randn(1, 52).to(device)
    
    workflows = {
        "Reactive": torch.tensor([[0.2, 0.1, 0.2, -0.5, 0.8, 0.6, 0.3, 0.1]]).float(),
        "Proactive": torch.tensor([[0.9, 0.4, 0.5, 0.5, 0.9, 0.3, 0.4, 0.3]]).float(),
        "Balanced": torch.tensor([[0.5, 0.3, 0.4, 0.0, 0.7, 0.4, 0.4, 0.2]]).float()
    }
    
    # Test concatenation method
    print("\n1. CONCATENATION METHOD")
    print("-" * 40)
    model = ConcatenationMethod()
    
    for name, workflow in workflows.items():
        probs = model.forward(state, workflow)
        top5 = torch.topk(probs[0], 5)
        print(f"\n{name} workflow top 5 actions:")
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"  Action {idx.item():2d}: {prob.item():.3f}")
    
    # Test modulation method
    print("\n2. MODULATION METHOD")
    print("-" * 40)
    model = ModulationMethod()
    
    for name, workflow in workflows.items():
        probs = model.forward(state, workflow)
        top5 = torch.topk(probs[0], 5)
        print(f"\n{name} workflow top 5 actions:")
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"  Action {idx.item():2d}: {prob.item():.3f}")
    
    # Test hierarchical method
    print("\n3. HIERARCHICAL METHOD")
    print("-" * 40)
    model = HierarchicalMethod()
    
    for name, workflow in workflows.items():
        probs = model.forward(state, workflow)
        
        # Also show sub-policy weights
        weights = model.strategy_selector(workflow)
        print(f"\n{name} workflow:")
        print(f"  Sub-policy weights: Fortify={weights[0,0]:.2f}, "
              f"Analyze={weights[0,1]:.2f}, Remediate={weights[0,2]:.2f}")
        
        top5 = torch.topk(probs[0], 5)
        print(f"  Top 5 actions:")
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"    Action {idx.item():2d}: {prob.item():.3f}")


# ============================================================================
# KEY INSIGHTS
# ============================================================================

def explain_mapping():
    """Explain the key concepts"""
    
    print("\n" + "="*80)
    print("KEY INSIGHTS: WORKFLOW → POLICY MAPPING")
    print("="*80)
    
    insights = """
    1. WORKFLOW AS CONTEXT:
       - The workflow embedding provides strategic context to the policy
       - It doesn't directly specify actions, but biases action selection
       - The policy still responds to state, but through a workflow "lens"
    
    2. LEARNING THE MAPPING:
       - During training, the network learns associations:
         * High dim[0] (early fortify) → Higher P(decoy actions) when t < 10
         * High dim[3] (restore pref) → Higher P(restore) when compromise detected
         * High dim[2] (analyze freq) → Higher P(analyze) throughout episode
       - These associations emerge from reward signals during PPO training
    
    3. ADVANTAGES OF INDIRECTION:
       - Workflow provides consistent strategy across episode
       - Policy can still adapt to specific situations
       - Enables transfer: Same workflow works across different scenarios
       - Interpretable: Can understand why agent behaves certain way
    
    4. TRAINING PROCESS:
       Step 1: Sample workflow w from search space
       Step 2: Create policy π(a|s,w) conditioned on workflow
       Step 3: Run episodes with this workflow-conditioned policy
       Step 4: Update policy with PPO using augmented states [s, w]
       Step 5: Evaluate workflow performance
       Step 6: Update workflow search based on performance
    
    5. WHY IT WORKS:
       - Reduces exploration space: 8D workflow vs 100+ step action sequence
       - Provides inductive bias: Workflow structure encodes domain knowledge
       - Enables meta-learning: Learn which strategies work against which opponents
       - Maintains flexibility: Policy can override workflow in critical situations
    """
    
    print(insights)


if __name__ == "__main__":
    # Demonstrate different mapping methods
    demonstrate_workflow_effect()
    
    # Explain the concepts
    explain_mapping()
