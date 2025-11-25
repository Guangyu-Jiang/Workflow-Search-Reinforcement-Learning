#!/usr/bin/env python3
"""
Pure Order-Based Workflow for CAGE2
Focus solely on the priority order of fixing hosts
9 distinct action groups = 9! = 362,880 possible orderings
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from itertools import permutations
import random
from scipy.stats import kendalltau, spearmanr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PURE ORDER WORKFLOW REPRESENTATION
# ============================================================================

class PureOrderWorkflow:
    """
    Workflow as a pure priority ordering of the 9 distinct fixable units
    """
    
    def __init__(self):
        # The 9 distinct fixable units (based on action groups)
        self.units = [
            'Defender',
            'Enterprise0',
            'Enterprise1', 
            'Enterprise2',
            'Op_Server0',  # Represents Op_Server0 + Op_Hosts
            'User0',
            'User1',
            'User2',
            'User3'  # Represents User3 + User4
        ]
        
        # Map units to action IDs
        self.unit_actions = {
            'Defender': {'analyze': 2, 'remove': 15, 'restore': 132},
            'Enterprise0': {'analyze': 3, 'remove': 16, 'restore': 133},
            'Enterprise1': {'analyze': 4, 'remove': 17, 'restore': 134},
            'Enterprise2': {'analyze': 5, 'remove': 18, 'restore': 135},
            'Op_Server0': {'analyze': 9, 'remove': 22, 'restore': 139},  # Also for Op_Hosts
            'User0': {'analyze': 11, 'remove': 24, 'restore': 141},
            'User1': {'analyze': 12, 'remove': 25, 'restore': 142},
            'User2': {'analyze': 13, 'remove': 26, 'restore': 143},
            'User3': {'analyze': 14, 'remove': 27, 'restore': 144}  # Also for User4
        }
        
        # Map observation indices to units (for detecting compromises)
        # Based on BlueTableWrapper order
        self.obs_idx_to_unit = {
            0: 'Defender',
            1: 'Enterprise0',
            2: 'Enterprise1',
            3: 'Enterprise2',
            4: 'Op_Server0',  # Op_Host0 maps to Op_Server0 group
            5: 'Op_Server0',  # Op_Host1 maps to Op_Server0 group
            6: 'Op_Server0',  # Op_Host2 maps to Op_Server0 group
            7: 'Op_Server0',  # Op_Server0
            8: 'User0',
            9: 'User1',
            10: 'User2',
            11: 'User3',  # User3
            12: 'User3'   # User4 maps to User3 group
        }
        
        # Canonical orderings
        self.canonical_orders = self._create_canonical_orders()
    
    def _create_canonical_orders(self) -> Dict[str, List[str]]:
        """Create meaningful canonical orderings"""
        
        orders = {
            # Critical infrastructure first
            'critical_first': [
                'Defender',
                'Op_Server0',
                'Enterprise0',
                'Enterprise1',
                'Enterprise2',
                'User0',
                'User1',
                'User2',
                'User3'
            ],
            
            # User experience first
            'user_first': [
                'User0',
                'User1',
                'User2',
                'User3',
                'Defender',
                'Enterprise0',
                'Enterprise1',
                'Enterprise2',
                'Op_Server0'
            ],
            
            # Enterprise focus
            'enterprise_first': [
                'Enterprise0',
                'Enterprise1',
                'Enterprise2',
                'Defender',
                'Op_Server0',
                'User0',
                'User1',
                'User2',
                'User3'
            ],
            
            # Balanced (based on typical compromise patterns)
            'balanced': [
                'Defender',
                'User0',  # Often compromised first
                'Enterprise0',
                'Op_Server0',
                'User1',
                'Enterprise1',
                'User2',
                'Enterprise2',
                'User3'
            ],
            
            # Reverse (least critical first)
            'reverse': [
                'User3',
                'User2',
                'User1',
                'User0',
                'Enterprise2',
                'Enterprise1',
                'Enterprise0',
                'Op_Server0',
                'Defender'
            ]
        }
        
        return orders
    
    def order_to_vector(self, order: List[str]) -> np.ndarray:
        """
        Convert order to 8-dimensional continuous vector
        Uses position encoding and relative rankings
        """
        vector = np.zeros(8)
        
        # Method 1: Position-based encoding (first 4 dims)
        # Encode relative positions of key units
        defender_pos = order.index('Defender') / 8 if 'Defender' in order else 0.5
        opserver_pos = order.index('Op_Server0') / 8 if 'Op_Server0' in order else 0.5
        
        # Average position of enterprise hosts
        ent_positions = []
        for e in ['Enterprise0', 'Enterprise1', 'Enterprise2']:
            if e in order:
                ent_positions.append(order.index(e))
        avg_ent_pos = np.mean(ent_positions) / 8 if ent_positions else 0.5
        
        # Average position of user hosts  
        user_positions = []
        for u in ['User0', 'User1', 'User2', 'User3']:
            if u in order:
                user_positions.append(order.index(u))
        avg_user_pos = np.mean(user_positions) / 8 if user_positions else 0.5
        
        vector[0] = defender_pos
        vector[1] = opserver_pos
        vector[2] = avg_ent_pos
        vector[3] = avg_user_pos
        
        # Method 2: Relative rankings (next 4 dims)
        # Encode whether critical units come before less critical ones
        vector[4] = 1.0 if defender_pos < avg_user_pos else -1.0  # Defender before users?
        vector[5] = 1.0 if opserver_pos < avg_ent_pos else -1.0  # OpServer before enterprise?
        vector[6] = 1.0 if avg_ent_pos < avg_user_pos else -1.0  # Enterprise before users?
        
        # Clustering metric - are similar units grouped?
        vector[7] = self._compute_clustering(order)
        
        return vector
    
    def _compute_clustering(self, order: List[str]) -> float:
        """
        Measure how clustered similar units are (0 to 1)
        """
        score = 0.0
        
        # Check enterprise clustering
        ent_units = ['Enterprise0', 'Enterprise1', 'Enterprise2']
        ent_positions = [order.index(e) for e in ent_units if e in order]
        if len(ent_positions) > 1:
            ent_spread = max(ent_positions) - min(ent_positions)
            ent_clustering = 1.0 - (ent_spread - len(ent_positions) + 1) / 6
            score += ent_clustering * 0.5
        
        # Check user clustering
        user_units = ['User0', 'User1', 'User2', 'User3']
        user_positions = [order.index(u) for u in user_units if u in order]
        if len(user_positions) > 1:
            user_spread = max(user_positions) - min(user_positions)
            user_clustering = 1.0 - (user_spread - len(user_positions) + 1) / 6
            score += user_clustering * 0.5
        
        return score
    
    def vector_to_order(self, vector: np.ndarray) -> List[str]:
        """
        Convert 8D vector back to an order (approximate)
        Find closest canonical order or generate based on features
        """
        # Find closest canonical order
        min_dist = float('inf')
        best_order = self.canonical_orders['balanced']
        
        for name, order in self.canonical_orders.items():
            order_vec = self.order_to_vector(order)
            dist = np.linalg.norm(vector - order_vec)
            if dist < min_dist:
                min_dist = dist
                best_order = order
        
        # Could also generate new order based on vector features
        # but for now return closest canonical
        return best_order.copy()
    
    def sample_order(self, method: str = 'random') -> List[str]:
        """Sample a priority order"""
        
        if method == 'random':
            order = self.units.copy()
            random.shuffle(order)
            return order
        
        elif method == 'canonical':
            name = random.choice(list(self.canonical_orders.keys()))
            return self.canonical_orders[name].copy()
        
        elif method == 'swap':
            # Start from canonical and swap a few positions
            base = random.choice(list(self.canonical_orders.values()))
            order = base.copy()
            
            # Random swaps
            for _ in range(random.randint(1, 3)):
                i, j = random.sample(range(len(order)), 2)
                order[i], order[j] = order[j], order[i]
            
            return order
        
        elif method == 'weighted':
            # Sample based on importance weights
            weights = {
                'Defender': 10,
                'Op_Server0': 8,
                'Enterprise0': 6,
                'Enterprise1': 6,
                'Enterprise2': 6,
                'User0': 3,
                'User1': 3,
                'User2': 3,
                'User3': 3
            }
            
            # Weighted sampling without replacement
            order = []
            remaining = self.units.copy()
            
            while remaining:
                probs = [weights[u] for u in remaining]
                probs = np.array(probs) / sum(probs)
                idx = np.random.choice(len(remaining), p=probs)
                order.append(remaining.pop(idx))
            
            return order
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")


# ============================================================================
# ORDER-BASED POLICY
# ============================================================================

class OrderBasedPolicy:
    """
    Policy that follows a strict priority order for fixing hosts
    """
    
    def __init__(self, order: List[str], workflow: PureOrderWorkflow):
        self.order = order
        self.workflow = workflow
        self.priority = {unit: i for i, unit in enumerate(order)}
        
    def get_action(self, observation: np.ndarray, 
                   fix_method: str = 'restore') -> int:
        """
        Get action based on priority order and current compromises
        
        Args:
            observation: 52-dim observation vector
            fix_method: 'analyze', 'remove', or 'restore'
        """
        
        # Detect compromised units from observation
        compromised = self._detect_compromised(observation)
        
        if not compromised:
            return 0  # Sleep if nothing compromised
        
        # Find highest priority compromised unit
        highest_priority = min(compromised, key=lambda u: self.priority[u])
        
        # Get action for that unit
        if highest_priority in self.workflow.unit_actions:
            actions = self.workflow.unit_actions[highest_priority]
            return actions.get(fix_method, 0)
        
        return 0
    
    def _detect_compromised(self, obs: np.ndarray) -> Set[str]:
        """
        Detect which units are compromised from observation
        Each host has 4 features, compromise is at index 2
        """
        compromised = set()
        
        for i in range(0, min(52, len(obs)), 4):
            host_idx = i // 4
            compromise_flag = obs[i + 2] if i + 2 < len(obs) else 0
            
            if compromise_flag > 0.1:  # Threshold for compromise
                if host_idx in self.workflow.obs_idx_to_unit:
                    unit = self.workflow.obs_idx_to_unit[host_idx]
                    compromised.add(unit)
        
        return compromised


# ============================================================================
# ORDER COMPLIANCE CHECKING
# ============================================================================

class OrderComplianceChecker:
    """
    Check if agent follows the specified priority order
    """
    
    def __init__(self, order: List[str], workflow: PureOrderWorkflow):
        self.order = order
        self.workflow = workflow
        self.priority = {unit: i for i, unit in enumerate(order)}
        self.fix_history = []
        
    def record_fix(self, action: int, true_compromised: Set[str], timestep: int):
        """
        Record a fix action and the true compromised state
        
        Args:
            action: Action ID taken
            true_compromised: Set of truly compromised units
            timestep: Current timestep
        """
        
        # Map action to unit
        fixed_unit = None
        for unit, actions in self.workflow.unit_actions.items():
            if action in actions.values():
                fixed_unit = unit
                break
        
        if fixed_unit and true_compromised:
            # What should have been fixed based on order?
            should_fix = min(true_compromised, key=lambda u: self.priority.get(u, 999))
            
            self.fix_history.append({
                'timestep': timestep,
                'fixed': fixed_unit,
                'should_fix': should_fix,
                'compromised': true_compromised.copy(),
                'correct': fixed_unit == should_fix
            })
    
    def compute_compliance(self) -> Dict[str, float]:
        """
        Compute compliance metrics
        """
        
        if not self.fix_history:
            return {'overall': 0.0}
        
        # Metric 1: Exact match rate
        exact_matches = sum(1 for f in self.fix_history if f['correct'])
        exact_rate = exact_matches / len(self.fix_history)
        
        # Metric 2: Priority difference (how far off were we?)
        priority_diffs = []
        for f in self.fix_history:
            if f['fixed'] in self.priority and f['should_fix'] in self.priority:
                diff = abs(self.priority[f['fixed']] - self.priority[f['should_fix']])
                priority_diffs.append(1.0 - min(diff / 8, 1.0))  # Normalize
        
        avg_priority_score = np.mean(priority_diffs) if priority_diffs else 0.0
        
        # Metric 3: Order preservation (Kendall's tau)
        if len(self.fix_history) > 1:
            actual_sequence = [f['fixed'] for f in self.fix_history]
            expected_sequence = [f['should_fix'] for f in self.fix_history]
            
            # Convert to ranks
            actual_ranks = [self.priority.get(u, 9) for u in actual_sequence]
            expected_ranks = [self.priority.get(u, 9) for u in expected_sequence]
            
            if len(set(actual_ranks)) > 1 and len(set(expected_ranks)) > 1:
                tau, _ = kendalltau(actual_ranks, expected_ranks)
                order_score = (tau + 1) / 2  # Normalize to [0, 1]
            else:
                order_score = exact_rate
        else:
            order_score = exact_rate
        
        return {
            'exact_match_rate': exact_rate,
            'priority_score': avg_priority_score,
            'order_preservation': order_score,
            'overall': (exact_rate + avg_priority_score + order_score) / 3
        }


# ============================================================================
# GP-UCB SEARCH FOR ORDERS
# ============================================================================

class OrderWorkflowSearch:
    """
    Search over the space of priority orders using GP-UCB
    """
    
    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        self.workflow = PureOrderWorkflow()
        
        # For GP-UCB
        self.observed_orders = []
        self.observed_vectors = []
        self.observed_rewards = []
        
        # Start with canonical orders
        for name, order in self.workflow.canonical_orders.items():
            self.observed_orders.append(order)
            self.observed_vectors.append(self.workflow.order_to_vector(order))
            self.observed_rewards.append(0.0)  # Will be updated
    
    def select_next_order(self) -> Tuple[List[str], np.ndarray]:
        """
        Select next order to evaluate
        Returns both the order and its vector representation
        """
        
        if len(self.observed_rewards) < len(self.observed_orders):
            # Return unevaluated canonical order
            idx = len(self.observed_rewards) - 1
            return self.observed_orders[idx], self.observed_vectors[idx]
        
        # For now, use epsilon-greedy strategy
        if random.random() < 0.2:  # Explore
            order = self.workflow.sample_order('swap')
        else:  # Exploit
            # Return best so far with small perturbation
            best_idx = np.argmax(self.observed_rewards)
            order = self.observed_orders[best_idx].copy()
            
            # Small perturbation
            if random.random() < 0.5:
                i, j = random.sample(range(len(order)), 2)
                order[i], order[j] = order[j], order[i]
        
        vector = self.workflow.order_to_vector(order)
        return order, vector
    
    def update(self, order: List[str], vector: np.ndarray, 
               reward: float, compliance: float):
        """
        Update search with evaluation results
        """
        
        # Check if order already observed
        for i, obs_order in enumerate(self.observed_orders):
            if obs_order == order:
                # Update existing
                self.observed_rewards[i] = (self.observed_rewards[i] + reward) / 2
                return
        
        # Add new observation
        self.observed_orders.append(order)
        self.observed_vectors.append(vector)
        self.observed_rewards.append(reward * compliance)  # Weight by compliance
    
    def get_best_order(self) -> Tuple[List[str], float]:
        """Get best order found so far"""
        
        if not self.observed_rewards:
            return self.workflow.canonical_orders['balanced'], 0.0
        
        best_idx = np.argmax(self.observed_rewards)
        return self.observed_orders[best_idx], self.observed_rewards[best_idx]


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_pure_order_workflow():
    """Demonstrate the pure order workflow system"""
    
    print("="*70)
    print("PURE ORDER-BASED WORKFLOW SYSTEM")
    print("="*70)
    
    # Initialize workflow
    workflow = PureOrderWorkflow()
    
    # Show canonical orders
    print("\n1. CANONICAL PRIORITY ORDERS:")
    print("-" * 40)
    
    for name, order in workflow.canonical_orders.items():
        vector = workflow.order_to_vector(order)
        print(f"\n{name}:")
        print(f"  Order: {' → '.join(order[:5])}...")
        print(f"  8D Vector: [{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}, {vector[3]:.2f}, ...]")
    
    # Test policy execution
    print("\n2. ORDER-BASED POLICY EXECUTION:")
    print("-" * 40)
    
    order = workflow.canonical_orders['critical_first']
    policy = OrderBasedPolicy(order, workflow)
    
    # Simulate observation with compromises
    obs = np.zeros(52)
    # Set User0 as compromised (index 8, feature 2)
    obs[8*4 + 2] = 1.0
    # Set Op_Server0 as compromised (index 7, feature 2)  
    obs[7*4 + 2] = 1.0
    
    action = policy.get_action(obs, 'restore')
    print(f"\nOrder: critical_first")
    print(f"Compromised: User0, Op_Server0")
    print(f"Action taken: {action} (should fix Op_Server0 first)")
    
    # Test compliance
    print("\n3. COMPLIANCE CHECKING:")
    print("-" * 40)
    
    checker = OrderComplianceChecker(order, workflow)
    
    # Simulate some fixes
    checker.record_fix(139, {'Op_Server0', 'User0'}, 0)  # Correct
    checker.record_fix(141, {'User0', 'Enterprise0'}, 1)  # Should fix Enterprise0
    checker.record_fix(133, {'Enterprise0'}, 2)  # Correct
    
    compliance = checker.compute_compliance()
    print("\nCompliance metrics:")
    for metric, value in compliance.items():
        print(f"  {metric}: {value:.2f}")
    
    # Test search
    print("\n4. ORDER SEARCH:")
    print("-" * 40)
    
    searcher = OrderWorkflowSearch()
    
    # Get next order to try
    next_order, next_vector = searcher.select_next_order()
    print(f"\nNext order to evaluate: {' → '.join(next_order[:5])}...")
    
    # Update with fake results
    searcher.update(next_order, next_vector, reward=85.0, compliance=0.9)
    
    best_order, best_reward = searcher.get_best_order()
    print(f"\nBest order so far: {' → '.join(best_order[:5])}...")
    print(f"Best reward: {best_reward:.1f}")


if __name__ == "__main__":
    demonstrate_pure_order_workflow()
