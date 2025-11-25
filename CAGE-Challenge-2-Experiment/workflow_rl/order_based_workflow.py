#!/usr/bin/env python3
"""
Order-Based Workflow Representation for CAGE2
Workflows are represented purely as permutations of 5 unit types
"""

import numpy as np
from typing import List, Tuple, Dict
from itertools import permutations
import random


class OrderBasedWorkflow:
    """
    Represents workflows as priority orderings of unit types
    No embedding - just the order itself
    """
    
    def __init__(self):
        # The 5 unit types in CAGE2
        self.unit_types = ['defender', 'enterprise', 'op_server', 'op_host', 'user']
        
        # Map hosts to their types
        self.host_to_type = {
            'Defender': 'defender',
            'Enterprise0': 'enterprise',
            'Enterprise1': 'enterprise', 
            'Enterprise2': 'enterprise',
            'Op_Server0': 'op_server',
            'Op_Host0': 'op_host',
            'Op_Host1': 'op_host',
            'Op_Host2': 'op_host',
            'User0': 'user',
            'User1': 'user',
            'User2': 'user',
            'User3': 'user',
            'User4': 'user'
        }
        
        # All possible type orderings (5! = 120)
        self.all_orderings = list(permutations(self.unit_types))
        
        # Create index mapping for fast lookup
        self.order_to_index = {order: i for i, order in enumerate(self.all_orderings)}
        self.index_to_order = {i: order for i, order in enumerate(self.all_orderings)}
    
    def order_to_onehot(self, order: List[str]) -> np.ndarray:
        """
        Convert order to one-hot encoding for neural network input
        Returns a 25-dimensional vector (5 positions × 5 types)
        
        Example: [defender, enterprise, op_server, op_host, user]
        Position 0: defender -> [1,0,0,0,0]
        Position 1: enterprise -> [0,1,0,0,0]
        etc.
        Flattened: [1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1]
        """
        onehot = np.zeros(25)  # 5 positions × 5 types
        
        for position, unit_type in enumerate(order):
            type_index = self.unit_types.index(unit_type)
            onehot[position * 5 + type_index] = 1.0
        
        return onehot
    
    def onehot_to_order(self, onehot: np.ndarray) -> List[str]:
        """
        Convert one-hot encoding back to order
        """
        order = []
        onehot = onehot.reshape(5, 5)  # Reshape to [positions, types]
        
        for position in range(5):
            type_index = np.argmax(onehot[position])
            order.append(self.unit_types[type_index])
        
        return order
    
    def order_to_ranking(self, order: List[str]) -> np.ndarray:
        """
        Convert order to ranking vector
        Returns a 5-dimensional vector where each element is the priority rank of that unit type
        
        Example: [defender, enterprise, op_server, op_host, user]
        Returns: [0, 1, 2, 3, 4] (defender=0, enterprise=1, etc.)
        """
        ranking = np.zeros(5)
        for rank, unit_type in enumerate(order):
            type_index = self.unit_types.index(unit_type)
            ranking[type_index] = rank
        return ranking
    
    def ranking_to_order(self, ranking: np.ndarray) -> List[str]:
        """
        Convert ranking vector back to order
        """
        # Create pairs of (unit_type, rank)
        pairs = [(self.unit_types[i], ranking[i]) for i in range(5)]
        # Sort by rank
        pairs.sort(key=lambda x: x[1])
        # Return ordered unit types
        return [unit_type for unit_type, _ in pairs]
    
    def sample_order(self, method: str = 'uniform') -> List[str]:
        """
        Sample a unit type ordering
        
        Methods:
        - uniform: Random permutation
        - critical_first: Prioritize critical assets
        - user_last: Deprioritize users
        - balanced: Mix of strategies
        """
        
        if method == 'uniform':
            return list(random.choice(self.all_orderings))
        
        elif method == 'critical_first':
            # Start with critical units
            critical = ['defender', 'op_server']
            others = ['enterprise', 'op_host', 'user']
            random.shuffle(critical)
            random.shuffle(others)
            return critical + others
        
        elif method == 'user_last':
            # Put users last
            non_users = ['defender', 'enterprise', 'op_server', 'op_host']
            random.shuffle(non_users)
            return non_users + ['user']
        
        elif method == 'balanced':
            # Some predefined balanced orders
            balanced_orders = [
                ['defender', 'op_server', 'enterprise', 'op_host', 'user'],
                ['defender', 'enterprise', 'op_server', 'user', 'op_host'],
                ['op_server', 'defender', 'enterprise', 'op_host', 'user'],
                ['enterprise', 'defender', 'op_server', 'op_host', 'user']
            ]
            return random.choice(balanced_orders)
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def get_canonical_workflows(self) -> Dict[str, List[str]]:
        """
        Return canonical workflow strategies for initialization
        """
        return {
            'critical_first': ['defender', 'op_server', 'enterprise', 'op_host', 'user'],
            'enterprise_focus': ['enterprise', 'defender', 'op_server', 'op_host', 'user'],
            'user_priority': ['user', 'defender', 'enterprise', 'op_server', 'op_host'],
            'operational_focus': ['op_server', 'op_host', 'defender', 'enterprise', 'user'],
            'balanced': ['defender', 'enterprise', 'op_server', 'user', 'op_host'],
            'reverse': ['user', 'op_host', 'enterprise', 'op_server', 'defender']
        }
    
    def compute_kendall_distance(self, order1: List[str], order2: List[str]) -> float:
        """
        Compute Kendall tau distance between two orderings
        Counts the number of pairwise disagreements, normalized to [0, 1]
        
        This is used as the distance metric for the GP kernel
        """
        
        # Get positions of each type in both orders
        pos1 = {t: order1.index(t) for t in order1}
        pos2 = {t: order2.index(t) for t in order2}
        
        # Count pairwise disagreements
        disagreements = 0
        for i, type1 in enumerate(self.unit_types):
            for j, type2 in enumerate(self.unit_types):
                if i < j:  # Only check each pair once
                    # Check if relative order is different
                    in_order1 = pos1[type1] < pos1[type2]
                    in_order2 = pos2[type1] < pos2[type2]
                    if in_order1 != in_order2:
                        disagreements += 1
        
        # Normalize to [0, 1]
        max_disagreements = len(self.unit_types) * (len(self.unit_types) - 1) / 2
        return disagreements / max_disagreements
    
    def compute_spearman_distance(self, order1: List[str], order2: List[str]) -> float:
        """
        Compute Spearman footrule distance between two orderings
        Sum of absolute differences in positions, normalized to [0, 1]
        
        Alternative distance metric for the GP kernel
        """
        
        # Get positions of each type in both orders
        pos1 = {t: order1.index(t) for t in order1}
        pos2 = {t: order2.index(t) for t in order2}
        
        # Sum absolute differences
        total_distance = sum(abs(pos1[t] - pos2[t]) for t in self.unit_types)
        
        # Maximum possible distance (reverse order)
        max_distance = len(self.unit_types) ** 2 // 2
        
        return total_distance / max_distance
    
    def get_order_index(self, order: List[str]) -> int:
        """
        Get the index of an order in the full permutation space (0-119)
        """
        order_tuple = tuple(order)
        return self.order_to_index.get(order_tuple, -1)
    
    def get_order_from_index(self, index: int) -> List[str]:
        """
        Get the order corresponding to an index (0-119)
        """
        return list(self.index_to_order.get(index, self.unit_types))
