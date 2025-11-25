#!/usr/bin/env python3
"""
Unit Type-Based Workflow Representation for CAGE2
Workflows defined as priority orders over 5 unit types
"""

import numpy as np
from typing import List, Tuple, Dict
from itertools import permutations
import random


class UnitTypeWorkflow:
    """
    Represents workflows as priority orderings of unit types
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
    
    def order_to_vector(self, order: List[str]) -> np.ndarray:
        """
        Convert unit type order to 8D continuous vector for GP-UCB
        
        Features:
        1-5: Position of each type (normalized to [0,1])
        6: Critical-first score (defender + op_server position)
        7: User-last score (user position)
        8: Clustering score (similar types together)
        """
        
        vector = np.zeros(8)
        
        # Features 1-5: Normalized positions
        for i, unit_type in enumerate(self.unit_types):
            if unit_type in order:
                position = order.index(unit_type)
                vector[i] = position / 4.0  # Normalize to [0,1]
        
        # Feature 6: Critical-first score
        defender_pos = order.index('defender') if 'defender' in order else 4
        op_server_pos = order.index('op_server') if 'op_server' in order else 4
        vector[5] = 1.0 - (defender_pos + op_server_pos) / 8.0
        
        # Feature 7: User-last score  
        user_pos = order.index('user') if 'user' in order else 0
        vector[6] = user_pos / 4.0
        
        # Feature 8: Clustering score (enterprise units together)
        if 'enterprise' in order and 'op_server' in order:
            ent_pos = order.index('enterprise')
            ops_pos = order.index('op_server')
            vector[7] = 1.0 - abs(ent_pos - ops_pos) / 4.0
        else:
            vector[7] = 0.5
        
        return vector
    
    def vector_to_order(self, vector: np.ndarray) -> List[str]:
        """
        Convert vector back to closest valid ordering
        """
        
        # Extract positions from first 5 features
        positions = [(self.unit_types[i], vector[i]) for i in range(5)]
        
        # Sort by position value
        positions.sort(key=lambda x: x[1])
        
        # Return ordering
        return [unit_type for unit_type, _ in positions]
    
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
        Return canonical workflow strategies
        """
        
        return {
            'critical_first': ['defender', 'op_server', 'enterprise', 'op_host', 'user'],
            'enterprise_focus': ['enterprise', 'defender', 'op_server', 'op_host', 'user'],
            'user_priority': ['user', 'defender', 'enterprise', 'op_server', 'op_host'],
            'operational_focus': ['op_server', 'op_host', 'defender', 'enterprise', 'user'],
            'balanced': ['defender', 'enterprise', 'op_server', 'user', 'op_host'],
            'reverse': ['user', 'op_host', 'enterprise', 'op_server', 'defender']
        }
    
    def compute_distance(self, order1: List[str], order2: List[str]) -> float:
        """
        Compute distance between two orderings (for GP kernel)
        Uses Kendall's tau distance
        """
        
        # Get positions of each type in both orders
        pos1 = {t: order1.index(t) for t in order1}
        pos2 = {t: order2.index(t) for t in order2}
        
        # Count disagreements
        disagreements = 0
        for i, type1 in enumerate(self.unit_types):
            for j, type2 in enumerate(self.unit_types):
                if i < j:  # Only check each pair once
                    # Check if relative order is different
                    in_order1 = pos1.get(type1, 5) < pos1.get(type2, 5)
                    in_order2 = pos2.get(type1, 5) < pos2.get(type2, 5)
                    if in_order1 != in_order2:
                        disagreements += 1
        
        # Normalize to [0, 1]
        max_disagreements = len(self.unit_types) * (len(self.unit_types) - 1) / 2
        return disagreements / max_disagreements
