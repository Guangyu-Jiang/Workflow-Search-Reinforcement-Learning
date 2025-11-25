#!/usr/bin/env python3
"""
Subnet-Level Priority Workflows with Continuous Parameters
Solves the problem of too few discrete orders by adding continuous priority weights
and response strategies within each subnet
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import permutations
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# HYBRID WORKFLOW REPRESENTATION
# ============================================================================

@dataclass
class SubnetPriorityWorkflow:
    """
    Workflow combining:
    1. Subnet ordering (3! = 6 permutations)
    2. Continuous priority weights within/between subnets
    3. Response strategy parameters
    """
    
    # Discrete component: subnet order
    subnet_order: List[str]  # e.g., ['operational', 'enterprise', 'user']
    
    # Continuous components (8 dimensions total)
    # Dimensions 0-2: Relative importance weights for each subnet
    subnet_weights: np.ndarray  # [user_weight, enterprise_weight, operational_weight]
    
    # Dimension 3: Intra-subnet strategy (how to handle multiple compromised hosts in same subnet)
    intra_subnet_strategy: float  # -1: sequential, 0: parallel, +1: worst-first
    
    # Dimension 4: Response aggressiveness 
    response_aggressiveness: float  # -1: analyze first, 0: balanced, +1: immediate fix
    
    # Dimension 5: Fix method preference
    fix_method: float  # -1: remove, 0: adaptive, +1: restore
    
    # Dimension 6: Defender priority
    defender_priority: float  # -1: low, +1: highest
    
    # Dimension 7: Adaptation rate
    adaptation_rate: float  # -1: stick to plan, +1: highly adaptive
    
    def to_vector(self) -> np.ndarray:
        """Convert to 8D continuous vector for GP-UCB"""
        # Encode subnet order as continuous (using position encoding)
        order_encoding = self._encode_order()
        
        vector = np.concatenate([
            self.subnet_weights,  # 3 dims
            [self.intra_subnet_strategy],  # 1 dim
            [self.response_aggressiveness],  # 1 dim
            [self.fix_method],  # 1 dim
            [self.defender_priority],  # 1 dim
            [self.adaptation_rate]  # 1 dim
        ])
        
        # Mix in order encoding to differentiate the 6 permutations
        vector[:3] += order_encoding * 0.3  # Subtle influence
        
        return vector
    
    def _encode_order(self) -> np.ndarray:
        """Encode discrete subnet order as continuous signal"""
        order_map = {
            'user': 0,
            'enterprise': 1,
            'operational': 2
        }
        
        encoding = np.zeros(3)
        for i, subnet in enumerate(self.subnet_order):
            if subnet in order_map:
                # Earlier position = higher value
                encoding[order_map[subnet]] = 1.0 - (i * 0.3)
        
        return encoding


class WorkflowSpace:
    """
    Structured workflow space combining discrete and continuous components
    """
    
    def __init__(self):
        self.subnet_names = ['user', 'enterprise', 'operational']
        self.all_orders = list(permutations(self.subnet_names))  # 6 permutations
        
        # Host mapping for CAGE2
        self.subnet_hosts = {
            'user': ['User0', 'User1', 'User2', 'User3', 'User4'],
            'enterprise': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
            'operational': ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2'],
            'defender': ['Defender']
        }
        
        # Action mapping
        self.host_actions = {
            'User0': {'analyze': 11, 'remove': 24, 'restore': 141},
            'User1': {'analyze': 12, 'remove': 25, 'restore': 142},
            'User2': {'analyze': 13, 'remove': 26, 'restore': 143},
            'User3': {'analyze': 14, 'remove': 27, 'restore': 144},
            'User4': {'analyze': 14, 'remove': 27, 'restore': 144},
            'Enterprise0': {'analyze': 3, 'remove': 16, 'restore': 133},
            'Enterprise1': {'analyze': 4, 'remove': 17, 'restore': 134},
            'Enterprise2': {'analyze': 5, 'remove': 18, 'restore': 135},
            'Op_Server0': {'analyze': 9, 'remove': 22, 'restore': 139},
            'Op_Host0': {'analyze': 9, 'remove': 22, 'restore': 139},
            'Op_Host1': {'analyze': 9, 'remove': 22, 'restore': 139},
            'Op_Host2': {'analyze': 9, 'remove': 22, 'restore': 139},
            'Defender': {'analyze': 2, 'remove': 15, 'restore': 132}
        }
    
    def sample_workflow(self, method: str = 'random') -> SubnetPriorityWorkflow:
        """Sample a workflow from the space"""
        
        if method == 'random':
            order = random.choice(self.all_orders)
            weights = np.random.dirichlet([1, 1, 1])  # Sum to 1
            
            return SubnetPriorityWorkflow(
                subnet_order=list(order),
                subnet_weights=weights,
                intra_subnet_strategy=np.random.uniform(-1, 1),
                response_aggressiveness=np.random.uniform(-1, 1),
                fix_method=np.random.uniform(-1, 1),
                defender_priority=np.random.uniform(-1, 1),
                adaptation_rate=np.random.uniform(-1, 1)
            )
        
        elif method == 'canonical':
            # Return one of the canonical strategies
            canonical = self.get_canonical_workflows()
            return random.choice(list(canonical.values()))
        
        elif method == 'guided':
            # Sample with bias toward reasonable strategies
            order = random.choice(self.all_orders)
            
            # Bias weights based on order
            if order[0] == 'operational':
                weights = np.random.dirichlet([1, 2, 3])  # Favor operational
            elif order[0] == 'enterprise':
                weights = np.random.dirichlet([1, 3, 2])  # Favor enterprise
            else:
                weights = np.random.dirichlet([3, 1, 1])  # Favor user
            
            return SubnetPriorityWorkflow(
                subnet_order=list(order),
                subnet_weights=weights,
                intra_subnet_strategy=np.random.normal(0, 0.5),  # Tend toward parallel
                response_aggressiveness=np.random.normal(0.3, 0.3),  # Slightly aggressive
                fix_method=np.random.normal(0.2, 0.4),  # Slight restore preference
                defender_priority=np.random.normal(0.7, 0.2),  # High defender priority
                adaptation_rate=np.random.normal(0, 0.3)  # Moderate adaptation
            )
    
    def get_canonical_workflows(self) -> Dict[str, SubnetPriorityWorkflow]:
        """Define canonical workflow strategies"""
        
        workflows = {}
        
        # 1. Critical Infrastructure First
        workflows['critical_first'] = SubnetPriorityWorkflow(
            subnet_order=['operational', 'enterprise', 'user'],
            subnet_weights=np.array([0.2, 0.3, 0.5]),
            intra_subnet_strategy=1.0,  # Worst-first within subnet
            response_aggressiveness=0.8,  # Fast response
            fix_method=0.5,  # Prefer restore
            defender_priority=1.0,  # Highest priority
            adaptation_rate=-0.5  # Stick to plan
        )
        
        # 2. User Experience First
        workflows['user_first'] = SubnetPriorityWorkflow(
            subnet_order=['user', 'enterprise', 'operational'],
            subnet_weights=np.array([0.5, 0.3, 0.2]),
            intra_subnet_strategy=0.0,  # Parallel within subnet
            response_aggressiveness=0.6,
            fix_method=0.8,  # Strong restore preference
            defender_priority=0.3,
            adaptation_rate=0.3
        )
        
        # 3. Adaptive Defense
        workflows['adaptive'] = SubnetPriorityWorkflow(
            subnet_order=['enterprise', 'operational', 'user'],
            subnet_weights=np.array([0.33, 0.34, 0.33]),  # Balanced
            intra_subnet_strategy=0.0,
            response_aggressiveness=0.0,  # Balanced
            fix_method=0.0,  # Adaptive
            defender_priority=0.5,
            adaptation_rate=0.9  # Highly adaptive
        )
        
        # 4. Aggressive Cleanup
        workflows['aggressive'] = SubnetPriorityWorkflow(
            subnet_order=['operational', 'user', 'enterprise'],
            subnet_weights=np.array([0.3, 0.2, 0.5]),
            intra_subnet_strategy=-0.5,  # Sequential
            response_aggressiveness=1.0,  # Immediate
            fix_method=-0.8,  # Prefer remove
            defender_priority=0.8,
            adaptation_rate=-0.3
        )
        
        # 5. Information Gathering
        workflows['information'] = SubnetPriorityWorkflow(
            subnet_order=['enterprise', 'user', 'operational'],
            subnet_weights=np.array([0.3, 0.4, 0.3]),
            intra_subnet_strategy=0.3,
            response_aggressiveness=-0.8,  # Analyze first
            fix_method=0.2,
            defender_priority=0.4,
            adaptation_rate=0.5
        )
        
        # 6. Fortress Defense
        workflows['fortress'] = SubnetPriorityWorkflow(
            subnet_order=['operational', 'enterprise', 'user'],
            subnet_weights=np.array([0.1, 0.2, 0.7]),  # Heavy operational focus
            intra_subnet_strategy=0.8,
            response_aggressiveness=0.5,
            fix_method=0.3,
            defender_priority=1.0,
            adaptation_rate=-0.8  # Rigid plan
        )
        
        return workflows
    
    def workflow_distance(self, w1: SubnetPriorityWorkflow, w2: SubnetPriorityWorkflow) -> float:
        """Compute distance between two workflows"""
        
        # Convert to vectors
        v1 = w1.to_vector()
        v2 = w2.to_vector()
        
        # Weighted distance (some dimensions more important)
        weights = np.array([
            1.2, 1.2, 1.2,  # Subnet weights - important
            0.8,  # Intra-subnet strategy
            1.0,  # Response aggressiveness
            0.9,  # Fix method
            1.1,  # Defender priority
            0.7   # Adaptation rate
        ])
        
        diff = (v1 - v2) * weights
        return np.linalg.norm(diff)


# ============================================================================
# WORKFLOW EXECUTION AND COMPLIANCE
# ============================================================================

class WorkflowExecutor:
    """
    Executes actions based on workflow and current state
    """
    
    def __init__(self, workflow: SubnetPriorityWorkflow, workflow_space: WorkflowSpace):
        self.workflow = workflow
        self.space = workflow_space
        self.execution_history = []
        
    def get_next_action(self, compromised_hosts: Dict[str, float], 
                       timestep: int) -> Tuple[str, int]:
        """
        Get next action based on workflow and compromised hosts
        
        Args:
            compromised_hosts: {hostname: compromise_level}
            timestep: Current timestep
            
        Returns:
            (target_host, action_id)
        """
        
        if not compromised_hosts:
            return None, 0  # Sleep
        
        # Check defender first if high priority
        if 'Defender' in compromised_hosts and self.workflow.defender_priority > 0.5:
            return self._get_fix_action('Defender', compromised_hosts['Defender'])
        
        # Group by subnet
        subnet_compromises = self._group_by_subnet(compromised_hosts)
        
        # Select subnet based on workflow order and weights
        target_subnet = self._select_subnet(subnet_compromises)
        
        if not target_subnet:
            return None, 0
        
        # Select host within subnet based on intra-subnet strategy
        target_host = self._select_host_in_subnet(
            subnet_compromises[target_subnet],
            self.workflow.intra_subnet_strategy
        )
        
        # Get action type based on workflow parameters
        return self._get_fix_action(target_host, compromised_hosts[target_host])
    
    def _group_by_subnet(self, compromised_hosts: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Group compromised hosts by subnet"""
        subnet_groups = {}
        
        for host, level in compromised_hosts.items():
            subnet = None
            for sn, hosts in self.space.subnet_hosts.items():
                if host in hosts:
                    subnet = sn
                    break
            
            if subnet and subnet != 'defender':  # Handle defender separately
                if subnet not in subnet_groups:
                    subnet_groups[subnet] = {}
                subnet_groups[subnet][host] = level
        
        return subnet_groups
    
    def _select_subnet(self, subnet_compromises: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Select subnet to fix based on workflow order and weights"""
        
        if not subnet_compromises:
            return None
        
        # Score each subnet
        subnet_scores = {}
        
        for subnet in subnet_compromises:
            # Base score from order
            try:
                order_idx = self.workflow.subnet_order.index(subnet)
                order_score = 1.0 / (order_idx + 1)  # Earlier = higher
            except ValueError:
                order_score = 0.1
            
            # Weight score
            weight_idx = self.space.subnet_names.index(subnet)
            weight_score = self.workflow.subnet_weights[weight_idx]
            
            # Severity score (average compromise level)
            severity = np.mean(list(subnet_compromises[subnet].values()))
            
            # Combined score
            subnet_scores[subnet] = order_score * 2 + weight_score + severity * 0.5
        
        # Select highest score
        return max(subnet_scores, key=subnet_scores.get)
    
    def _select_host_in_subnet(self, hosts: Dict[str, float], strategy: float) -> str:
        """Select which host to fix within a subnet"""
        
        if len(hosts) == 1:
            return list(hosts.keys())[0]
        
        if strategy < -0.5:
            # Sequential: first in list
            return list(hosts.keys())[0]
        elif strategy > 0.5:
            # Worst-first: highest compromise
            return max(hosts, key=hosts.get)
        else:
            # Parallel/random: pick randomly
            return random.choice(list(hosts.keys()))
    
    def _get_fix_action(self, host: str, compromise_level: float) -> Tuple[str, int]:
        """Determine action type and ID"""
        
        if host not in self.space.host_actions:
            return host, 0
        
        actions = self.space.host_actions[host]
        
        # Determine action type based on workflow parameters
        if self.workflow.response_aggressiveness < -0.5:
            # Analyze first
            action_type = 'analyze'
        elif self.workflow.fix_method < -0.5:
            # Prefer remove
            action_type = 'remove'
        elif self.workflow.fix_method > 0.5:
            # Prefer restore
            action_type = 'restore'
        else:
            # Adaptive based on compromise level
            if compromise_level > 0.7:
                action_type = 'restore'
            elif compromise_level > 0.3:
                action_type = 'remove'
            else:
                action_type = 'analyze'
        
        action_id = actions.get(action_type, 0)
        
        # Record execution
        self.execution_history.append({
            'timestep': len(self.execution_history),
            'host': host,
            'action_type': action_type,
            'action_id': action_id,
            'compromise_level': compromise_level
        })
        
        return host, action_id


class WorkflowComplianceChecker:
    """
    Check if actual execution follows the workflow
    """
    
    def __init__(self, workflow: SubnetPriorityWorkflow, workflow_space: WorkflowSpace):
        self.workflow = workflow
        self.space = workflow_space
        self.fix_history = []  # List of (timestep, host, true_compromised_hosts)
        
    def record_fix(self, timestep: int, action: int, true_state: Dict[str, float]):
        """
        Record a fix action with true environment state
        
        Args:
            timestep: When action was taken
            action: Action ID
            true_state: True compromise state {hostname: level}
        """
        
        # Map action to host
        target_host = self._action_to_host(action)
        
        if target_host:
            self.fix_history.append({
                'timestep': timestep,
                'host': target_host,
                'action': action,
                'compromised': true_state.copy(),
                'subnet': self._get_subnet(target_host)
            })
    
    def _action_to_host(self, action: int) -> Optional[str]:
        """Map action ID to target host"""
        for host, actions in self.space.host_actions.items():
            if action in actions.values():
                return host
        return None
    
    def _get_subnet(self, host: str) -> Optional[str]:
        """Get subnet for a host"""
        for subnet, hosts in self.space.subnet_hosts.items():
            if host in hosts:
                return subnet
        return None
    
    def compute_compliance(self) -> Dict[str, float]:
        """
        Compute compliance metrics
        
        Returns dict with:
        - order_compliance: Did we fix subnets in right order?
        - weight_compliance: Did we respect subnet weights?
        - strategy_compliance: Did we follow intra-subnet strategy?
        - overall: Combined score
        """
        
        if not self.fix_history:
            return {'overall': 0.0}
        
        metrics = {}
        
        # 1. Order compliance: Check if subnet fixes follow intended order
        metrics['order_compliance'] = self._check_order_compliance()
        
        # 2. Weight compliance: Check if fix frequency matches weights
        metrics['weight_compliance'] = self._check_weight_compliance()
        
        # 3. Strategy compliance: Check intra-subnet behavior
        metrics['strategy_compliance'] = self._check_strategy_compliance()
        
        # 4. Priority compliance: Check if higher priority compromised hosts fixed first
        metrics['priority_compliance'] = self._check_priority_compliance()
        
        # Overall score
        metrics['overall'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _check_order_compliance(self) -> float:
        """Check if subnets are fixed in intended order"""
        
        # Extract subnet fix sequence
        subnet_sequence = []
        for record in self.fix_history:
            subnet = record['subnet']
            if subnet and subnet not in subnet_sequence and subnet != 'defender':
                subnet_sequence.append(subnet)
        
        if not subnet_sequence:
            return 0.5
        
        # Compare to intended order
        score = 0.0
        for i, subnet in enumerate(subnet_sequence):
            if subnet in self.workflow.subnet_order:
                intended_pos = self.workflow.subnet_order.index(subnet)
                # Give points for being in approximately right position
                position_diff = abs(i - intended_pos)
                score += max(0, 1 - position_diff * 0.3)
        
        return score / len(subnet_sequence)
    
    def _check_weight_compliance(self) -> float:
        """Check if fix frequency matches subnet weights"""
        
        # Count fixes per subnet
        subnet_counts = {'user': 0, 'enterprise': 0, 'operational': 0}
        
        for record in self.fix_history:
            subnet = record['subnet']
            if subnet in subnet_counts:
                subnet_counts[subnet] += 1
        
        total = sum(subnet_counts.values())
        if total == 0:
            return 0.5
        
        # Compare to intended weights
        actual_weights = np.array([
            subnet_counts['user'] / total,
            subnet_counts['enterprise'] / total,
            subnet_counts['operational'] / total
        ])
        
        intended_weights = self.workflow.subnet_weights
        
        # Compute similarity (1 - normalized distance)
        distance = np.linalg.norm(actual_weights - intended_weights)
        max_distance = np.sqrt(2)  # Maximum possible distance
        
        return 1 - (distance / max_distance)
    
    def _check_strategy_compliance(self) -> float:
        """Check if intra-subnet strategy is followed"""
        
        scores = []
        
        # Group fixes by subnet and time window
        time_windows = {}
        for record in self.fix_history:
            window = record['timestep'] // 10
            if window not in time_windows:
                time_windows[window] = {'user': [], 'enterprise': [], 'operational': []}
            
            subnet = record['subnet']
            if subnet in time_windows[window]:
                time_windows[window][subnet].append(record)
        
        # Check each window
        for window, subnet_fixes in time_windows.items():
            for subnet, fixes in subnet_fixes.items():
                if len(fixes) > 1:
                    # Multiple fixes in same subnet - check strategy
                    if self.workflow.intra_subnet_strategy > 0.5:
                        # Should fix worst first
                        compromise_levels = [f['compromised'].get(f['host'], 0) for f in fixes]
                        if compromise_levels == sorted(compromise_levels, reverse=True):
                            scores.append(1.0)
                        else:
                            scores.append(0.5)
                    else:
                        # Any order is fine
                        scores.append(1.0)
        
        return np.mean(scores) if scores else 0.8
    
    def _check_priority_compliance(self) -> float:
        """Check if we fix highest priority compromised hosts"""
        
        scores = []
        
        for record in self.fix_history:
            compromised = record['compromised']
            fixed_host = record['host']
            
            if not compromised:
                continue
            
            # Compute priority score for fixed host
            fixed_priority = self._compute_host_priority(fixed_host, compromised[fixed_host])
            
            # Check if it was highest priority
            max_priority = max(
                self._compute_host_priority(h, level) 
                for h, level in compromised.items()
            )
            
            # Score based on how close we were to optimal
            if max_priority > 0:
                scores.append(fixed_priority / max_priority)
        
        return np.mean(scores) if scores else 0.5
    
    def _compute_host_priority(self, host: str, compromise_level: float) -> float:
        """Compute priority score for a host"""
        
        subnet = self._get_subnet(host)
        
        if not subnet or subnet == 'defender':
            return 10.0 if host == 'Defender' else 1.0
        
        # Get subnet priority from workflow
        try:
            subnet_idx = self.workflow.subnet_order.index(subnet)
            order_priority = 1.0 / (subnet_idx + 1)
        except ValueError:
            order_priority = 0.1
        
        # Get weight priority
        weight_idx = self.space.subnet_names.index(subnet)
        weight_priority = self.workflow.subnet_weights[weight_idx]
        
        # Combine with compromise level
        return (order_priority * 2 + weight_priority) * compromise_level


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_workflow_system():
    """Demonstrate the subnet priority workflow system"""
    
    print("="*70)
    print("SUBNET PRIORITY WORKFLOW SYSTEM")
    print("="*70)
    
    # Create workflow space
    space = WorkflowSpace()
    
    # Show canonical workflows
    print("\n1. CANONICAL WORKFLOWS:")
    print("-" * 40)
    
    canonical = space.get_canonical_workflows()
    for name, workflow in list(canonical.items())[:3]:
        print(f"\n{name}:")
        print(f"  Order: {' → '.join(workflow.subnet_order)}")
        print(f"  Weights: U={workflow.subnet_weights[0]:.2f}, "
              f"E={workflow.subnet_weights[1]:.2f}, "
              f"O={workflow.subnet_weights[2]:.2f}")
        print(f"  Strategy: {workflow.intra_subnet_strategy:.2f}")
        print(f"  8D Vector: {workflow.to_vector()[:4]}...")
    
    # Test workflow execution
    print("\n2. WORKFLOW EXECUTION:")
    print("-" * 40)
    
    workflow = canonical['critical_first']
    executor = WorkflowExecutor(workflow, space)
    
    # Simulate compromised hosts
    compromised = {
        'User0': 0.8,
        'User1': 0.3,
        'Enterprise0': 0.9,
        'Op_Server0': 0.5
    }
    
    print(f"\nCompromised hosts: {list(compromised.keys())}")
    print(f"Using workflow: critical_first")
    
    # Get next action
    target_host, action_id = executor.get_next_action(compromised, timestep=10)
    print(f"Next action: Fix {target_host} with action {action_id}")
    
    # Test compliance checking
    print("\n3. COMPLIANCE CHECKING:")
    print("-" * 40)
    
    checker = WorkflowComplianceChecker(workflow, space)
    
    # Simulate some fixes
    checker.record_fix(0, 139, {'Op_Server0': 0.5, 'User0': 0.8})  # Fix Op_Server0
    checker.record_fix(1, 133, {'Enterprise0': 0.9, 'User0': 0.8})  # Fix Enterprise0
    checker.record_fix(2, 141, {'User0': 0.8, 'User1': 0.3})  # Fix User0
    
    compliance = checker.compute_compliance()
    print(f"\nCompliance scores:")
    for metric, score in compliance.items():
        print(f"  {metric}: {score:.2f}")


if __name__ == "__main__":
    demonstrate_workflow_system()
