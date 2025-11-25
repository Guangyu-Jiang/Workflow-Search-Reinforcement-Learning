#!/usr/bin/env python3
"""
Workflow Based on Host Fixing Order
Define workflows as the prioritized sequence for defending/restoring hosts
Even though red's activities make execution unpredictable, we can still
define and follow a strategic order
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from itertools import permutations
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# HOST FIXING ORDER REPRESENTATION
# ============================================================================

class HostFixingOrderWorkflow:
    """
    Workflow defined as the order in which we prioritize fixing hosts
    """
    
    def __init__(self):
        # Define all hosts in CAGE2
        self.hosts = {
            'User0': {'subnet': 'user', 'importance': 1, 'index': 0},
            'User1': {'subnet': 'user', 'importance': 1, 'index': 1},
            'User2': {'subnet': 'user', 'importance': 1, 'index': 2},
            'User3': {'subnet': 'user', 'importance': 1, 'index': 3},
            'User4': {'subnet': 'user', 'importance': 1, 'index': 4},
            'Enterprise0': {'subnet': 'enterprise', 'importance': 3, 'index': 5},
            'Enterprise1': {'subnet': 'enterprise', 'importance': 3, 'index': 6},
            'Enterprise2': {'subnet': 'enterprise', 'importance': 3, 'index': 7},
            'Op_Server0': {'subnet': 'operational', 'importance': 5, 'index': 8},
            'Op_Host0': {'subnet': 'operational', 'importance': 2, 'index': 9},
            'Op_Host1': {'subnet': 'operational', 'importance': 2, 'index': 10},
            'Op_Host2': {'subnet': 'operational', 'importance': 2, 'index': 11},
            'Defender': {'subnet': 'defender', 'importance': 10, 'index': 12}
        }
        
        # Map host indices to action IDs for restore/remove/analyze
        self.host_to_actions = {
            'User0': {'analyze': 11, 'remove': 24, 'restore': 141, 'decoy': 1003},
            'User1': {'analyze': 12, 'remove': 25, 'restore': 142, 'decoy': 1004},
            'User2': {'analyze': 13, 'remove': 26, 'restore': 143, 'decoy': 1005},
            'User3': {'analyze': 14, 'remove': 27, 'restore': 144, 'decoy': 1006},
            'User4': {'analyze': 14, 'remove': 27, 'restore': 144, 'decoy': 1006},  # Shared with User3
            'Enterprise0': {'analyze': 3, 'remove': 16, 'restore': 133, 'decoy': 1000},
            'Enterprise1': {'analyze': 4, 'remove': 17, 'restore': 134, 'decoy': 1001},
            'Enterprise2': {'analyze': 5, 'remove': 18, 'restore': 135, 'decoy': 1002},
            'Op_Server0': {'analyze': 9, 'remove': 22, 'restore': 139, 'decoy': 1008},
            'Op_Host0': {'analyze': 9, 'remove': 22, 'restore': 139, 'decoy': 1007},
            'Op_Host1': {'analyze': 9, 'remove': 22, 'restore': 139, 'decoy': 1007},
            'Op_Host2': {'analyze': 9, 'remove': 22, 'restore': 139, 'decoy': 1007},
            'Defender': {'analyze': 2, 'remove': 15, 'restore': 132, 'decoy': None}
        }
        
    def create_workflow_from_order(self, host_order: List[str]) -> 'FixingOrderWorkflow':
        """
        Create a workflow from an ordered list of hosts
        
        Example: ['Op_Server0', 'Enterprise0', 'Enterprise1', 'User0', ...]
        means fix operational server first, then enterprise hosts, then users
        """
        return FixingOrderWorkflow(host_order, self.hosts, self.host_to_actions)
    
    def create_canonical_workflows(self) -> Dict[str, List[str]]:
        """
        Create canonical host fixing orders based on different strategies
        """
        workflows = {}
        
        # 1. Critical First: Protect most important assets first
        workflows['critical_first'] = [
            'Defender',
            'Op_Server0',
            'Enterprise0', 'Enterprise1', 'Enterprise2',
            'Op_Host0', 'Op_Host1', 'Op_Host2',
            'User0', 'User1', 'User2', 'User3', 'User4'
        ]
        
        # 2. User First: Prioritize user experience
        workflows['user_first'] = [
            'User0', 'User1', 'User2', 'User3', 'User4',
            'Defender',
            'Enterprise0', 'Enterprise1', 'Enterprise2',
            'Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2'
        ]
        
        # 3. Enterprise Focus: Business continuity priority
        workflows['enterprise_focus'] = [
            'Enterprise0', 'Enterprise1', 'Enterprise2',
            'Defender',
            'Op_Server0',
            'User0', 'User1', 'User2', 'User3', 'User4',
            'Op_Host0', 'Op_Host1', 'Op_Host2'
        ]
        
        # 4. Layered Defense: Fix by network layers
        workflows['layered_defense'] = [
            'Defender',  # Core
            'Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2',  # Operational layer
            'Enterprise0', 'Enterprise1', 'Enterprise2',  # Business layer
            'User0', 'User1', 'User2', 'User3', 'User4'  # User layer
        ]
        
        # 5. Balanced: Mix based on compromise likelihood
        workflows['balanced'] = [
            'Defender',
            'User0',  # Often compromised first
            'Enterprise0',
            'Op_Server0',
            'User1', 'User2',
            'Enterprise1', 'Enterprise2',
            'User3', 'User4',
            'Op_Host0', 'Op_Host1', 'Op_Host2'
        ]
        
        return workflows


class FixingOrderWorkflow:
    """
    A specific workflow instance with host fixing order
    """
    
    def __init__(self, host_order: List[str], host_info: Dict, host_to_actions: Dict):
        self.host_order = host_order
        self.host_info = host_info
        self.host_to_actions = host_to_actions
        
        # Create priority mapping
        self.priority = {host: i for i, host in enumerate(host_order)}
        
        # Track fixing progress
        self.fixed_hosts = set()
        self.current_target = 0
        
    def get_next_host_to_fix(self, compromised_hosts: Set[str]) -> Optional[str]:
        """
        Get the next host to fix based on workflow order and current compromises
        """
        # Find highest priority compromised host
        for host in self.host_order:
            if host in compromised_hosts and host not in self.fixed_hosts:
                return host
        return None
    
    def get_action_for_host(self, host: str, observation: np.ndarray, 
                           action_type: str = 'adaptive') -> int:
        """
        Get the appropriate action for fixing a host
        
        action_type: 'analyze', 'remove', 'restore', 'decoy', or 'adaptive'
        """
        if host not in self.host_to_actions:
            return 0  # Sleep if host not found
        
        actions = self.host_to_actions[host]
        
        if action_type == 'adaptive':
            # Choose action based on observation
            host_idx = self.host_info[host]['index']
            
            # Check compromise level from observation (simplified)
            if host_idx * 4 + 2 < len(observation):
                compromise_level = observation[host_idx * 4 + 2]
                
                if compromise_level > 0.7:
                    action_type = 'restore'  # Heavily compromised
                elif compromise_level > 0.3:
                    action_type = 'remove'   # Moderately compromised
                elif compromise_level > 0:
                    action_type = 'analyze'  # Suspicious activity
                else:
                    action_type = 'decoy'    # Preventive
            else:
                action_type = 'analyze'  # Default to analyze
        
        return actions.get(action_type, 0)


# ============================================================================
# WORKFLOW EMBEDDING FOR HOST ORDERS
# ============================================================================

class HostOrderEmbedding:
    """
    Convert host fixing orders to continuous embeddings
    """
    
    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        self.host_manager = HostFixingOrderWorkflow()
        
        # Learn embedding for each host position
        self.position_encoder = nn.Sequential(
            nn.Linear(13, 32),  # 13 hosts
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        ).to(device)
        
    def order_to_embedding(self, host_order: List[str]) -> np.ndarray:
        """
        Convert host order to continuous embedding
        
        Method 1: Position-based encoding
        - Early positions get higher weight
        - Similar orders produce similar embeddings
        """
        embedding = np.zeros(self.embedding_dim)
        
        # Create position vector
        position_vector = np.zeros(13)
        for pos, host in enumerate(host_order):
            if host in self.host_manager.hosts:
                host_idx = self.host_manager.hosts[host]['index']
                # Weight by position (earlier = more important)
                position_vector[host_idx] = 1.0 / (pos + 1)
        
        # Convert to embedding
        with torch.no_grad():
            pos_tensor = torch.FloatTensor(position_vector).unsqueeze(0).to(device)
            embedding_tensor = self.position_encoder(pos_tensor)
            embedding = embedding_tensor.cpu().numpy()[0]
        
        return embedding
    
    def order_to_features(self, host_order: List[str]) -> np.ndarray:
        """
        Convert host order to interpretable features
        
        Method 2: Extract semantic features from order
        """
        features = []
        
        # Feature 1: Average position of user hosts (0-1)
        user_positions = [i for i, h in enumerate(host_order) 
                         if 'User' in h]
        avg_user_pos = np.mean(user_positions) / len(host_order) if user_positions else 0.5
        features.append(avg_user_pos)
        
        # Feature 2: Average position of enterprise hosts (0-1)
        ent_positions = [i for i, h in enumerate(host_order) 
                        if 'Enterprise' in h]
        avg_ent_pos = np.mean(ent_positions) / len(host_order) if ent_positions else 0.5
        features.append(avg_ent_pos)
        
        # Feature 3: Average position of operational hosts (0-1)
        op_positions = [i for i, h in enumerate(host_order) 
                       if 'Op_' in h]
        avg_op_pos = np.mean(op_positions) / len(host_order) if op_positions else 0.5
        features.append(avg_op_pos)
        
        # Feature 4: Defender position (0-1)
        def_pos = host_order.index('Defender') / len(host_order) if 'Defender' in host_order else 0.5
        features.append(def_pos)
        
        # Feature 5: Clustering metric (are similar hosts grouped?)
        clustering = self._compute_clustering(host_order)
        features.append(clustering)
        
        # Feature 6-8: Subnet priority order
        subnet_order = self._get_subnet_order(host_order)
        features.extend(subnet_order)
        
        return np.array(features)
    
    def _compute_clustering(self, host_order: List[str]) -> float:
        """
        Measure how clustered hosts of same type are
        0 = scattered, 1 = perfectly clustered
        """
        cluster_score = 0
        total_pairs = 0
        
        for subnet in ['user', 'enterprise', 'operational']:
            hosts_in_subnet = [i for i, h in enumerate(host_order)
                              if subnet in h.lower() or 
                              (subnet == 'operational' and 'Op_' in h)]
            
            if len(hosts_in_subnet) > 1:
                # Check adjacency
                for i in range(len(hosts_in_subnet) - 1):
                    if hosts_in_subnet[i+1] - hosts_in_subnet[i] == 1:
                        cluster_score += 1
                    total_pairs += 1
        
        return cluster_score / total_pairs if total_pairs > 0 else 0.5
    
    def _get_subnet_order(self, host_order: List[str]) -> List[float]:
        """
        Determine which subnet is prioritized first, second, third
        """
        first_positions = {
            'user': float('inf'),
            'enterprise': float('inf'),
            'operational': float('inf')
        }
        
        for i, host in enumerate(host_order):
            if 'User' in host:
                first_positions['user'] = min(first_positions['user'], i)
            elif 'Enterprise' in host:
                first_positions['enterprise'] = min(first_positions['enterprise'], i)
            elif 'Op_' in host:
                first_positions['operational'] = min(first_positions['operational'], i)
        
        # Convert to normalized scores
        scores = []
        for subnet in ['user', 'enterprise', 'operational']:
            if first_positions[subnet] == float('inf'):
                scores.append(0.5)
            else:
                scores.append(1.0 - first_positions[subnet] / len(host_order))
        
        return scores


# ============================================================================
# POLICY THAT FOLLOWS HOST FIXING ORDER
# ============================================================================

class HostOrderGuidedPolicy(nn.Module):
    """
    Policy that follows a host fixing order workflow
    """
    
    def __init__(self, state_dim: int = 52, action_dim: int = 27):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.host_manager = HostFixingOrderWorkflow()
        
        # Network to process state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Network to process workflow embedding
        self.workflow_encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Decision network
        self.decision = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state: torch.Tensor, workflow_embedding: torch.Tensor,
                current_workflow: FixingOrderWorkflow) -> torch.Tensor:
        """
        Generate actions following the host fixing order
        """
        # Encode state and workflow
        state_features = self.state_encoder(state)
        workflow_features = self.workflow_encoder(workflow_embedding)
        
        # Combine features
        combined = torch.cat([state_features, workflow_features], dim=-1)
        action_logits = self.decision(combined)
        
        # Bias toward next host in fixing order
        biased_logits = self._apply_order_bias(
            action_logits, state, current_workflow
        )
        
        return torch.softmax(biased_logits, dim=-1)
    
    def _apply_order_bias(self, logits: torch.Tensor, 
                          state: torch.Tensor,
                          workflow: FixingOrderWorkflow) -> torch.Tensor:
        """
        Bias actions toward fixing hosts in the specified order
        """
        # Detect compromised hosts from state (simplified)
        compromised = self._detect_compromised(state)
        
        # Get next host to fix
        next_host = workflow.get_next_host_to_fix(compromised)
        
        if next_host:
            # Boost actions for this host
            if next_host in workflow.host_to_actions:
                actions = workflow.host_to_actions[next_host]
                for action_type, action_id in actions.items():
                    if action_id is not None and action_id < self.action_dim:
                        logits[0, action_id] += 3.0  # Strong bias
        
        return logits
    
    def _detect_compromised(self, state: torch.Tensor) -> Set[str]:
        """
        Detect which hosts are compromised from state
        """
        compromised = set()
        state_np = state.cpu().numpy()[0]
        
        for host, info in self.host_manager.hosts.items():
            idx = info['index']
            # Check compromise indicator (every 4th element starting at index 2)
            if idx * 4 + 2 < len(state_np):
                if state_np[idx * 4 + 2] > 0.1:  # Threshold
                    compromised.add(host)
        
        return compromised


# ============================================================================
# WORKFLOW COMPLIANCE FOR HOST ORDERS
# ============================================================================

class HostOrderCompliance:
    """
    Check if agent is following the host fixing order
    """
    
    def __init__(self, workflow: FixingOrderWorkflow):
        self.workflow = workflow
        self.action_history = []
        self.host_manager = HostFixingOrderWorkflow()
        
    def update(self, action: int, state: np.ndarray, timestep: int):
        """Record action for compliance checking"""
        # Determine which host this action targets
        target_host = self._action_to_host(action)
        self.action_history.append({
            'action': action,
            'host': target_host,
            'timestep': timestep,
            'compromised': self._get_compromised_from_state(state)
        })
    
    def _action_to_host(self, action: int) -> Optional[str]:
        """Map action ID to target host"""
        for host, actions in self.host_manager.host_to_actions.items():
            if action in actions.values():
                return host
        return None
    
    def _get_compromised_from_state(self, state: np.ndarray) -> Set[str]:
        """Extract compromised hosts from state"""
        compromised = set()
        for host, info in self.host_manager.hosts.items():
            idx = info['index']
            if idx * 4 + 2 < len(state) and state[idx * 4 + 2] > 0.1:
                compromised.add(host)
        return compromised
    
    def compute_compliance(self) -> float:
        """
        Compute how well the agent follows the host fixing order
        """
        if not self.action_history:
            return 0.0
        
        scores = []
        
        # Check if fixes follow priority order
        fix_sequence = []
        for record in self.action_history:
            if record['host'] and record['host'] in record['compromised']:
                if record['host'] not in fix_sequence:
                    fix_sequence.append(record['host'])
        
        # Compare fix sequence to workflow order
        if fix_sequence:
            order_score = self._compute_order_similarity(
                fix_sequence, 
                self.workflow.host_order
            )
            scores.append(order_score)
        
        # Check if higher priority hosts are fixed before lower priority
        priority_score = self._compute_priority_compliance()
        scores.append(priority_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_order_similarity(self, actual: List[str], expected: List[str]) -> float:
        """
        Compute similarity between actual and expected fixing order
        """
        score = 0.0
        
        # For each host in actual sequence, check if it appears before 
        # lower priority hosts
        for i, host in enumerate(actual):
            if host in expected:
                expected_pos = expected.index(host)
                # Check how many hosts that should come after are fixed before
                violations = 0
                for j in range(i):
                    other_host = actual[j]
                    if other_host in expected:
                        other_pos = expected.index(other_host)
                        if other_pos > expected_pos:
                            violations += 1
                
                # Score based on violations
                host_score = 1.0 - (violations / (i + 1)) if i > 0 else 1.0
                score += host_score
        
        return score / len(actual) if actual else 0.0
    
    def _compute_priority_compliance(self) -> float:
        """
        Check if high priority compromised hosts are addressed first
        """
        # Group actions by time windows
        time_windows = {}
        for record in self.action_history:
            window = record['timestep'] // 10
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(record)
        
        window_scores = []
        for window, records in time_windows.items():
            # Check if highest priority compromised host was targeted
            compromised_hosts = set()
            for record in records:
                compromised_hosts.update(record['compromised'])
            
            if compromised_hosts:
                # Find highest priority compromised host
                highest_priority_host = None
                highest_priority = float('inf')
                
                for host in compromised_hosts:
                    if host in self.workflow.priority:
                        if self.workflow.priority[host] < highest_priority:
                            highest_priority = self.workflow.priority[host]
                            highest_priority_host = host
                
                # Check if it was targeted
                targeted_hosts = [r['host'] for r in records if r['host']]
                if highest_priority_host in targeted_hosts:
                    window_scores.append(1.0)
                else:
                    window_scores.append(0.0)
        
        return np.mean(window_scores) if window_scores else 0.5


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_host_order_workflows():
    """
    Show how host fixing order workflows work
    """
    print("="*70)
    print("HOST FIXING ORDER WORKFLOWS")
    print("="*70)
    
    # Create workflow manager
    manager = HostFixingOrderWorkflow()
    
    # Show canonical workflows
    canonical = manager.create_canonical_workflows()
    
    print("\n1. CANONICAL HOST FIXING ORDERS:")
    print("-" * 40)
    for name, order in canonical.items():
        print(f"\n{name}:")
        print(f"  Order: {' → '.join(order[:5])} → ...")
    
    print("\n2. WORKFLOW EMBEDDING:")
    print("-" * 40)
    
    embedder = HostOrderEmbedding()
    
    # Compare embeddings of different orders
    for name, order in list(canonical.items())[:2]:
        embedding = embedder.order_to_embedding(order)
        features = embedder.order_to_features(order)
        
        print(f"\n{name}:")
        print(f"  Embedding (first 4): {embedding[:4]}")
        print(f"  Features:")
        print(f"    - User priority: {features[0]:.2f}")
        print(f"    - Enterprise priority: {features[1]:.2f}")
        print(f"    - Operational priority: {features[2]:.2f}")
        print(f"    - Clustering: {features[4]:.2f}")
    
    print("\n3. FOLLOWING THE ORDER:")
    print("-" * 40)
    
    # Create a workflow
    workflow = manager.create_workflow_from_order(canonical['critical_first'])
    
    # Simulate compromised hosts
    compromised = {'Op_Server0', 'User0', 'Enterprise1'}
    
    print(f"\nCompromised hosts: {compromised}")
    print(f"Workflow order: critical_first")
    
    # Get next host to fix
    next_host = workflow.get_next_host_to_fix(compromised)
    print(f"Next host to fix: {next_host}")
    
    # Get action for that host
    dummy_obs = np.random.randn(52)
    action = workflow.get_action_for_host(next_host, dummy_obs, 'restore')
    print(f"Action to take: {action} (restore {next_host})")


def explain_host_order_approach():
    """
    Explain the host fixing order approach
    """
    explanation = """
    
    HOST FIXING ORDER WORKFLOWS:
    ============================
    
    CONCEPT:
    - Workflow = Prioritized order for fixing compromised hosts
    - Even though red makes execution unpredictable, we follow a strategic order
    - Different orders represent different defense philosophies
    
    ADVANTAGES:
    1. INTERPRETABLE: "Fix operational server first, then enterprise, then users"
    2. STRATEGIC: Encodes defense priorities clearly
    3. FLEXIBLE: Adapts to actual compromises while maintaining priorities
    4. VERIFIABLE: Can check if agent follows the intended order
    
    CHALLENGES & SOLUTIONS:
    
    1. "Red agent makes it uncontrollable"
       → We don't control what gets compromised
       → But we control our response priority
       → Workflow guides which compromised host to fix first
    
    2. "Actual execution varies"
       → The order is a priority guide, not rigid sequence
       → If User0 and Enterprise0 are compromised:
         * Critical-first workflow → Fix Enterprise0
         * User-first workflow → Fix User0
    
    3. "How to embed permutations?"
       → Extract semantic features from order
       → Position-based encoding
       → Learn embeddings that preserve order similarity
    
    IMPLEMENTATION:
    1. Define host fixing order (e.g., critical → enterprise → users)
    2. Convert to embedding/features
    3. Policy learns to follow this priority
    4. When hosts are compromised, fix in workflow order
    5. Compliance checks if fixes follow priority
    
    The key: Workflow defines our RESPONSE STRATEGY, not what happens!
    """
    print(explanation)


if __name__ == "__main__":
    # Demonstrate the system
    demonstrate_host_order_workflows()
    
    # Explain the approach
    explain_host_order_approach()
