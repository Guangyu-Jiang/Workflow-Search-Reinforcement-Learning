"""
Workflow representation for CAGE2
Based on analysis of optimal policies (bline and meander models)
"""

import numpy as np
from typing import List, Dict, Tuple
from enum import Enum

class WorkflowPrimitive(Enum):
    """Basic workflow building blocks identified from optimal policies"""
    FORTIFY = "fortify"      # Deploy decoys
    ANALYSE = "analyse"      # Gather information
    REMOVE = "remove"        # Remove user-level compromise
    RESTORE = "restore"      # Restore compromised hosts
    WAIT = "wait"           # Monitor/Sleep

class WorkflowMilestone:
    """Represents a milestone in the workflow"""
    def __init__(self, primitive: WorkflowPrimitive, target_subnet: str, 
                 duration: int = 1, priority: float = 1.0):
        self.primitive = primitive
        self.target_subnet = target_subnet  # 'user', 'enterprise', 'operational'
        self.duration = duration  # How many steps to execute
        self.priority = priority  # Importance weight
        
    def __repr__(self):
        return f"{self.primitive.value}_{self.target_subnet}_{self.duration}"

class WorkflowVector:
    """
    Low-dimensional continuous representation of workflows
    Based on patterns observed in optimal policies
    """
    def __init__(self, dim=8):
        self.dim = dim
        self.vector = np.zeros(dim)
        
        # Dimension meanings (from analysis):
        # 0: Fortification timing (0=never, 0.5=reactive, 1=proactive)
        # 1: Fortification intensity (0=minimal, 1=maximal)
        # 2: Analysis frequency (0=minimal, 1=continuous)
        # 3: Remove vs Restore preference (-1=remove only, 1=restore only)
        # 4: Response speed (0=delayed, 1=immediate)
        # 5: User subnet focus (0=ignore, 1=prioritize)
        # 6: Enterprise subnet focus (0=ignore, 1=prioritize)
        # 7: Operational subnet focus (0=ignore, 1=prioritize)
        
    def from_milestones(self, milestones: List[WorkflowMilestone]):
        """Convert milestone sequence to vector representation"""
        # Reset vector
        self.vector = np.zeros(self.dim)
        
        if not milestones:
            return self.vector
            
        # Calculate features from milestone sequence
        total_steps = len(milestones)
        fortify_steps = sum(1 for m in milestones if m.primitive == WorkflowPrimitive.FORTIFY)
        analyse_steps = sum(1 for m in milestones if m.primitive == WorkflowPrimitive.ANALYSE)
        remove_steps = sum(1 for m in milestones if m.primitive == WorkflowPrimitive.REMOVE)
        restore_steps = sum(1 for m in milestones if m.primitive == WorkflowPrimitive.RESTORE)
        
        # Feature 0: Fortification timing (early vs late)
        if fortify_steps > 0:
            first_fortify = next(i for i, m in enumerate(milestones) 
                               if m.primitive == WorkflowPrimitive.FORTIFY)
            self.vector[0] = 1.0 - (first_fortify / max(total_steps, 1))
        
        # Feature 1: Fortification intensity
        self.vector[1] = fortify_steps / max(total_steps, 1)
        
        # Feature 2: Analysis frequency
        self.vector[2] = analyse_steps / max(total_steps, 1)
        
        # Feature 3: Remove vs Restore preference
        if remove_steps + restore_steps > 0:
            self.vector[3] = (restore_steps - remove_steps) / (remove_steps + restore_steps)
        
        # Feature 4: Response speed (based on action density)
        wait_steps = sum(1 for m in milestones if m.primitive == WorkflowPrimitive.WAIT)
        self.vector[4] = 1.0 - (wait_steps / max(total_steps, 1))
        
        # Features 5-7: Subnet focus
        subnet_counts = {'user': 0, 'enterprise': 0, 'operational': 0}
        for m in milestones:
            if m.target_subnet in subnet_counts:
                subnet_counts[m.target_subnet] += 1
        
        total_subnet_actions = sum(subnet_counts.values())
        if total_subnet_actions > 0:
            self.vector[5] = subnet_counts['user'] / total_subnet_actions
            self.vector[6] = subnet_counts['enterprise'] / total_subnet_actions
            self.vector[7] = subnet_counts['operational'] / total_subnet_actions
            
        return self.vector
    
    def to_milestones(self, num_steps=30) -> List[WorkflowMilestone]:
        """Convert vector back to milestone sequence"""
        milestones = []
        
        # Decode vector to workflow parameters
        fortify_ratio = self.vector[1]
        analyse_ratio = self.vector[2]
        remove_restore_pref = self.vector[3]
        
        # Calculate number of each action type
        fortify_steps = int(fortify_ratio * num_steps)
        analyse_steps = int(analyse_ratio * num_steps)
        
        # Determine cleanup strategy
        if remove_restore_pref < 0:
            remove_steps = int((1 - fortify_ratio - analyse_ratio) * num_steps * 0.7)
            restore_steps = int((1 - fortify_ratio - analyse_ratio) * num_steps * 0.3)
        else:
            remove_steps = int((1 - fortify_ratio - analyse_ratio) * num_steps * 0.3)
            restore_steps = int((1 - fortify_ratio - analyse_ratio) * num_steps * 0.7)
        
        # Create milestones based on timing preference
        if self.vector[0] > 0.5:  # Early fortification
            for _ in range(fortify_steps):
                subnet = self._select_subnet()
                milestones.append(WorkflowMilestone(WorkflowPrimitive.FORTIFY, subnet))
        
        # Add analysis milestones
        for _ in range(analyse_steps):
            subnet = self._select_subnet()
            milestones.append(WorkflowMilestone(WorkflowPrimitive.ANALYSE, subnet))
        
        # Add cleanup milestones
        for _ in range(remove_steps):
            subnet = self._select_subnet()
            milestones.append(WorkflowMilestone(WorkflowPrimitive.REMOVE, subnet))
            
        for _ in range(restore_steps):
            subnet = self._select_subnet()
            milestones.append(WorkflowMilestone(WorkflowPrimitive.RESTORE, subnet))
        
        # Add late fortification if needed
        if self.vector[0] <= 0.5:
            for _ in range(fortify_steps):
                subnet = self._select_subnet()
                milestones.append(WorkflowMilestone(WorkflowPrimitive.FORTIFY, subnet))
        
        return milestones[:num_steps]
    
    def _select_subnet(self) -> str:
        """Select subnet based on focus weights"""
        weights = self.vector[5:8]
        if np.sum(weights) == 0:
            weights = np.ones(3) / 3
        else:
            weights = weights / np.sum(weights)
            
        return np.random.choice(['user', 'enterprise', 'operational'], p=weights)

class PredefinedWorkflows:
    """Workflows extracted from optimal policy analysis"""
    
    @staticmethod
    def get_bline_workflow() -> List[WorkflowMilestone]:
        """Reactive defense workflow from bline model"""
        return [
            WorkflowMilestone(WorkflowPrimitive.ANALYSE, 'user', 2),
            WorkflowMilestone(WorkflowPrimitive.REMOVE, 'user', 5),
            WorkflowMilestone(WorkflowPrimitive.FORTIFY, 'enterprise', 1),
            WorkflowMilestone(WorkflowPrimitive.REMOVE, 'user', 5),
            WorkflowMilestone(WorkflowPrimitive.RESTORE, 'enterprise', 2),
            WorkflowMilestone(WorkflowPrimitive.REMOVE, 'user', 10),
        ]
    
    @staticmethod
    def get_meander_workflow() -> List[WorkflowMilestone]:
        """Proactive fortification workflow from meander model"""
        return [
            WorkflowMilestone(WorkflowPrimitive.FORTIFY, 'enterprise', 3),
            WorkflowMilestone(WorkflowPrimitive.FORTIFY, 'operational', 3),
            WorkflowMilestone(WorkflowPrimitive.ANALYSE, 'user', 2),
            WorkflowMilestone(WorkflowPrimitive.RESTORE, 'user', 1),
            WorkflowMilestone(WorkflowPrimitive.ANALYSE, 'enterprise', 3),
            WorkflowMilestone(WorkflowPrimitive.RESTORE, 'user', 1),
        ]
    
    @staticmethod
    def get_hybrid_workflow() -> List[WorkflowMilestone]:
        """Hybrid workflow combining best of both"""
        return [
            WorkflowMilestone(WorkflowPrimitive.FORTIFY, 'enterprise', 2),
            WorkflowMilestone(WorkflowPrimitive.FORTIFY, 'operational', 1),
            WorkflowMilestone(WorkflowPrimitive.ANALYSE, 'user', 1),
            WorkflowMilestone(WorkflowPrimitive.REMOVE, 'user', 2),
            WorkflowMilestone(WorkflowPrimitive.ANALYSE, 'enterprise', 1),
            WorkflowMilestone(WorkflowPrimitive.RESTORE, 'enterprise', 1),
        ]
