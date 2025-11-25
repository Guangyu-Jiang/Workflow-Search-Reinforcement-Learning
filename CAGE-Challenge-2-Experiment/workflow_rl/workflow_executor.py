"""
Workflow executor that maps high-level workflows to CAGE2 actions
"""

import numpy as np
from typing import List, Dict, Optional
from workflow_representation import WorkflowMilestone, WorkflowPrimitive

class WorkflowExecutor:
    """Executes workflow milestones as CAGE2 actions"""
    
    def __init__(self):
        # Map workflow primitives to action indices in the reduced action space
        # These map to indices in the action_space list from train.py
        self.action_mapping = {
            # Restore actions (indices 0-7 in action_space)
            'restore_enterprise': [0, 1, 2],  # Restore Enterprise 0,1,2
            'restore_operational': [3],  # Restore Op_Server0
            'restore_user': [16, 17, 18, 19],  # Restore User 1-4
            'restore_defender': [20],  # Restore Defender
            
            # Analyse actions (indices 4-11 in action_space)
            'analyse_enterprise': [4, 5, 6],  # Analyse Enterprise 0,1,2
            'analyse_operational': [7],  # Analyse Op_Server0
            'analyse_user': [12, 13, 14, 15],  # Analyse User 1-4
            'analyse_defender': [21],  # Analyse Defender
            
            # Remove actions (indices 8-15, 20-24 in action_space)
            'remove_enterprise': [8, 9, 10],  # Remove Enterprise 0,1,2
            'remove_operational': [11],  # Remove Op_Server0
            'remove_user': [23, 24, 25, 26],  # Remove User 1-4
            'remove_defender': [22],  # Remove Defender
            
            # For decoys, we'll just use analyse actions as placeholders
            # since we don't have decoy actions in the basic action space
            'decoy_enterprise': [4, 5, 6],  # Use analyse as placeholder
            'decoy_operational': [7],
            'decoy_user': [12, 13, 14, 15],
            'decoy_defender': [21],
        }
        
        self.current_milestone_idx = 0
        self.current_action_idx = 0
        self.workflow = []
        self.action_history = []
        
    def set_workflow(self, workflow: List[WorkflowMilestone]):
        """Set the workflow to execute"""
        self.workflow = workflow
        self.current_milestone_idx = 0
        self.current_action_idx = 0
        self.action_history = []
        
    def get_action(self, observation: np.ndarray) -> int:
        """Get next action based on current workflow milestone"""
        if self.current_milestone_idx >= len(self.workflow):
            return 0  # Sleep if workflow complete
            
        milestone = self.workflow[self.current_milestone_idx]
        
        # Get appropriate action for this milestone
        action = self._milestone_to_action(milestone, observation)
        
        # Track progress
        self.action_history.append(action)
        self.current_action_idx += 1
        
        # Check if milestone duration reached
        if self.current_action_idx >= milestone.duration:
            self.current_milestone_idx += 1
            self.current_action_idx = 0
            
        return action
    
    def _milestone_to_action(self, milestone: WorkflowMilestone, 
                            observation: np.ndarray) -> int:
        """Convert milestone to specific CAGE2 action"""
        primitive = milestone.primitive
        subnet = milestone.target_subnet
        
        # Build action key
        if primitive == WorkflowPrimitive.FORTIFY:
            action_key = f'decoy_{subnet}'
        elif primitive == WorkflowPrimitive.ANALYSE:
            action_key = f'analyse_{subnet}'
        elif primitive == WorkflowPrimitive.REMOVE:
            action_key = f'remove_{subnet}'
        elif primitive == WorkflowPrimitive.RESTORE:
            action_key = f'restore_{subnet}'
        else:  # WAIT
            return 0  # Sleep action
            
        # Get available actions for this category
        if action_key in self.action_mapping:
            actions = self.action_mapping[action_key]
            # Select action based on observation or round-robin
            return self._select_action_from_list(actions, observation)
        
        return 0  # Default to sleep
    
    def _select_action_from_list(self, actions: List[int], 
                                 observation: np.ndarray) -> int:
        """Select specific action from list based on observation"""
        # Simple heuristic: prioritize hosts with detected activity
        # In practice, this would use the observation to make smart choices
        
        # For now, cycle through available actions
        idx = self.current_action_idx % len(actions)
        return actions[idx]
    
    def reset(self):
        """Reset executor state"""
        self.current_milestone_idx = 0
        self.current_action_idx = 0
        self.action_history = []

class AdaptiveWorkflowExecutor(WorkflowExecutor):
    """Executor that can adapt workflow based on observations"""
    
    def __init__(self):
        super().__init__()
        self.adaptation_enabled = True
        self.threat_threshold = 0.5
        
    def get_action(self, observation: np.ndarray) -> int:
        """Get action with potential workflow adaptation"""
        if self.adaptation_enabled:
            self._adapt_workflow(observation)
            
        return super().get_action(observation)
    
    def _adapt_workflow(self, observation: np.ndarray):
        """Adapt workflow based on current observation"""
        threat_level = self._assess_threat(observation)
        
        if threat_level > self.threat_threshold:
            # Switch to more aggressive response
            self._insert_emergency_milestone()
            
    def _assess_threat(self, observation: np.ndarray) -> float:
        """Assess threat level from observation"""
        # Count suspicious activities (simplified)
        # Observation indices 0,4,8,12,... are activity indicators
        activity_count = sum(observation[i] for i in range(0, len(observation), 4) 
                            if i < len(observation))
        
        # Normalize to [0,1]
        threat_level = min(activity_count / 5.0, 1.0)
        return threat_level
    
    def _insert_emergency_milestone(self):
        """Insert emergency response milestone"""
        if self.current_milestone_idx < len(self.workflow):
            # Insert a restore milestone for critical hosts
            emergency = WorkflowMilestone(
                WorkflowPrimitive.RESTORE,
                'enterprise',
                duration=1,
                priority=2.0
            )
            self.workflow.insert(self.current_milestone_idx + 1, emergency)
