"""
Workflow search algorithm using Bayesian Optimization
"""

import numpy as np
from typing import List, Tuple, Dict
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from workflow_representation import WorkflowVector, WorkflowMilestone, PredefinedWorkflows
from workflow_executor import WorkflowExecutor, AdaptiveWorkflowExecutor

class WorkflowSearchSpace:
    """Defines the search space for workflows"""
    
    def __init__(self, dim=8, bounds=(-1, 1)):
        self.dim = dim
        self.bounds = bounds
        self.explored_workflows = []
        self.performance_history = []
        
    def sample_random(self) -> np.ndarray:
        """Sample random point in workflow space"""
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
    
    def sample_gaussian(self, mean: np.ndarray, std: float = 0.2) -> np.ndarray:
        """Sample around a point with Gaussian noise"""
        sample = mean + np.random.randn(self.dim) * std
        return np.clip(sample, self.bounds[0], self.bounds[1])
    
    def add_result(self, workflow_vector: np.ndarray, performance: float):
        """Store exploration result"""
        self.explored_workflows.append(workflow_vector)
        self.performance_history.append(performance)
        
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get best workflow found so far"""
        if not self.performance_history:
            return None, float('-inf')
        best_idx = np.argmax(self.performance_history)
        return self.explored_workflows[best_idx], self.performance_history[best_idx]

class BayesianWorkflowSearch:
    """Bayesian optimization for workflow search"""
    
    def __init__(self, search_space: WorkflowSearchSpace):
        self.search_space = search_space
        self.kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        self.beta = 2.0  # Exploration parameter for UCB
        
    def select_next_workflow(self, n_candidates=100) -> np.ndarray:
        """Select next workflow to evaluate using UCB"""
        if len(self.search_space.explored_workflows) < 5:
            # Initial random exploration
            return self.search_space.sample_random()
        
        # Fit GP model
        X = np.array(self.search_space.explored_workflows)
        y = np.array(self.search_space.performance_history)
        self.gp.fit(X, y)
        
        # Generate candidates
        candidates = []
        for _ in range(n_candidates):
            if np.random.random() < 0.5:
                # Explore around best
                best_workflow, _ = self.search_space.get_best()
                candidate = self.search_space.sample_gaussian(best_workflow)
            else:
                # Random exploration
                candidate = self.search_space.sample_random()
            candidates.append(candidate)
        
        # Compute UCB for each candidate
        candidates = np.array(candidates)
        mean, std = self.gp.predict(candidates, return_std=True)
        ucb = mean + self.beta * std
        
        # Select best candidate
        best_idx = np.argmax(ucb)
        return candidates[best_idx]

class WorkflowRLAgent:
    """Main agent that combines workflow search with execution"""
    
    def __init__(self, env, workflow_dim=8, use_adaptive=True):
        self.env = env
        self.workflow_dim = workflow_dim
        
        # Initialize components
        self.search_space = WorkflowSearchSpace(dim=workflow_dim)
        self.searcher = BayesianWorkflowSearch(self.search_space)
        
        if use_adaptive:
            self.executor = AdaptiveWorkflowExecutor()
        else:
            self.executor = WorkflowExecutor()
            
        self.current_workflow_vector = None
        self.current_workflow = None
        
        # Initialize with known good workflows
        self._seed_with_known_workflows()
        
    def _seed_with_known_workflows(self):
        """Seed search with workflows from optimal policies"""
        # Convert known workflows to vectors
        known_workflows = [
            PredefinedWorkflows.get_bline_workflow(),
            PredefinedWorkflows.get_meander_workflow(),
            PredefinedWorkflows.get_hybrid_workflow()
        ]
        
        for workflow in known_workflows:
            vector = WorkflowVector(self.workflow_dim)
            vector.from_milestones(workflow)
            self.search_space.explored_workflows.append(vector.vector)
            # Initialize with neutral performance
            self.search_space.performance_history.append(0.0)
    
    def select_workflow(self, episode_num: int) -> np.ndarray:
        """Select workflow for current episode"""
        if episode_num < 3:
            # Use predefined workflows for initial episodes
            idx = episode_num % 3
            self.current_workflow_vector = self.search_space.explored_workflows[idx]
        else:
            # Use Bayesian optimization
            self.current_workflow_vector = self.searcher.select_next_workflow()
        
        # Convert vector to milestone sequence
        wf_vector = WorkflowVector(self.workflow_dim)
        wf_vector.vector = self.current_workflow_vector
        self.current_workflow = wf_vector.to_milestones(num_steps=30)
        
        # Set workflow in executor
        self.executor.set_workflow(self.current_workflow)
        
        return self.current_workflow_vector
    
    def get_action(self, observation: np.ndarray) -> int:
        """Get action for current step"""
        return self.executor.get_action(observation)
    
    def update_performance(self, episode_reward: float):
        """Update search with episode performance"""
        if self.current_workflow_vector is not None:
            self.search_space.add_result(
                self.current_workflow_vector,
                episode_reward
            )
    
    def reset(self):
        """Reset for new episode"""
        self.executor.reset()
        
    def get_best_workflow(self) -> Tuple[np.ndarray, float]:
        """Get best workflow found"""
        return self.search_space.get_best()
