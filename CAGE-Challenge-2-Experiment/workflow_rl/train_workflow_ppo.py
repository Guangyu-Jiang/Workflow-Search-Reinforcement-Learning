#!/usr/bin/env python3
"""
Workflow Search + PPO Integration for CAGE2
Combines high-level workflow search with low-level PPO policy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import inspect
from typing import Dict, List
import json

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.PPOAgent import PPOAgent

from workflow_representation import WorkflowVector, PredefinedWorkflows
from workflow_search import WorkflowSearchSpace, BayesianWorkflowSearch

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WorkflowGuidedPPOAgent(PPOAgent):
    """
    PPO agent that is guided by high-level workflows
    The workflow biases action selection while PPO handles low-level decisions
    """
    
    def __init__(self, workflow_vector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workflow_vector = workflow_vector
        self.workflow_params = self._decode_workflow(workflow_vector)
        self.step_count = 0
        self.max_steps = 100
        
    def _decode_workflow(self, vector):
        """Decode workflow vector into parameters"""
        return {
            'fortify_timing': vector[0],      # 0=late, 1=early
            'fortify_intensity': vector[1],   # 0=none, 1=heavy
            'analyse_frequency': vector[2],   # 0=minimal, 1=continuous
            'remove_restore_pref': vector[3], # -1=remove, 1=restore
            'response_speed': vector[4],      # 0=delayed, 1=immediate
            'user_focus': vector[5],          # 0-1
            'enterprise_focus': vector[6],    # 0-1
            'operational_focus': vector[7],   # 0-1
        }
    
    def get_action(self, observation, action_space=None):
        """Get action with workflow guidance"""
        # Get base PPO action
        base_action = super().get_action(observation, action_space)
        
        # Apply workflow bias
        biased_action = self._apply_workflow_bias(base_action, observation)
        
        self.step_count += 1
        return biased_action
    
    def _apply_workflow_bias(self, base_action, observation):
        """Bias action selection based on workflow parameters"""
        
        # Early fortification phase
        if self.step_count < 10 and self.workflow_params['fortify_timing'] > 0.7:
            # Prioritize decoy actions in early phase
            if np.random.random() < self.workflow_params['fortify_intensity']:
                # Select a decoy action
                decoy_hosts = [1000, 1001, 1002, 1008]  # Enterprise and OpServer
                return np.random.choice(decoy_hosts)
        
        # Analyse phase based on frequency
        if np.random.random() < self.workflow_params['analyse_frequency'] * 0.3:
            # Select analyse action based on subnet focus
            if np.random.random() < self.workflow_params['enterprise_focus']:
                return np.random.choice([3, 4, 5])  # Analyse enterprise
            elif np.random.random() < self.workflow_params['operational_focus']:
                return 9  # Analyse OpServer
            else:
                return np.random.choice([11, 12, 13, 14])  # Analyse users
        
        # Compromise response
        if self._detect_compromise(observation):
            if self.workflow_params['remove_restore_pref'] > 0:
                # Prefer restore
                if self.workflow_params['enterprise_focus'] > 0.5:
                    return np.random.choice([133, 134, 135])  # Restore enterprise
                else:
                    return np.random.choice([141, 142, 143, 144])  # Restore users
            else:
                # Prefer remove
                if self.workflow_params['enterprise_focus'] > 0.5:
                    return np.random.choice([16, 17, 18])  # Remove enterprise
                else:
                    return np.random.choice([24, 25, 26, 27])  # Remove users
        
        # Otherwise use base PPO action
        return base_action
    
    def _detect_compromise(self, observation):
        """Simple compromise detection from observation"""
        # Check for suspicious activity in observation
        # Indices 0,4,8,12... indicate activity
        for i in range(0, min(len(observation), 52), 4):
            if observation[i] > 0 and observation[i+2] > 0:  # Activity + compromise
                return True
        return False
    
    def reset(self):
        """Reset for new episode"""
        self.step_count = 0
        self.end_episode()

def train_workflow_ppo(num_workflows=20, episodes_per_workflow=10, max_steps=50):
    """
    Train PPO with workflow search
    """
    print("="*80)
    print("WORKFLOW-GUIDED PPO TRAINING FOR CAGE2")
    print("="*80)
    
    # Initialize workflow search
    search_space = WorkflowSearchSpace(dim=8)
    searcher = BayesianWorkflowSearch(search_space)
    
    # Define action space (from train.py)
    action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    action_space += [11, 12, 13, 14]  # analyse user hosts
    action_space += [141, 142, 143, 144]  # restore user hosts
    action_space += [132]  # restore defender
    action_space += [2]  # analyse defender
    action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts
    
    # Initialize with known good workflows
    initial_workflows = [
        np.array([0.2, 0.1, 0.2, -0.5, 0.8, 0.6, 0.3, 0.1]),  # Bline-like
        np.array([0.9, 0.4, 0.5, 0.5, 0.9, 0.3, 0.4, 0.3]),   # Meander-like
        np.array([0.5, 0.3, 0.4, 0.0, 0.7, 0.4, 0.4, 0.2])    # Hybrid
    ]
    
    for wf in initial_workflows:
        search_space.explored_workflows.append(wf)
        search_space.performance_history.append(0.0)
    
    results = {
        'workflows': [],
        'performances': [],
        'best_workflow': None,
        'best_performance': float('-inf')
    }
    
    # Main training loop
    for workflow_idx in range(num_workflows):
        print(f"\n--- Workflow {workflow_idx + 1}/{num_workflows} ---")
        
        # Select next workflow to evaluate
        if workflow_idx < 3:
            workflow_vector = initial_workflows[workflow_idx]
        else:
            workflow_vector = searcher.select_next_workflow()
        
        print(f"Testing workflow: [{workflow_vector[0]:.2f}, {workflow_vector[1]:.2f}, "
              f"{workflow_vector[2]:.2f}, {workflow_vector[3]:.2f}, ...]")
        
        # Create PPO agent with this workflow
        agent = WorkflowGuidedPPOAgent(
            workflow_vector=workflow_vector,
            input_dims=52,
            action_space=action_space,
            lr=0.002,
            betas=[0.9, 0.990],
            gamma=0.99,
            K_epochs=4,
            eps_clip=0.2,
            start_actions=[]  # No fixed start actions
        )
        
        # Train PPO with this workflow
        workflow_rewards = []
        
        for episode in range(episodes_per_workflow):
            # Alternate between red agents
            red_agent = B_lineAgent if episode % 2 == 0 else RedMeanderAgent
            
            # Create environment
            cyborg = CybORG(PATH, 'sim', agents={'Red': red_agent})
            env = ChallengeWrapper2(env=cyborg, agent_name="Blue")
            
            # Run episode
            state = env.reset()
            agent.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store for PPO training
                agent.store(reward, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # PPO update every few episodes
            if (episode + 1) % 5 == 0:
                agent.train()
                agent.clear_memory()
            
            workflow_rewards.append(episode_reward)
            
            if episode % 5 == 0:
                print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        # Evaluate workflow performance
        avg_performance = np.mean(workflow_rewards)
        std_performance = np.std(workflow_rewards)
        
        print(f"Workflow performance: {avg_performance:.2f} ± {std_performance:.2f}")
        
        # Update search
        search_space.add_result(workflow_vector, avg_performance)
        
        # Track results
        results['workflows'].append(workflow_vector.tolist())
        results['performances'].append(avg_performance)
        
        if avg_performance > results['best_performance']:
            results['best_performance'] = avg_performance
            results['best_workflow'] = workflow_vector
            
            # Save best model
            torch.save(agent.policy.state_dict(), 'best_workflow_ppo_model.pth')
            print(f"New best workflow! Performance: {avg_performance:.2f}")
    
    # Final results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    print(f"\nBest workflow found:")
    print(f"  Vector: {results['best_workflow']}")
    print(f"  Performance: {results['best_performance']:.2f}")
    
    # Interpret best workflow
    params = {
        'fortify_timing': 'Early' if results['best_workflow'][0] > 0.5 else 'Late',
        'fortify_intensity': 'Heavy' if results['best_workflow'][1] > 0.5 else 'Light',
        'analyse_frequency': 'High' if results['best_workflow'][2] > 0.5 else 'Low',
        'strategy': 'Restore' if results['best_workflow'][3] > 0 else 'Remove',
        'response': 'Immediate' if results['best_workflow'][4] > 0.5 else 'Delayed'
    }
    
    print("\nWorkflow interpretation:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Save results
    with open('workflow_ppo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to workflow_ppo_results.json")
    
    return results

def compare_with_baseline():
    """Compare workflow-guided PPO with standard PPO"""
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE PPO")
    print("="*80)
    
    # Load baseline PPO model
    action_space = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22,
                   11, 12, 13, 14, 141, 142, 143, 144, 132, 2, 15, 24, 25, 26, 27]
    
    baseline_agent = PPOAgent(
        input_dims=52,
        action_space=action_space,
        restore=True,
        ckpt='/home/ubuntu/CAGE2/-cyborg-cage-2/Models/bline/model.pth',
        deterministic=True,
        training=False
    )
    
    # Best workflow from our search (example)
    best_workflow = np.array([0.8, 0.35, 0.4, 0.3, 0.85, 0.35, 0.45, 0.2])
    workflow_agent = WorkflowGuidedPPOAgent(
        workflow_vector=best_workflow,
        input_dims=52,
        action_space=action_space,
        lr=0.002,
        betas=[0.9, 0.990],
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2
    )
    
    print("\nEvaluating on 10 episodes each...")
    
    for agent_name, agent in [("Baseline PPO", baseline_agent), 
                              ("Workflow-Guided PPO", workflow_agent)]:
        rewards = []
        
        for episode in range(10):
            cyborg = CybORG(PATH, 'sim', agents={'Red': B_lineAgent})
            env = ChallengeWrapper2(env=cyborg, agent_name="Blue")
            
            state = env.reset()
            episode_reward = 0
            
            for step in range(50):
                action = agent.get_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
            agent.end_episode()
        
        print(f"\n{agent_name}:")
        print(f"  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

if __name__ == "__main__":
    # Run workflow-guided PPO training (reduced for demo)
    results = train_workflow_ppo(
        num_workflows=5,        # Test 5 different workflows
        episodes_per_workflow=5, # 5 episodes per workflow
        max_steps=30            # 30 steps per episode
    )
    
    # Compare with baseline
    compare_with_baseline()
