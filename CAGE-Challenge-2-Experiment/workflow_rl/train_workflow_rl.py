#!/usr/bin/env python3
"""
Main training script for Workflow-Conditioned RL in CAGE2
Combines GP-UCB workflow search with PPO execution
"""

import os
import sys
import inspect
import numpy as np
import torch
import random
from typing import List, Tuple, Dict
import json
from datetime import datetime

# Add paths
sys.path.insert(0, '/home/ubuntu/CAGE2/cage-challenge-2')
sys.path.insert(0, '/home/ubuntu/CAGE2/-cyborg-cage-2')

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

from workflow_rl.workflow_conditioned_ppo import WorkflowConditionedPPO
from workflow_rl.unit_type_workflow import UnitTypeWorkflow
from workflow_rl.gp_ucb_workflow_search import GPUCBWorkflowSearch


class WorkflowRLTrainer:
    """
    Main training class for workflow-conditioned RL
    """
    
    def __init__(self, 
                 scenario_path: str,
                 red_agent_type=B_lineAgent,
                 max_steps: int = 100,
                 eval_episodes: int = 5,
                 train_episodes: int = 50,
                 alignment_alpha: float = 0.1,
                 alignment_beta: float = 0.2,
                 checkpoint_dir: str = "workflow_rl_checkpoints"):
        
        self.scenario_path = scenario_path
        self.red_agent_type = red_agent_type
        self.max_steps = max_steps
        self.eval_episodes = eval_episodes
        self.train_episodes = train_episodes
        self.alignment_alpha = alignment_alpha
        self.alignment_beta = alignment_beta
        
        # Create checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize components
        self.workflow_manager = UnitTypeWorkflow()
        self.gp_search = GPUCBWorkflowSearch(beta=2.0)
        
        # Training history
        self.training_history = []
    
    def create_env(self):
        """Create CAGE2 environment"""
        cyborg = CybORG(self.scenario_path, 'sim', agents={'Red': self.red_agent_type})
        env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
        return env, cyborg
    
    def train_workflow(self, workflow_order: List[str], workflow_vector: np.ndarray,
                      workflow_id: int, compliance_threshold: float = 0.95,
                      min_episodes: int = 10) -> Tuple[float, float, int]:
        """
        Train PPO agent with specific workflow, with early stopping based on compliance
        
        Args:
            workflow_order: Priority order of unit types
            workflow_vector: Workflow embedding vector
            workflow_id: ID for saving checkpoint
            compliance_threshold: Stop training when compliance rate reaches this (default 0.95)
            min_episodes: Minimum episodes before considering early stopping (default 10)
        
        Returns:
            (average_reward, compliance_rate, episodes_trained)
        """
        
        print(f"\nTraining with workflow: {' → '.join(workflow_order)}")
        
        # Create agent
        env, _ = self.create_env()
        input_dims = env.observation_space.shape[0]
        
        agent = WorkflowConditionedPPO(
            input_dims=input_dims,
            workflow=workflow_vector,
            alignment_alpha=self.alignment_alpha,
            alignment_beta=self.alignment_beta,
            training=True,
            deterministic=False
        )
        
        # Training metrics
        episode_rewards = []
        episode_compliances = []
        early_stopped = False
        actual_episodes = 0
        
        # Training loop with early stopping
        for episode in range(self.train_episodes):
            env, cyborg = self.create_env()
            obs = env.reset()
            
            episode_reward = 0
            episode_env_reward = 0
            episode_align_reward = 0
            compliant_actions = 0
            total_fix_actions = 0
            
            for step in range(self.max_steps):
                # Get action from agent
                action = agent.get_action(obs)
                
                # Execute action
                next_obs, env_reward, done, info = env.step(action)
                
                # Get true state for alignment check
                true_state = cyborg.get_agent_state('True')
                
                # Compute alignment reward
                align_reward = agent.compute_alignment_reward(
                    action, true_state, workflow_order
                )
                
                # Track compliance
                if align_reward != 0:  # Fix action
                    total_fix_actions += 1
                    if align_reward > 0:
                        compliant_actions += 1
                
                # Total reward
                total_reward = env_reward + align_reward
                
                # Update agent
                agent.update(total_reward, done)
                
                # Track rewards
                episode_reward += total_reward
                episode_env_reward += env_reward
                episode_align_reward += align_reward
                
                obs = next_obs
                if done:
                    break
            
            # Episode complete
            agent.end_episode()
            actual_episodes = episode + 1
            
            # Compute compliance rate
            compliance_rate = compliant_actions / total_fix_actions if total_fix_actions > 0 else 1.0
            
            episode_rewards.append(episode_env_reward)  # Use env reward for evaluation
            episode_compliances.append(compliance_rate)
            
            # Progress reporting
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_compliance = np.mean(episode_compliances[-10:])
                print(f"  Episode {episode}: Avg Reward={avg_reward:.2f}, "
                      f"Compliance={avg_compliance:.2%}")
            
            # Early stopping check
            if episode >= min_episodes - 1:  # After minimum episodes
                # Calculate rolling average compliance (last 5 episodes)
                recent_compliance = np.mean(episode_compliances[-5:])
                
                if recent_compliance >= compliance_threshold:
                    print(f"  Early stopping at episode {episode + 1}: "
                          f"Compliance {recent_compliance:.2%} >= {compliance_threshold:.2%}")
                    early_stopped = True
                    break
        
        # Save trained agent
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"workflow_{workflow_id}_agent.pth"
        )
        agent.save(checkpoint_path)
        
        # If early stopped, immediately evaluate on pure environment reward
        if early_stopped:
            print(f"  Evaluating early-stopped agent (trained for {actual_episodes} episodes)")
            # Use more evaluation episodes for better estimate
            eval_reward, eval_compliance = self.evaluate_pure_performance(
                agent, workflow_order, episodes=10
            )
            final_avg_reward = eval_reward
            final_avg_compliance = eval_compliance
        else:
            # Use training performance if completed all episodes
            final_avg_reward = np.mean(episode_rewards[-10:])
            final_avg_compliance = np.mean(episode_compliances[-10:])
        
        print(f"  Final performance: Reward={final_avg_reward:.2f}, "
              f"Compliance={final_avg_compliance:.2%}, Episodes={actual_episodes}")
        
        return final_avg_reward, final_avg_compliance, actual_episodes
    
    def evaluate_pure_performance(self, agent, workflow_order: List[str], 
                                  episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate agent on pure environment reward (no alignment rewards)
        Used for early-stopped agents to get unbiased performance estimate
        
        Args:
            agent: Trained PPO agent
            workflow_order: Workflow being evaluated
            episodes: Number of evaluation episodes
            
        Returns:
            (average_env_reward, average_compliance_rate)
        """
        eval_rewards = []
        eval_compliances = []
        
        for episode in range(episodes):
            env, cyborg = self.create_env()
            obs = env.reset()
            
            episode_reward = 0
            compliant_actions = 0
            total_fix_actions = 0
            prev_true_state = None
            
            for step in range(self.max_steps):
                # Get action (deterministic)
                action = agent.get_action(obs)
                
                # Execute action
                next_obs, env_reward, done, info = env.step(action)
                
                # Get true state for compliance tracking only (not for reward)
                true_state = cyborg.get_agent_state('True')
                
                # Track compliance (but don't add to reward)
                if prev_true_state is not None:
                    align_reward = agent.compute_alignment_reward(
                        action, true_state, workflow_order
                    )
                    if align_reward != 0:  # Fix action
                        total_fix_actions += 1
                        if align_reward > 0:
                            compliant_actions += 1
                
                # Only track environment reward
                episode_reward += env_reward
                
                obs = next_obs
                prev_true_state = true_state
                if done:
                    break
            
            compliance_rate = compliant_actions / total_fix_actions if total_fix_actions > 0 else 1.0
            eval_rewards.append(episode_reward)
            eval_compliances.append(compliance_rate)
        
        return np.mean(eval_rewards), np.mean(eval_compliances)
    
    def evaluate_workflow(self, workflow_order: List[str], workflow_vector: np.ndarray,
                         checkpoint_path: str) -> Tuple[float, float]:
        """
        Evaluate trained agent
        
        Returns:
            (average_reward, compliance_rate)
        """
        
        # Load trained agent
        env, _ = self.create_env()
        input_dims = env.observation_space.shape[0]
        
        agent = WorkflowConditionedPPO(
            input_dims=input_dims,
            workflow=workflow_vector,
            training=False,
            deterministic=True  # Deterministic for evaluation
        )
        agent.load(checkpoint_path)
        
        # Evaluation metrics
        eval_rewards = []
        eval_compliances = []
        
        for episode in range(self.eval_episodes):
            env, cyborg = self.create_env()
            obs = env.reset()
            
            episode_reward = 0
            compliant_actions = 0
            total_fix_actions = 0
            
            for step in range(self.max_steps):
                action = agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                
                # Check compliance
                true_state = cyborg.get_agent_state('True')
                align_reward = agent.compute_alignment_reward(
                    action, true_state, workflow_order
                )
                
                if align_reward != 0:
                    total_fix_actions += 1
                    if align_reward > 0:
                        compliant_actions += 1
                
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            compliance_rate = compliant_actions / total_fix_actions if total_fix_actions > 0 else 1.0
            eval_rewards.append(episode_reward)
            eval_compliances.append(compliance_rate)
        
        return np.mean(eval_rewards), np.mean(eval_compliances)
    
    def run_workflow_search(self, n_iterations: int = 20):
        """
        Main workflow search loop
        """
        
        print("="*70)
        print("STARTING WORKFLOW-CONDITIONED RL TRAINING")
        print("="*70)
        print(f"Red Agent: {self.red_agent_type.__name__}")
        print(f"Max Steps: {self.max_steps}")
        print(f"Train Episodes per Workflow: {self.train_episodes}")
        print(f"Eval Episodes: {self.eval_episodes}")
        print(f"Alignment Rewards: α={self.alignment_alpha}, β={self.alignment_beta}")
        print("="*70)
        
        for iteration in range(n_iterations):
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration + 1}/{n_iterations}")
            print(f"{'='*50}")
            
            # 1. Select next workflow using GP-UCB
            workflow_order, workflow_vector = self.gp_search.select_next_workflow()
            
            # 2. Train PPO with this workflow
            train_reward, train_compliance, episodes_trained = self.train_workflow(
                workflow_order, workflow_vector, iteration
            )
            
            # 3. Evaluate trained agent
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"workflow_{iteration}_agent.pth"
            )
            eval_reward, eval_compliance = self.evaluate_workflow(
                workflow_order, workflow_vector, checkpoint_path
            )
            
            # 4. Update GP-UCB
            self.gp_search.update(
                workflow_order, workflow_vector, 
                eval_reward, eval_compliance
            )
            
            # 5. Record results
            result = {
                'iteration': iteration,
                'workflow': workflow_order,
                'train_reward': train_reward,
                'train_compliance': train_compliance,
                'eval_reward': eval_reward,
                'eval_compliance': eval_compliance,
                'episodes_trained': episodes_trained
            }
            self.training_history.append(result)
            
            # 6. Print results
            print(f"\nResults:")
            print(f"  Workflow: {' → '.join(workflow_order)}")
            print(f"  Training: Reward={train_reward:.2f}, Compliance={train_compliance:.2%}, Episodes={episodes_trained}")
            print(f"  Evaluation: Reward={eval_reward:.2f}, Compliance={eval_compliance:.2%}")
            
            # 7. Print best so far
            best_workflow, best_reward = self.gp_search.get_best_workflow()
            print(f"\nBest workflow so far: {' → '.join(best_workflow)}")
            print(f"Best reward: {best_reward:.2f}")
            
            # 8. Print search statistics
            stats = self.gp_search.get_statistics()
            print(f"\nSearch statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save training results"""
        
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save best workflow
        best_workflow, best_reward = self.gp_search.get_best_workflow()
        best_result = {
            'workflow': best_workflow,
            'reward': best_reward,
            'training_history': self.training_history
        }
        
        best_path = os.path.join(self.checkpoint_dir, "best_workflow.json")
        with open(best_path, 'w') as f:
            json.dump(best_result, f, indent=2)
        
        print(f"\nResults saved to {self.checkpoint_dir}")


def main():
    """Main training function"""
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Get scenario path
    path = str(inspect.getfile(CybORG))
    scenario_path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    
    # Create trainer
    trainer = WorkflowRLTrainer(
        scenario_path=scenario_path,
        red_agent_type=B_lineAgent,
        max_steps=50,  # Shorter episodes for faster training
        eval_episodes=3,
        train_episodes=20,  # Fewer episodes for demo
        alignment_alpha=0.1,
        alignment_beta=0.2,
        checkpoint_dir=f"workflow_rl_checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run workflow search
    trainer.run_workflow_search(n_iterations=10)  # 10 iterations for demo
    
    # Print final results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    best_workflow, best_reward = trainer.gp_search.get_best_workflow()
    print(f"Best workflow found: {' → '.join(best_workflow)}")
    print(f"Best average reward: {best_reward:.2f}")
    
    # Show top 5 workflows
    print("\nTop 5 workflows:")
    workflows_rewards = list(zip(trainer.gp_search.observed_workflows, 
                                trainer.gp_search.observed_rewards))
    workflows_rewards.sort(key=lambda x: x[1], reverse=True)
    
    for i, (workflow, reward) in enumerate(workflows_rewards[:5]):
        print(f"{i+1}. {' → '.join(workflow)}: {reward:.2f}")


if __name__ == "__main__":
    main()